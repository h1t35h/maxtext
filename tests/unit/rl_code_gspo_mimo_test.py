# Copyright 2023–2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for V2 GSPO-MiMo MaxText configs, trainer plumbing, diagnostics, and study scripts."""

from __future__ import annotations

import functools
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from flax import nnx
import jax
import numpy as np
import optax

from maxtext.configs import pyconfig
from maxtext.configs import types
from maxtext.trainers.post_train.rl import train_rl
from maxtext.trainers.post_train.rl.code_rl import metrics as code_rl_metrics
from maxtext.trainers.post_train.rl.code_rl.scripts import calibrate_logp_mismatch
from maxtext.trainers.post_train.rl.code_rl.scripts import eval_code_checkpoints
from maxtext.trainers.post_train.rl.code_rl.scripts import run_diagnostics_d1_d4
from maxtext.trainers.post_train.rl.code_rl.scripts import run_v2_study
from maxtext.utils.globals import MAXTEXT_CONFIGS_DIR

POST_TRAIN_CONFIG_DIR = Path(MAXTEXT_CONFIGS_DIR) / "post_train"

V2_CANONICAL_CONFIG_NAMES = (
    "rl_code_gspo_mimo.yml",
    "rl_code_r1p_gspo_scalar.yml",
    "rl_code_r2p_gspo_mimo.yml",
    "rl_code_r2pp_turn_gspo_mimo.yml",
    "rl_code_r3p_mimo_grpo.yml",
    "rl_code_r3p_seq_mimo_grpo.yml",
)


class RlCodeGspoMimoV2Test(unittest.TestCase):
  """Tests V2 MaxText schema, YAML configs, trainer wiring, diagnostics, and statistical evaluator."""

  def _load_rl_config(
      self, config_filename: str, extra_args: list[str] | None = None
  ) -> pyconfig.HyperParameters:
    config_path = str(POST_TRAIN_CONFIG_DIR / config_filename)
    argv = ["", config_path]
    if extra_args:
      argv.extend(extra_args)
    return pyconfig.initialize(argv, config_class=types.RLConfig)

  def test_v2_types_schema_and_flat_field_mapping(self) -> None:
    mimo = types.MimoConfig()
    self.assertAlmostEqual(mimo.span_beta_pos, 0.5)
    self.assertTrue(mimo.span_renormalize)

    rl = types.RL()
    self.assertEqual(rl.gspo_pool_mode, "sequence")
    self.assertIsNone(rl.compute_logps_micro_batch_size)
    self.assertEqual(rl.judge_max_flagged_segments_per_turn, 2)
    self.assertIn("gspo_pool_mode", types._RL_NESTED_FIELDS)
    self.assertIn("compute_logps_micro_batch_size", types._RL_NESTED_FIELDS)
    self.assertIn("judge_max_flagged_segments_per_turn", types._RL_NESTED_FIELDS)

    cfg = self._load_rl_config(
        "rl_code_r2pp_turn_gspo_mimo.yml",
        extra_args=[
            "compute_logps_micro_batch_size=4",
            "judge_max_flagged_segments_per_turn=2",
        ],
    )
    self.assertEqual(cfg.rl.loss_algo, "gspo-turn")
    self.assertEqual(cfg.rl.gspo_pool_mode, "turn")
    self.assertEqual(cfg.rl.compute_logps_micro_batch_size, 4)
    self.assertEqual(cfg.rl.judge_max_flagged_segments_per_turn, 2)
    self.assertAlmostEqual(cfg.rl.mimo.span_beta_pos, 0.5)
    self.assertTrue(cfg.rl.mimo.span_renormalize)

  def test_all_v2_canonical_yaml_configs_load_and_build_tunix_configs(self) -> None:
    expected_settings = {
        "rl_code_gspo_mimo.yml": ("gspo-token", "sequence", "token-mean", "mimo", "mimo"),
        "rl_code_r1p_gspo_scalar.yml": ("gspo-token", "sequence", "sequence-mean-token-mean", "grpo", "none"),
        "rl_code_r2p_gspo_mimo.yml": ("gspo-token", "sequence", "sequence-mean-token-mean", "mimo", "mimo"),
        "rl_code_r2pp_turn_gspo_mimo.yml": ("gspo-turn", "turn", "sequence-mean-token-mean", "mimo", "mimo"),
        "rl_code_r3p_mimo_grpo.yml": ("grpo", "sequence", "token-mean", "mimo", "mimo"),
        "rl_code_r3p_seq_mimo_grpo.yml": ("grpo", "sequence", "sequence-mean-token-mean", "mimo", "mimo"),
    }
    for cfg_name in V2_CANONICAL_CONFIG_NAMES:
      with self.subTest(config=cfg_name):
        cfg = self._load_rl_config(cfg_name)
        self.assertEqual(cfg.rl.off_policy_steps, 0)
        self.assertEqual(cfg.train_micro_batch_size, 2)
        self.assertEqual(cfg.rollout_micro_batch_size, 2)
        self.assertEqual(cfg.rl.mini_batch_size, 2)
        mimo_cfg = train_rl._build_mimo_config(cfg)  # pylint: disable=protected-access
        self.assertIsNotNone(mimo_cfg)
        self.assertAlmostEqual(mimo_cfg.span_beta_pos, 0.5)
        self.assertTrue(mimo_cfg.span_renormalize)

        training_cfg, grpo_cfg = train_rl.build_tunix_rl_configs(cfg)
        self.assertIsInstance(training_cfg, train_rl.rl_cluster_lib.RLTrainingConfig)
        self.assertEqual(training_cfg.train_micro_batch_size, 2)
        self.assertEqual(training_cfg.rollout_micro_batch_size, 2)
        self.assertEqual(training_cfg.mini_batch_size, 2)

        exp_algo, exp_pool, exp_agg, exp_adv_est, exp_adv_mode = expected_settings[cfg_name]
        self.assertEqual(grpo_cfg.loss_algo, exp_algo)
        self.assertEqual(grpo_cfg.gspo_pool_mode, exp_pool)
        self.assertEqual(grpo_cfg.loss_agg_mode, exp_agg)
        self.assertEqual(grpo_cfg.advantage_estimator, exp_adv_est)
        self.assertEqual(grpo_cfg.token_advantage_mode, exp_adv_mode)

  def test_train_rl_plumbing_forwards_v2_fields_and_dynamic_max_response_length(self) -> None:
    captured_cluster_kwargs: dict[str, object] = {}
    captured_learner_kwargs: dict[str, object] = {}
    captured_postproc_kwargs: dict[str, object] = {}

    class FakeEnv:
      pass

    class FakePostProcessor:

      def __init__(
          self,
          tokenizer=None,
          mimo_config=None,
          max_faulty_segments_per_turn: int = 2,
          max_flagged_segments_per_turn: int = 2,
      ):
        captured_postproc_kwargs["tokenizer"] = tokenizer
        captured_postproc_kwargs["mimo_config"] = mimo_config
        captured_postproc_kwargs["max_faulty_segments_per_turn"] = max_faulty_segments_per_turn
        captured_postproc_kwargs["max_flagged_segments_per_turn"] = max_flagged_segments_per_turn

    trainer_config = SimpleNamespace(
        enable_checkpointing=False,
        train_micro_batch_size=2,
        rollout_micro_batch_size=2,
        tensorboard_dir="/tmp/tb",
        log_period=1,
        profiler="",
        vllm_additional_config=None,
        vllm_hf_overrides={},
        weight_dtype=SimpleNamespace(value="bfloat16"),
        vllm_block_size=None,
        batch_size=8,
        decode_sampling_top_k=0,
        decode_sampling_temperature=1.0,
        decode_sampling_nucleus_p=1.0,
        logical_axis_rules=[],
        gc_collect_after_weight_sync=False,
        eval_interval=10,
        train_steps=30,
        max_seq_token_per_tpu=0,
        max_target_length=4608,
        max_prefill_predict_length=1536,
        kv_cache_buffer=256,
        tokenizer_path="Qwen/Qwen3-4B",
        hbm_utilization_vllm=0.72,
        vllm_hf_config_path="",
        enable_dp_attention=False,
        max_num_batched_tokens=4096,
        max_num_seqs=64,
        async_scheduling=False,
        stop_strings=None,
        free_kv_cache_during_weight_sync=False,
        enable_tunix_perf_metrics=False,
        reward_num_workers=0,
        reward_worker_timeout_seconds=30.0,
        data_template_path="",
        rl=SimpleNamespace(
            use_agentic_rollout=True,
            num_generations=8,
            num_iterations=1,
            mini_batch_size=2,
            compute_logps_micro_batch_size=2,
            grpo_beta=0.0,
            grpo_epsilon=3e-4,
            epsilon_high=4e-4,
            epsilon_neg=None,
            epsilon_neg_high=None,
            loss_algo="gspo-turn",
            gspo_pool_mode="turn",
            loss_agg_mode="sequence-mean-token-mean",
            max_concurrency=64,
            off_policy_steps=0,
            system_prompt="",
            reshard_chunk_size=None,
            use_rollout_logps=False,
            force_on_policy_ratio=False,
            log_sampler_trainer_agreement=False,
            advantage_estimator="mimo",
            clip_mode="ppo",
            token_advantage_mode="mimo",
            mimo=types.MimoConfig(span_beta_pos=0.5, span_renormalize=True),
            exact_token_continuity=True,
            sampler_is="token",
            sampler_is_threshold=2.0,
            sampler_rs=None,
            sampler_rs_min=None,
            sampler_rs_max=None,
            seq_logprob_error_threshold=None,
            env_class_path="fake.module:FakeEnv",
            env_kwargs={"max_steps": 3},
            group_postprocessor_path="fake.module:FakePostProcessor",
            judge_base_url="",
            judge_model="Qwen/Qwen3-32B",
            judge_max_in_flight_groups=8,
            judge_timeout_s=60.0,
            judge_max_flagged_segments_per_turn=2,
        ),
    )

    def _load_fake(path: str):
      if "FakeEnv" in path:
        return FakeEnv
      return FakePostProcessor

    with (
        mock.patch.object(train_rl.utils_rl, "get_optimizer", return_value=optax.adamw(1e-6)),
        mock.patch.object(train_rl, "build_reward_fns", return_value=[]),
        mock.patch.object(train_rl.utils_rl, "load_custom_callable", side_effect=_load_fake),
        mock.patch.object(train_rl, "rollout_prefix_caching_enabled", return_value=False),
        mock.patch.object(
            train_rl.pyconfig,
            "initialize",
            return_value=SimpleNamespace(logical_axis_rules=[]),
        ),
        mock.patch.object(train_rl.base_rollout, "RolloutConfig", side_effect=lambda **kw: SimpleNamespace(**kw)),
        mock.patch.object(
            train_rl.rl_cluster_lib,
            "ClusterConfig",
            side_effect=lambda **kw: captured_cluster_kwargs.update(kw) or SimpleNamespace(**kw),
        ),
        mock.patch.object(train_rl.rl_cluster_lib, "RLCluster"),
        mock.patch.object(train_rl, "get_rollout_kwargs_for_parallelism", return_value={}),
        mock.patch(
            "tunix.rl.agentic.agentic_grpo_learner.GrpoLearner",
            side_effect=lambda **kw: captured_learner_kwargs.update(kw) or SimpleNamespace(**kw),
        ),
    ):
      train_rl.create_rl_components(
          trainer_config=trainer_config,
          sampler_config=SimpleNamespace(enable_expert_parallel=False),
          sampler_devices=[object()],
          actor_model=object(),
          actor_mesh=object(),
          reference_model=None,
          reference_mesh=None,
          rollout_mesh=object(),
          model_tokenizer="fake_tok",
      )

    training_cfg = captured_cluster_kwargs["training_config"]
    self.assertIsInstance(training_cfg, train_rl.rl_cluster_lib.RLTrainingConfig)
    self.assertEqual(training_cfg.compute_logps_micro_batch_size, 2)
    self.assertEqual(training_cfg.mini_batch_size, 2)
    self.assertEqual(training_cfg.train_micro_batch_size, 2)
    self.assertEqual(training_cfg.rollout_micro_batch_size, 2)

    algo_cfg = captured_learner_kwargs["algo_config"]
    self.assertEqual(algo_cfg.loss_algo, "gspo-turn")
    self.assertEqual(algo_cfg.gspo_pool_mode, "turn")
    self.assertEqual(algo_cfg.loss_agg_mode, "sequence-mean-token-mean")
    self.assertEqual(algo_cfg.max_response_length, 3072)
    self.assertEqual(captured_postproc_kwargs["max_faulty_segments_per_turn"], 2)
    self.assertEqual(captured_postproc_kwargs["max_flagged_segments_per_turn"], 2)

    metric_fns = captured_learner_kwargs["metric_fns"]
    self.assertEqual(len(metric_fns), 1)
    bound_fn = metric_fns[0]
    self.assertIsInstance(bound_fn, functools.partial)
    self.assertEqual(bound_fn.keywords.get("max_response_length"), 3072)

    # Verify non-agentic GrpoConfig branch also forwards V2 fields cleanly
    trainer_config.rl.use_agentic_rollout = False
    _, non_agentic_cfg = train_rl.build_tunix_rl_configs(trainer_config)
    self.assertIsInstance(non_agentic_cfg, train_rl.GrpoConfig)
    self.assertEqual(non_agentic_cfg.loss_algo, "gspo-turn")
    self.assertEqual(non_agentic_cfg.gspo_pool_mode, "turn")
    self.assertEqual(non_agentic_cfg.loss_agg_mode, "sequence-mean-token-mean")
    self.assertEqual(non_agentic_cfg.advantage_estimator, "mimo")
    self.assertEqual(non_agentic_cfg.token_advantage_mode, "mimo")

    # Verify bound metric function evaluates truncation at 3072 rather than 8192
    metrics_out = bound_fn(
        rewards=[1.0, 0.0],
        turn1_passed=[False, False],
        final_passed=[True, False],
        assistant_lengths=[3100, 1200],
    )
    self.assertAlmostEqual(float(metrics_out["code_rl/truncation_rate"][0]), 0.5)
    self.assertAlmostEqual(float(metrics_out["length/truncation_rate"][0]), 0.5)

  def test_drift_calibration_and_gate_g2_prime(self) -> None:
    rng = np.random.default_rng(42)
    mb1_ratios = rng.normal(loc=0.0, scale=1e-6, size=16)
    mb2_to_4_ratios = rng.normal(loc=0.0, scale=2.0e-4, size=48)
    assistant_lens = rng.integers(400, 2200, size=48)

    report = calibrate_logp_mismatch.calibrate_drift_epsilon(
        mb2_to_4_ratios,
        mb1_log_ratios=mb1_ratios,
        target_clip_range=(0.05, 0.10),
        max_response_length=3072,
        assistant_lengths=assistant_lens,
    )
    self.assertLess(report["mb1_abs_log_ratio"], 1e-4)
    self.assertLess(report["mb1_pg_clipfrac"], 0.01)
    self.assertGreaterEqual(report["mb2_to_4_pg_clipfrac"], 0.05)
    self.assertLessEqual(report["mb2_to_4_pg_clipfrac"], 0.10)
    self.assertLess(report["truncation_rate"], 0.10)
    self.assertAlmostEqual(
        report["recommended_epsilon_h"], report["recommended_epsilon_l"] * (4.0 / 3.0)
    )

    gate_g2p = calibrate_logp_mismatch.evaluate_gate_g2_prime(
        report, usable_pool_size=2400
    )
    self.assertTrue(gate_g2p["passed"])

  def test_diagnostics_d1_to_d4_under_multidevice_cpu_mesh(self) -> None:
    devices = jax.devices()
    num_devices = len(devices)
    batch_size = max(8, 4 * num_devices)
    micro_batch_size = max(4, num_devices)

    d1 = run_diagnostics_d1_d4.run_d1_logp_determinism(
        batch_size=batch_size,
        micro_batch_size=micro_batch_size,
        seed=42,
    )
    self.assertTrue(d1["v2_determinism_verified"])
    self.assertIn("comparisons", d1)
    self.assertEqual(
        set(d1["comparisons"].keys()),
        {"a_vs_b_v1", "a_vs_b_v2", "c_identical_shape", "d_trainer_vs_rollout"},
    )
    self.assertLess(d1["v2_aligned_microbatch_mean_abs_seq_log_ratio"], 1e-4)
    self.assertLess(d1["v2_mb1_pg_clipfrac_at_3e4"], 0.01)

    d2 = run_diagnostics_d1_d4.run_d2_r2_counterfactual(
        batch_size=batch_size,
        seed=42,
    )
    self.assertTrue(d2["counterfactual_verified"])
    self.assertIn("epsilon_sweep", d2)
    self.assertEqual(len(d2["epsilon_sweep"]), 3)
    self.assertLess(d2["v2_r2p_mb1_pg_clipfrac"], 0.01)

    d3 = run_diagnostics_d1_d4.run_d3_credit_assignment_audit(num_trajectories=50)
    self.assertTrue(d3["selectivity_in_target_band"])
    self.assertGreaterEqual(d3["v2_flagged_token_frac"], 0.15)
    self.assertLessEqual(d3["v2_flagged_token_frac"], 0.35)
    self.assertEqual(d3["v2_passing_turn_code_fp_rate"], 0.0)
    self.assertEqual(d3["tokenizer"], "Qwen/Qwen3-4B")

    d4 = run_diagnostics_d1_d4.run_d4_mass_conservation_audit()
    self.assertTrue(d4["mass_conservation_verified"])
    self.assertGreaterEqual(d4["v2_min_mass_pos_ratio"], 0.95)
    self.assertLessEqual(d4["v2_max_mass_pos_ratio"], 1.05)
    self.assertLess(d4["v2_max_abs_advantage_mean"], 0.03)

    with tempfile.TemporaryDirectory() as tmpdir:
      out_dir = Path(tmpdir) / "diagnostics"
      summary = run_diagnostics_d1_d4.run_all_diagnostics(out_dir, seed=42)
      self.assertTrue(summary["all_passed"])
      for stem in (
          "d1_logp_determinism",
          "d2_r2_counterfactual",
          "d3_credit_assignment_audit",
          "d4_mass_conservation_audit",
      ):
        self.assertTrue((out_dir / f"{stem}.json").exists())
        self.assertTrue((out_dir / f"{stem}.md").exists())

  def test_multiseed_statistics_and_holm_bonferroni_correction(self) -> None:
    seed_reports_r2p = [
        {
            "seed": 42,
            "turn1_solve": 0.50,
            "final_solve": 0.65,
            "repair_gain": 0.15,
            "mean_assistant_tokens": 1150.0,
            "p90_assistant_tokens": 1800.0,
            "truncation_rate": 0.02,
            "problem_scores": [1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0],
        },
        {
            "seed": 43,
            "turn1_solve": 0.52,
            "final_solve": 0.68,
            "repair_gain": 0.16,
            "mean_assistant_tokens": 1180.0,
            "p90_assistant_tokens": 1850.0,
            "truncation_rate": 0.03,
            "problem_scores": [1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0],
        },
        {
            "seed": 44,
            "turn1_solve": 0.48,
            "final_solve": 0.62,
            "repair_gain": 0.14,
            "mean_assistant_tokens": 1120.0,
            "p90_assistant_tokens": 1750.0,
            "truncation_rate": 0.01,
            "problem_scores": [1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0],
        },
    ]
    seed_reports_r3p = [
        {
            "seed": s,
            "turn1_solve": 0.45,
            "final_solve": 0.50,
            "repair_gain": 0.05,
            "mean_assistant_tokens": 1250.0,
            "p90_assistant_tokens": 1950.0,
            "truncation_rate": 0.04,
            "problem_scores": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        }
        for s in (42, 43, 44)
    ]

    summary_r2p = eval_code_checkpoints.summarize_multiseed_arm(seed_reports_r2p)
    self.assertEqual(summary_r2p["num_seeds"], 3)
    self.assertAlmostEqual(summary_r2p["final_solve_mean"], 0.65)
    self.assertGreater(summary_r2p["final_solve_std"], 0.0)
    self.assertEqual(len(summary_r2p["mean_problem_scores"]), 10)
    self.assertEqual(len(summary_r2p["pooled_problem_scores"]), 30)

    raw_comps = [
        {"name": "c1", "p_value": 0.005},
        {"name": "c2", "p_value": 0.020},
        {"name": "c3", "p_value": 0.040},
    ]
    corrected = eval_code_checkpoints.holm_bonferroni_correction(raw_comps, alpha=0.05)
    self.assertAlmostEqual(corrected[0]["p_holm"], 0.015)
    self.assertTrue(corrected[0]["significant_holm"])
    self.assertAlmostEqual(corrected[1]["p_holm"], 0.040)
    self.assertTrue(corrected[1]["significant_holm"])
    self.assertAlmostEqual(corrected[2]["p_holm"], 0.040)
    self.assertTrue(corrected[2]["significant_holm"])

    stats_out = eval_code_checkpoints.evaluate_v2_study_statistics(
        {
            "R2p": {"held_out": seed_reports_r2p, "lcb_v5": seed_reports_r2p},
            "R3p": {"held_out": seed_reports_r3p, "lcb_v5": seed_reports_r3p},
        },
        num_resamples=1000,
        seed=42,
    )
    self.assertIsNotNone(stats_out["primary_comparison"])
    self.assertEqual(stats_out["primary_comparison"]["num_paired_samples"], 10)
    self.assertGreater(stats_out["primary_comparison"]["mean_diff"], 0.0)
    self.assertTrue(stats_out["truncation_audit"]["all_below_10pct"])

  def test_run_v2_study_actor_reset_and_jitted_step_cache(self) -> None:
    model = nnx.Linear(4, 4, rngs=nnx.Rngs(0))
    init_state = jax.tree_util.tree_map(
        lambda x: np.asarray(jax.device_get(x)).copy(),
        nnx.to_pure_dict(nnx.state(model, nnx.Param)),
    )
    # Perturb model weights
    cur_state = nnx.state(model, nnx.Param)
    cur_pure = nnx.to_pure_dict(cur_state)
    perturbed = jax.tree_util.tree_map(lambda x: x + 5.0, cur_pure)
    nnx.replace_by_pure_dict(cur_state, perturbed)
    nnx.update(model, cur_state)

    sentinel_jitted_fn = object()
    fake_trainer = SimpleNamespace(
        model=model,
        _lora_enabled=False,
        optimizer=nnx.Optimizer(model, optax.adamw(1e-6), wrt=nnx.Param),
        grad_accumulator=None,
        _jitted_train_step_fn=sentinel_jitted_fn,
        _jitted_eval_step_fn=None,
        _train_steps=10,
        _iter_steps=10,
        _buffered_train_metrics=None,
        _prev_buffered_train_metrics=None,
    )
    fake_cluster = SimpleNamespace(
        actor_trainer=fake_trainer,
        global_steps=10,
        _buffered_train_metrics=[],
        _buffered_eval_metrics=[],
        sync_weights=lambda: None,
    )

    cache: dict[tuple[object, ...], object] = {}
    key_r2p = run_v2_study.get_loss_compile_key("R2p")
    # Seed 42 -> Seed 43 of same arm R2p should cache and reuse _jitted_train_step_fn
    returned_key = run_v2_study.reset_actor_for_arm_seed(
        fake_cluster,
        model,
        init_state,
        arm_name="R2p",
        jitted_step_cache=cache,
        prev_compile_key=key_r2p,
    )
    self.assertEqual(returned_key, key_r2p)
    self.assertIs(fake_trainer._jitted_train_step_fn, sentinel_jitted_fn)
    restored_pure = nnx.to_pure_dict(nnx.state(model, nnx.Param))
    np.testing.assert_allclose(
        np.asarray(restored_pure["kernel"]),
        np.asarray(init_state["kernel"]),
        rtol=1e-6,
        atol=1e-6,
    )

    # Verify run_gates_and_diagnostics_phase requires cluster_ctx unless offline_unit_test=True
    with tempfile.TemporaryDirectory() as tmpdir:
      out_dir = Path(tmpdir)
      with self.assertRaises(RuntimeError):
        run_v2_study.run_gates_and_diagnostics_phase(
            out_dir, cluster_ctx=None, offline_unit_test=False
        )
      gates_report = run_v2_study.run_gates_and_diagnostics_phase(
          out_dir, cluster_ctx=None, offline_unit_test=True
      )
      self.assertTrue(gates_report["diagnostics"]["all_passed"])
      self.assertTrue(gates_report["gate_g1_prime"]["passed"])
      self.assertTrue(gates_report["gate_g2_prime"]["passed"])

  def test_evaluate_multi_turn_benchmark_strict_cluster_asserts_and_stdio(self) -> None:
    """Verifies evaluate_multi_turn_benchmark against strict RLCluster.generate signature, entry_point, and stdio dict tests."""
    rollout_cfg = SimpleNamespace(temperature=0.7)
    generate_calls: list[dict[str, object]] = []

    class _StrictCluster:
      cluster_config = SimpleNamespace(rollout_config=rollout_cfg)

      def get_rollout_config(self, mode=None):
        del mode
        return rollout_cfg

      def generate(
          self,
          prompts=None,
          apply_chat_template: bool = False,
          mode=None,
          micro_batch_size: int | None = None,
          trace_tags=None,
          max_generation_steps: int | None = None,
          *,
          prompt_token_ids=None,
      ):
        del trace_tags
        # Record temperature during generate call and verify strict signature (no max_tokens_to_generate or temperature kwargs)
        generate_calls.append({
            "num_prompts": len(prompts or []),
            "apply_chat_template": apply_chat_template,
            "mode": mode,
            "micro_batch_size": micro_batch_size,
            "max_generation_steps": max_generation_steps,
            "temperature_during_call": rollout_cfg.temperature,
            "prompt_token_ids": prompt_token_ids,
        })
        out_texts: list[str] = []
        for p in prompts or []:
          if "Write add(a, b)" in p:
            out_texts.append("```python\ndef add(a, b):\n    return a + b\n```")
          elif "Execution failed" in p:
            out_texts.append("```python\nimport sys\nprint(sys.stdin.read().strip())\n```")
          else:
            out_texts.append("```python\nprint('wrong')\n```")
        return SimpleNamespace(text=out_texts)

    class _MinimalTok:

      def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [1] * max(1, len(text.split()))

    problems = [
        {
            "problem": "Write add(a, b)",
            "tests": ["assert candidate(1, 2) == 3", "assert candidate(-1, 5) == 4"],
            "format": "asserts",
            "entry_point": "add",
        },
        {
            "problem": "Echo stdin",
            "tests": {"format": "stdio", "inputs": ["5\n"], "outputs": ["5\n"]},
            "format": "stdio",
        },
    ]
    res = run_v2_study.evaluate_multi_turn_benchmark(
        _StrictCluster(),
        _MinimalTok(),
        problems,
        "unit_bench",
        "R2p",
        max_turns=2,
        max_generation_steps=1024,
    )
    self.assertAlmostEqual(res["turn1_solve"], 0.5)
    self.assertAlmostEqual(res["final_solve"], 1.0)
    self.assertAlmostEqual(res["repair_gain"], 0.5)
    self.assertEqual(res["problem_scores"], [1.0, 1.0])
    self.assertEqual(res["turn1_scores"], [1.0, 0.0])
    self.assertEqual(len(generate_calls), 2)
    self.assertEqual(generate_calls[0]["max_generation_steps"], 1024)
    self.assertAlmostEqual(float(generate_calls[0]["temperature_during_call"]), 0.0)
    # Verify original rollout temperature is restored in finally block
    self.assertAlmostEqual(rollout_cfg.temperature, 0.7)

  def test_unwrapped_python_code_env_with_ndarray_microbatch(self) -> None:
    """Verifies UnwrappedPythonCodeEnv / PythonCodeEnv with 1-element np.ndarray micro-batches and group_id/pair_index."""
    from tunix.rl.agentic.environments.code import code_env as code_env_mod

    env_cls = getattr(run_v2_study, "UnwrappedPythonCodeEnv", code_env_mod.PythonCodeEnv)
    single_example = {
        "prompts": np.array(["Write a function add(a, b)"]),
        "problem": np.array(["Write a function add(a, b)"]),
        "tests": np.array([json.dumps(["assert candidate(1, 2) == 3"])]),
        "format": np.array(["asserts"]),
        "entry_point": np.array(["add"]),
    }
    env = env_cls(single_example, group_id=5, pair_index=2)
    self.assertEqual(env.extra_kwargs.get("group_id"), 5)
    self.assertEqual(env.extra_kwargs.get("pair_index"), 2)

    obs, _ = env.reset()
    self.assertIsInstance(obs.get("prompts"), str)
    self.assertIsInstance(obs.get("tests"), str)

    # Turn 1: buggy implementation must actually run assertion and fail (reward == 0.0)
    step1 = env.step("```python\ndef add(a, b):\n    return a - b\n```")
    self.assertAlmostEqual(float(step1.reward), 0.0)
    self.assertFalse(step1.done)

    # Turn 2: fixed implementation passes assertion via entry_point='add' (reward == 1.0)
    step2 = env.step("```python\ndef add(a, b):\n    return a + b\n```")
    self.assertAlmostEqual(float(step2.reward), 1.0)
    self.assertTrue(step2.done)

  def test_load_all_benchmarks_jsonl_and_hf_arrow_fallback(self) -> None:
    """Verifies load_all_benchmarks loads both JSONL files and HuggingFace load_from_disk benchmark directories."""
    import base64
    import zlib
    import datasets

    with tempfile.TemporaryDirectory() as tmpdir:
      root = Path(tmpdir)
      # 1. Held-out as JSONL
      heldout_path = root / "heldout_eval.jsonl"
      heldout_path.write_text(
          json.dumps({
              "prompt_id": "ho_1",
              "problem": "Write sq(x).",
              "tests": ["assert sq(3) == 9"],
              "format": "asserts",
              "entry_point": "sq",
          })
          + "\n",
          encoding="utf-8",
      )

      # 2. HumanEval+ as HF Arrow dataset directory (`humanevalplus/test`)
      he_ds = datasets.Dataset.from_list([
          {
              "task_id": "HumanEval/0",
              "prompt": 'def incr(x: int) -> int:\n    """Return x + 1."""\n',
              "test": "def check(candidate):\n    assert candidate(2) == 3\n",
              "entry_point": "incr",
          }
      ])
      he_ds.save_to_disk(str(root / "humanevalplus" / "test"))

      # 3. MBPP+ as HF Arrow dataset directory (`mbppplus/test`)
      mbpp_ds = datasets.Dataset.from_list([
          {
              "task_id": 11,
              "prompt": "Write a function cube(n) that returns n**3.",
              "test_list": ["assert cube(2) == 8"],
              "test_imports": ["import math"],
          }
      ])
      mbpp_ds.save_to_disk(str(root / "mbppplus" / "test"))

      # 4. LiveCodeBench as HF Arrow dataset directory (`livecodebench_code_generation_lite/test`)
      lcb_private_encoded = base64.b64encode(
          zlib.compress(json.dumps([{"input": "7\n", "output": "14\n"}]).encode("utf-8"))
      ).decode("utf-8")
      lcb_ds = datasets.Dataset.from_list([
          {
              "question_id": "lcb_1",
              "question_content": "Read integer n and print 2*n.",
              "starter_code": "",
              "public_test_cases": json.dumps([{"input": "3\n", "output": "6\n"}]),
              "private_test_cases": lcb_private_encoded,
              "metadata": json.dumps({}),
          }
      ])
      lcb_ds.save_to_disk(str(root / "livecodebench_code_generation_lite" / "test"))

      benches = run_v2_study.load_all_benchmarks(root)
      self.assertEqual(len(benches.get("held_out", [])), 1)
      self.assertEqual(len(benches.get("humaneval_plus", [])), 1)
      self.assertEqual(benches["humaneval_plus"][0]["entry_point"], "incr")
      self.assertEqual(len(benches.get("mbpp_plus", [])), 1)
      self.assertEqual(len(benches.get("lcb_v5", [])), 1)
      lcb_item = benches["lcb_v5"][0]
      self.assertEqual(lcb_item["format"], "stdio")
      self.assertIsInstance(lcb_item["tests"], dict)
      self.assertEqual(len(lcb_item["tests"]["inputs"]), 2)

  def test_d1_compute_mb_logps_via_cluster_and_d2_default_telemetry_path(self) -> None:
    """Verifies _compute_mb_logps_via_cluster and run_d1_logp_determinism with strict RLEngine signatures and D2 default path."""
    actor_calls: list[dict[str, object]] = []
    old_calls: list[dict[str, object]] = []

    class _StrictLogpCluster:
      _anchor_policy_state = None
      synced = False

      def sync_weights(self) -> None:
        self.synced = True
        self._anchor_policy_state = {"initialized": True}

      def get_actor_per_token_logps(
          self,
          prompt_tokens,
          completion_tokens,
          pad_id: int,
          eos_id: int,
          micro_batch_size: int | None = None,
          temperature: float | None = None,
          segment_ids=None,
          segment_positions=None,
          token_mask=None,
      ):
        del pad_id, eos_id, temperature, segment_ids, segment_positions
        actor_calls.append({
            "micro_batch_size": micro_batch_size,
            "has_token_mask": token_mask is not None,
        })
        base = -0.5 * np.ones(completion_tokens.shape, dtype=np.float32)
        if token_mask is None:
          # Simulate V1 unmasked attention shift on the same cluster model
          return base - 0.02
        return base

      def get_old_per_token_logps(
          self,
          prompt_tokens,
          completion_tokens,
          micro_batch_size: int | None = None,
      ):
        # Strictly matches RLEngine.get_old_per_token_logps (no token_mask parameter!)
        old_calls.append({"micro_batch_size": micro_batch_size})
        del prompt_tokens
        return -0.501 * np.ones(completion_tokens.shape, dtype=np.float32)

    cluster = _StrictLogpCluster()
    d1 = run_diagnostics_d1_d4.run_d1_logp_determinism(
        batch_size=8,
        micro_batch_size=4,
        prompt_len=16,
        completion_len=16,
        seed=42,
        rl_cluster=cluster,
    )
    self.assertTrue(cluster.synced)
    self.assertGreaterEqual(len(actor_calls), 4)
    # Condition (a) called with token_mask=None, micro_batch_size=8
    self.assertTrue(
        any(c["micro_batch_size"] == 8 and not c["has_token_mask"] for c in actor_calls)
    )
    # Condition (b)/(c) called with token_mask!=None, micro_batch_size=4
    self.assertTrue(
        any(c["micro_batch_size"] == 4 and c["has_token_mask"] for c in actor_calls)
    )
    # Condition (d) called get_old_per_token_logps without raising TypeError on token_mask
    self.assertGreaterEqual(len(old_calls), 1)
    self.assertTrue(d1["root_cause_confirmed"])
    self.assertTrue(d1["v2_determinism_verified"])
    self.assertAlmostEqual(d1["v2_aligned_microbatch_mean_abs_seq_log_ratio"], 0.0)

    # Also test fallback when cluster only exposes strict get_old_per_token_logps(prompt_tokens, completion_tokens, micro_batch_size=None)
    class _OldOnlyCluster:

      def get_old_per_token_logps(
          self, prompt_tokens, completion_tokens, micro_batch_size: int | None = None
      ):
        del prompt_tokens, micro_batch_size
        return -0.25 * np.ones(completion_tokens.shape, dtype=np.float32)

    out_old = run_diagnostics_d1_d4._compute_mb_logps_via_cluster(  # pylint: disable=protected-access
        _OldOnlyCluster(),
        jax.numpy.ones((4, 8), dtype=jax.numpy.int32),
        jax.numpy.ones((4, 8), dtype=jax.numpy.int32),
        micro_batch_size=2,
        token_mask=jax.numpy.ones((4, 16), dtype=jax.numpy.int32),
    )
    self.assertEqual(out_old.shape, (4, 8))

    # Verify _DEFAULT_V1_R2_TELEMETRY_PATH resolves to existing V1 R2 telemetry regardless of cwd
    self.assertTrue(run_diagnostics_d1_d4._DEFAULT_V1_R2_TELEMETRY_PATH.exists())  # pylint: disable=protected-access
    d2 = run_diagnostics_d1_d4.run_d2_r2_counterfactual(
        v1_telemetry_path=None, batch_size=8, seed=42
    )
    self.assertEqual(d2.get("historical_r2_num_optimizer_steps"), 20)

  def test_g2_prime_epsilon_override_and_telemetry_validation(self) -> None:
    """Verifies _load_g2_prime_epsilon_override and _is_valid_tpu_telemetry helpers."""
    with tempfile.TemporaryDirectory() as tmpdir:
      out_dir = Path(tmpdir)
      self.assertIsNone(run_v2_study._load_g2_prime_epsilon_override(out_dir))  # pylint: disable=protected-access
      (out_dir / "gate_g2_prime_v2.json").write_text(
          json.dumps({
              "passed": True,
              "recommended_epsilon_l": 0.00255176,
              "recommended_epsilon_h": 0.00340235,
          }),
          encoding="utf-8",
      )
      eps_override = run_v2_study._load_g2_prime_epsilon_override(out_dir)  # pylint: disable=protected-access
      self.assertIsNotNone(eps_override)
      assert eps_override is not None
      self.assertAlmostEqual(eps_override[0], 0.00255176)
      self.assertAlmostEqual(eps_override[1], 0.00340235)

    self.assertFalse(run_v2_study._is_valid_tpu_telemetry(None, 4))  # pylint: disable=protected-access
    self.assertFalse(
        run_v2_study._is_valid_tpu_telemetry(  # pylint: disable=protected-access
            {"backend": "cpu_mock", "num_steps": 4}, 4
        )
    )
    self.assertFalse(
        run_v2_study._is_valid_tpu_telemetry(  # pylint: disable=protected-access
            {"backend": "tpu_rl_cluster", "num_steps": 2}, 4
        )
    )
    self.assertTrue(
        run_v2_study._is_valid_tpu_telemetry(  # pylint: disable=protected-access
            {"backend": "tpu_rl_cluster", "num_steps": 4}, 4
        )
    )


if __name__ == "__main__":
  unittest.main()
