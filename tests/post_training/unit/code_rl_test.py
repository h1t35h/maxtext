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

"""Unit tests for MaxText coding RL dataset processor, metrics, and YAML configs."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pytest
import yaml

from maxtext.configs import pyconfig
from maxtext.configs import types
from maxtext.trainers.post_train.rl import code_rl
from maxtext.trainers.post_train.rl import train_rl
from maxtext.utils.globals import MAXTEXT_CONFIGS_DIR

pytestmark = [pytest.mark.post_training]

POST_TRAIN_CONFIG_DIR = Path(MAXTEXT_CONFIGS_DIR) / "post_train"

ALL_CODE_RL_CONFIG_NAMES = (
    "rl_code_gspo_mimo.yml",
    "rl_code_r1_gspo_token.yml",
    "rl_code_r2_gspo_mimo.yml",
    "rl_code_r3_mimo_grpo.yml",
    "rl_code_a0_no_judge.yml",
    "rl_code_a1b_seq_mean.yml",
    "rl_code_a2b_tis.yml",
)


class CodeRlYamlConfigTest(unittest.TestCase):
  """Tests that all 7 coding RL YAML configs load cleanly via pyconfig.initialize."""

  def _load_rl_config(self, config_filename: str) -> pyconfig.HyperParameters:
    config_path = str(POST_TRAIN_CONFIG_DIR / config_filename)
    return pyconfig.initialize(
        ["", config_path],
        config_class=types.RLConfig,
    )

  def test_all_seven_yaml_configs_load_via_pyconfig_initialize(self) -> None:
    for cfg_name in ALL_CODE_RL_CONFIG_NAMES:
      with self.subTest(config=cfg_name):
        cfg = self._load_rl_config(cfg_name)
        self.assertEqual(cfg.model_name, "qwen3-4b")
        self.assertEqual(cfg.tokenizer_path, "Qwen/Qwen3-4B")
        self.assertEqual(cfg.max_prefill_predict_length, 2048)
        self.assertEqual(cfg.max_target_length, 10240)
        self.assertEqual(cfg.batch_size, 8)
        self.assertEqual(cfg.checkpoint_period, 25)
        self.assertEqual(cfg.rl.num_generations, 8)
        self.assertEqual(cfg.rl.mini_batch_size, 2)
        self.assertEqual(cfg.rl.off_policy_steps, 0)
        self.assertTrue(cfg.rl.use_agentic_rollout)
        # 150 batches * 4 mini-batches * 1 iter * 1.0 * 1 epoch = 600 optimizer steps
        self.assertEqual(cfg.steps, 600)
        self.assertEqual(cfg.train_steps, 600)
        self.assertEqual(cfg._pydantic_config.train_steps, 600)

  def test_all_seven_yaml_configs_instantiate_real_tunix_agentic_grpo_config(self) -> None:
    from tunix.rl.agentic.agentic_grpo_learner import GrpoConfig as AgenticGrpoConfig
    from maxtext.trainers.post_train.rl import train_rl

    for cfg_name in ALL_CODE_RL_CONFIG_NAMES:
      with self.subTest(config=cfg_name):
        cfg = self._load_rl_config(cfg_name)
        mimo_cfg = train_rl._build_mimo_config(cfg)  # pylint: disable=protected-access
        grpo_cfg = AgenticGrpoConfig(
            num_generations=cfg.rl.num_generations,
            num_iterations=cfg.rl.num_iterations,
            beta=cfg.rl.grpo_beta,
            epsilon=cfg.rl.grpo_epsilon,
            loss_algo=cfg.rl.loss_algo,
            loss_agg_mode=cfg.rl.loss_agg_mode,
            max_response_length=cfg.max_target_length - cfg.max_prefill_predict_length,
            max_concurrency=cfg.rl.max_concurrency,
            off_policy_steps=cfg.rl.off_policy_steps,
            system_prompt=cfg.rl.system_prompt,
            epsilon_high=cfg.rl.epsilon_high,
            use_rollout_logps=cfg.rl.use_rollout_logps,
            **train_rl._kwargs_supported_by(
                AgenticGrpoConfig,
                gspo_pool_mode=cfg.rl.gspo_pool_mode,
                advantage_estimator=cfg.rl.advantage_estimator,
                clip_mode=cfg.rl.clip_mode,
                epsilon_neg=cfg.rl.epsilon_neg,
                epsilon_neg_high=cfg.rl.epsilon_neg_high,
                token_advantage_mode=cfg.rl.token_advantage_mode,
                mimo=mimo_cfg,
                exact_token_continuity=cfg.rl.exact_token_continuity,
                sampler_is=cfg.rl.sampler_is,
                sampler_is_threshold=cfg.rl.sampler_is_threshold,
                sampler_rs=cfg.rl.sampler_rs,
                sampler_rs_min=cfg.rl.sampler_rs_min,
                sampler_rs_max=cfg.rl.sampler_rs_max,
                seq_logprob_error_threshold=cfg.rl.seq_logprob_error_threshold,
            ),
        )
        self.assertEqual(grpo_cfg.loss_algo, cfg.rl.loss_algo)

  def test_arm_configs_use_dry_base_config_inheritance(self) -> None:
    arm_names = ALL_CODE_RL_CONFIG_NAMES[1:]
    for arm_name in arm_names:
      with self.subTest(arm=arm_name):
        raw = yaml.safe_load((POST_TRAIN_CONFIG_DIR / arm_name).read_text(encoding="utf-8"))
        self.assertEqual(raw.get("base_config"), "rl_code_gspo_mimo.yml")
        # Arm configs should only contain base_config, run_name, and rl overrides
        self.assertLessEqual(set(raw.keys()), {"base_config", "run_name", "rl"})

  def test_arm_specific_algorithmic_settings(self) -> None:
    r1 = self._load_rl_config("rl_code_r1_gspo_token.yml")
    self.assertEqual(r1.rl.loss_algo, "gspo-token")
    self.assertEqual(r1.rl.clip_mode, "ppo")
    self.assertEqual(r1.rl.advantage_estimator, "grpo")
    self.assertEqual(r1.rl.token_advantage_mode, "mask")
    self.assertEqual(r1.rl.loss_agg_mode, "token-mean")

    r2 = self._load_rl_config("rl_code_r2_gspo_mimo.yml")
    self.assertEqual(r2.rl.loss_algo, "gspo-token")
    self.assertEqual(r2.rl.clip_mode, "ppo")
    self.assertEqual(r2.rl.advantage_estimator, "mimo")
    self.assertEqual(r2.rl.token_advantage_mode, "mimo")

    r3 = self._load_rl_config("rl_code_r3_mimo_grpo.yml")
    self.assertEqual(r3.rl.loss_algo, "grpo")
    self.assertEqual(r3.rl.clip_mode, "mask")
    self.assertEqual(r3.rl.advantage_estimator, "mimo")
    self.assertEqual(r3.rl.token_advantage_mode, "mimo")
    self.assertAlmostEqual(r3.rl.grpo_epsilon, 0.8)
    self.assertAlmostEqual(r3.rl.epsilon_high, 4.0)
    self.assertAlmostEqual(r3.rl.epsilon_neg, 0.8)
    self.assertAlmostEqual(r3.rl.epsilon_neg_high, 4.0)

    a0 = self._load_rl_config("rl_code_a0_no_judge.yml")
    self.assertEqual(a0.rl.judge_base_url, "")
    self.assertEqual(a0.rl.loss_algo, "grpo")
    self.assertEqual(a0.rl.clip_mode, "mask")
    self.assertAlmostEqual(a0.rl.grpo_epsilon, 0.8)
    self.assertAlmostEqual(a0.rl.epsilon_high, 4.0)
    self.assertAlmostEqual(a0.rl.epsilon_neg, 0.8)
    self.assertAlmostEqual(a0.rl.epsilon_neg_high, 4.0)

    a1b = self._load_rl_config("rl_code_a1b_seq_mean.yml")
    self.assertEqual(a1b.rl.loss_agg_mode, "sequence-mean-token-mean")

    a2b = self._load_rl_config("rl_code_a2b_tis.yml")
    self.assertEqual(a2b.rl.sampler_is, "token")
    self.assertAlmostEqual(a2b.rl.sampler_is_threshold, 2.0)
    self.assertEqual(a2b.rl.sampler_rs, "geometric")
    self.assertAlmostEqual(a2b.rl.sampler_rs_min, 0.9997)
    self.assertAlmostEqual(a2b.rl.sampler_rs_max, 1.0004)

  def test_v2_canonical_arm_configs_settings(self) -> None:
    r1p = self._load_rl_config("rl_code_r1p_gspo_scalar.yml")
    self.assertEqual(r1p.rl.loss_algo, "gspo-token")
    self.assertEqual(r1p.rl.gspo_pool_mode, "sequence")
    self.assertEqual(r1p.rl.loss_agg_mode, "sequence-mean-token-mean")
    self.assertEqual(r1p.rl.advantage_estimator, "grpo")
    self.assertEqual(r1p.rl.token_advantage_mode, "none")
    self.assertFalse(r1p.rl.use_rollout_logps)
    self.assertEqual(r1p.rl.sampler_is, "token")
    self.assertEqual(r1p.rl.off_policy_steps, 0)

    r2p = self._load_rl_config("rl_code_r2p_gspo_mimo.yml")
    self.assertEqual(r2p.rl.loss_algo, "gspo-token")
    self.assertEqual(r2p.rl.gspo_pool_mode, "sequence")
    self.assertEqual(r2p.rl.loss_agg_mode, "sequence-mean-token-mean")
    self.assertEqual(r2p.rl.advantage_estimator, "mimo")
    self.assertEqual(r2p.rl.token_advantage_mode, "mimo")
    self.assertFalse(r2p.rl.use_rollout_logps)
    self.assertEqual(r2p.rl.sampler_is, "token")
    self.assertAlmostEqual(r2p.rl.mimo.span_beta_pos, 0.5)
    self.assertTrue(r2p.rl.mimo.span_renormalize)

    r2pp = self._load_rl_config("rl_code_r2pp_turn_gspo_mimo.yml")
    self.assertEqual(r2pp.rl.loss_algo, "gspo-turn")
    self.assertEqual(r2pp.rl.gspo_pool_mode, "turn")
    self.assertEqual(r2pp.rl.loss_agg_mode, "sequence-mean-token-mean")
    self.assertEqual(r2pp.rl.token_advantage_mode, "mimo")
    self.assertFalse(r2pp.rl.use_rollout_logps)

    r3p = self._load_rl_config("rl_code_r3p_mimo_grpo.yml")
    self.assertEqual(r3p.rl.loss_algo, "grpo")
    self.assertEqual(r3p.rl.clip_mode, "ppo")
    self.assertAlmostEqual(r3p.rl.grpo_epsilon, 0.20)
    self.assertAlmostEqual(r3p.rl.epsilon_high, 0.28)
    self.assertFalse(r3p.rl.use_rollout_logps)
    self.assertEqual(r3p.rl.sampler_is, "token")

    for cfg in (r1p, r2p, r2pp, r3p):
      training_cfg, grpo_cfg = train_rl.build_tunix_rl_configs(cfg)
      self.assertIsInstance(training_cfg, train_rl.rl_cluster_lib.RLTrainingConfig)
      self.assertIsNotNone(grpo_cfg)




class CodeRlDatasetProcessorTest(unittest.TestCase):
  """Tests for maxtext.trainers.post_train.rl.code_rl.dataset_processor."""

  def test_process_asserts_record(self) -> None:
    raw = {
        "prompt_id": "mbpp_42",
        "problem": "Write a function `cube(n)` that returns `n ** 3`.",
        "tests": ["assert cube(2) == 8", "assert cube(-3) == -27"],
        "format": "asserts",
        "entry_point": "cube",
    }
    rec = code_rl.process_code_rl_record(raw)
    self.assertEqual(rec["prompt_id"], "mbpp_42")
    self.assertEqual(rec["format"], "asserts")
    self.assertIn("cube(n)", rec["prompts"])
    self.assertEqual(rec["question"], raw["problem"])
    self.assertIn("```python", rec["format_instructions"])
    self.assertEqual(rec["entry_point"], "cube")
    self.assertEqual(len(rec["tests"]), 2)

  def test_process_data_entrypoint_loadable_by_utils_rl(self) -> None:
    from maxtext.trainers.post_train.rl import utils_rl

    fn = utils_rl.load_custom_callable(
        "maxtext.trainers.post_train.rl.code_rl.dataset_processor", "process_data"
    )
    rec = fn(
        "json",
        None,
        None,
        None,
        {"problem": "Write add(a, b).", "tests": ["assert add(1, 2) == 3"], "prompt_id": "p1"},
    )
    self.assertIn("Write add(a, b).", rec["prompts"])
    self.assertEqual(rec["question"], "Write add(a, b).")
    self.assertEqual(rec["prompt_id"], "p1")

  def test_process_stdio_record_and_auto_detect_format(self) -> None:
    raw = {
        "task_id": 108,
        "question": "Read integer x from stdin and print x + 1.",
        "tests": {"inputs": ["10\n", "0\n"], "outputs": ["11\n", "1\n"]},
    }
    rec = code_rl.process_code_rl_record(raw)
    self.assertEqual(rec["prompt_id"], "108")
    self.assertEqual(rec["format"], "stdio")
    self.assertEqual(rec["tests"]["inputs"], ["10\n", "0\n"])
    self.assertEqual(rec["tests"]["outputs"], ["11\n", "1\n"])

  def test_process_record_rejects_missing_problem_or_tests(self) -> None:
    with self.assertRaises(ValueError):
      code_rl.process_code_rl_record({"problem": "   ", "tests": ["assert True"]})
    with self.assertRaises(ValueError):
      code_rl.process_code_rl_record({"problem": "Valid problem", "tests": []})
    with self.assertRaises(ValueError):
      code_rl.process_code_rl_record({
          "problem": "Valid problem",
          "tests": ["assert True"],
          "format": "invalid_fmt",
      })

  def test_format_code_rl_dataset_and_load_jsonl(self) -> None:
    raw_items = [
        {
            "prompt_id": f"p_{i}",
            "problem": f"Compute f_{i}(x).",
            "tests": [f"assert f_{i}(1) == 1"],
        }
        for i in range(4)
    ]
    processed = code_rl.format_code_rl_dataset(raw_items)
    self.assertEqual(len(processed), 4)
    self.assertEqual(processed[2]["prompt_id"], "p_2")

    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False, encoding="utf-8") as tmp:
      for item in raw_items:
        tmp.write(json.dumps(item) + "\n")
      tmp_path = tmp.name

    try:
      loaded = code_rl.load_jsonl_records(tmp_path)
      self.assertEqual(len(loaded), 4)
      self.assertEqual(loaded[0]["prompt_id"], "p_0")
    finally:
      os.unlink(tmp_path)


class CodeRlMetricsTest(unittest.TestCase):
  """Tests for maxtext.trainers.post_train.rl.code_rl.metrics."""

  def test_compute_code_rl_metrics_solve_and_repair_gain(self) -> None:
    m = code_rl.compute_code_rl_metrics(
        turn1_passed=[True, False, False, True],
        final_passed=[True, True, False, True],
        assistant_lengths=[1200, 3400, 8192, 950],
        max_response_length=8192,
    )
    self.assertAlmostEqual(m["code_rl/turn1_solve"], 0.50)
    self.assertAlmostEqual(m["code_rl/final_solve"], 0.75)
    self.assertAlmostEqual(m["code_rl/repair_gain"], 0.25)
    self.assertAlmostEqual(m["code_rl/truncation_rate"], 0.25)
    self.assertAlmostEqual(m["code_rl/mean_assistant_tokens"], float(np.mean([1200, 3400, 8192, 950])))

  def test_compute_code_rl_metrics_empty_batch(self) -> None:
    m = code_rl.compute_code_rl_metrics([], [], [])
    self.assertEqual(m["code_rl/turn1_solve"], 0.0)
    self.assertEqual(m["code_rl/final_solve"], 0.0)
    self.assertEqual(m["code_rl/repair_gain"], 0.0)
    self.assertEqual(m["code_rl/truncation_rate"], 0.0)

  def test_extract_trajectory_code_rl_metrics(self) -> None:
    trajectories = [
        {
            "conversation_masks": [1] * 500,
            "trajectory_reward": 1.0,
            "env_extras": {
                "passed": True,
                "turn_records": [{"test_result": {"all_passed": False}}],
            },
            "judge": {"passed": True, "assistant_len": 500},
        },
        {
            "conversation_masks": [1] * 8192,
            "trajectory_reward": 0.0,
            "env_extras": {
                "passed": False,
                "turn_records": [{"test_result": {"all_passed": False}}],
            },
            "judge": {"passed": False, "assistant_len": 8192},
        },
    ]
    m = code_rl.extract_trajectory_code_rl_metrics(trajectories, max_response_length=8192)
    self.assertAlmostEqual(m["code_rl/turn1_solve"], 0.0)
    self.assertAlmostEqual(m["code_rl/final_solve"], 0.5)
    self.assertAlmostEqual(m["code_rl/repair_gain"], 0.5)
    self.assertAlmostEqual(m["code_rl/truncation_rate"], 0.5)

    tunix_metrics = code_rl.get_code_rl_metrics(
        prompts=["p1", "p2"],
        completions=["c1", "c2"],
        rewards=np.array([1.0, 0.0]),
        advantages=np.array([0.5, -0.5]),
        trajectories=trajectories,
    )
    for required_key in (
        "reward/turn1_solve",
        "reward/final_solve",
        "reward/repair_gain",
        "length/assistant_tokens_mean",
        "length/assistant_tokens_p90",
        "length/truncation_rate_at_8192",
    ):
      self.assertIn(required_key, tunix_metrics)
      val, agg_fn = tunix_metrics[required_key]
      self.assertIsInstance(val, (int, float))
      self.assertTrue(callable(agg_fn))
    self.assertAlmostEqual(tunix_metrics["reward/repair_gain"][0], 0.5)


class CodeRlScriptsTest(unittest.TestCase):
  """Tests for maxtext.trainers.post_train.rl.code_rl.scripts (M4)."""

  def test_decontamination_and_pass_at_8_dataset_builder(self) -> None:
    from maxtext.trainers.post_train.rl.code_rl.scripts import build_code_rl_dataset

    eval_problems = [
        "Write a python function to find the longest palindromic substring in a given string s of lowercase letters."
    ]
    candidates = [
        {
            "prompt_id": "clean_1",
            "source": "mbpp_train",
            "task_id": 601,
            "problem": "Write a function to compute the greatest common divisor of two positive integers a and b.",
            "tests": [f"assert gcd({i}, {i*2}) == {i}" for i in range(1, 6)],
            "base_pass_rate": 0.5,
        },
        {
            "prompt_id": "contaminated_1",
            "source": "custom",
            "task_id": 900,
            "problem": "Write a python function to find the longest palindromic substring in a given string s of lowercase letters.",
            "tests": ["assert True"] * 5,
            "base_pass_rate": 0.5,
        },
        {
            "prompt_id": "mbpp_test_excluded",
            "source": "mbpp",
            "task_id": 250,
            "problem": "Distinctive clean problem text that is inside MBPP test split IDs 11..510.",
            "tests": ["assert True"] * 5,
            "base_pass_rate": 0.5,
        },
        {
            "prompt_id": "too_few_tests",
            "source": "taco",
            "task_id": 700,
            "problem": "Clean problem with only 3 tests.",
            "tests": ["assert True"] * 3,
            "base_pass_rate": 0.5,
        },
        {
            "prompt_id": "pass_rate_zero",
            "source": "taco",
            "task_id": 701,
            "problem": "Clean problem with 0/8 pass rate.",
            "tests": ["assert True"] * 5,
            "base_pass_rate": 0.0,
        },
        {
            "prompt_id": "pass_rate_one",
            "source": "taco",
            "task_id": 702,
            "problem": "Clean problem with 8/8 pass rate.",
            "tests": ["assert True"] * 5,
            "base_pass_rate": 1.0,
        },
    ]
    curated = build_code_rl_dataset.decontaminate_and_filter_dataset(
        candidates, eval_problems
    )
    self.assertEqual(len(curated), 1)
    self.assertEqual(curated[0]["prompt_id"], "clean_1")
    self.assertAlmostEqual(curated[0]["base_pass_rate"], 0.5)

  def test_split_train_and_heldout_and_cli(self) -> None:
    from maxtext.trainers.post_train.rl.code_rl.scripts import build_code_rl_dataset

    pool = [
        {
            "prompt_id": f"item_{i}",
            "source": "taco",
            "task_id": 1000 + i,
            "problem": f"Compute unique mathematical transformation {i} on integer array.",
            "tests": [f"assert f({j}) == {j}" for j in range(5)],
            "base_pass_rate": 0.375,
        }
        for i in range(12)
    ]
    with tempfile.TemporaryDirectory() as tmpdir:
      cand_path = Path(tmpdir) / "candidates.jsonl"
      out_dir = Path(tmpdir) / "out"
      with cand_path.open("w", encoding="utf-8") as f:
        for item in pool:
          f.write(json.dumps(item) + "\n")
      rc = build_code_rl_dataset.main(
          [
              "--candidates-jsonl",
              str(cand_path),
              "--output-dir",
              str(out_dir),
              "--heldout-size",
              "4",
          ]
      )
      self.assertEqual(rc, 0)
      train_recs = code_rl.load_jsonl_records(out_dir / "train.jsonl")
      heldout_recs = code_rl.load_jsonl_records(out_dir / "heldout_eval.jsonl")
      self.assertEqual(len(train_recs), 8)
      self.assertEqual(len(heldout_recs), 4)

  def test_calibrate_logp_mismatch_and_gate_g2(self) -> None:
    from maxtext.trainers.post_train.rl.code_rl.scripts import calibrate_logp_mismatch

    rng = np.random.default_rng(123)
    mask = np.ones((64, 32), dtype=np.int32)
    rollout_lp = rng.uniform(-2.5, -0.5, size=(64, 32))
    trainer_lp = rollout_lp + rng.normal(0.0, 0.001, size=(64, 32))
    report = calibrate_logp_mismatch.calibrate_logp_mismatch(
        trainer_lp, rollout_lp, mask
    )
    self.assertGreaterEqual(report["step0_seq_clip_frac"], 0.10)
    self.assertLessEqual(report["step0_seq_clip_frac"], 0.20)
    self.assertGreater(report["recommended_epsilon_l"], 0.0)
    self.assertGreater(
        report["recommended_epsilon_h"], report["recommended_epsilon_l"]
    )
    gate = calibrate_logp_mismatch.evaluate_gate_g2(
        report, usable_pool_size=2400
    )
    self.assertTrue(gate["passed"])

    # Skewed negative shift (Challenger 1 regression test)
    rng_skew = np.random.default_rng(7)
    mask_skew = np.ones((200, 64), dtype=np.int32)
    rollout_skew = rng_skew.uniform(-2.5, -0.5, size=(200, 64))
    trainer_skew = rollout_skew - 0.015 + rng_skew.normal(0.0, 0.005, size=(200, 64))
    rep_skew = calibrate_logp_mismatch.calibrate_logp_mismatch(
        trainer_skew, rollout_skew, mask_skew
    )
    self.assertGreaterEqual(rep_skew["step0_seq_clip_frac"], 0.10)
    self.assertLessEqual(rep_skew["step0_seq_clip_frac"], 0.20)

    self.assertEqual(calibrate_logp_mismatch.recommend_mismatch_strategy(1e-4), "NONE")
    self.assertEqual(calibrate_logp_mismatch.recommend_mismatch_strategy(1e-3), "TIS")
    self.assertEqual(
        calibrate_logp_mismatch.recommend_mismatch_strategy(5e-3),
        "TIS_AND_GEO_RS",
    )

  def test_serve_judge_v8i_script_flags_and_fallback_models(self) -> None:
    script_path = (
        Path(__file__).resolve().parents[3]
        / "src"
        / "maxtext"
        / "trainers"
        / "post_train"
        / "rl"
        / "code_rl"
        / "scripts"
        / "serve_judge_v8i.sh"
    )
    self.assertTrue(script_path.exists())
    content = script_path.read_text(encoding="utf-8")
    for required_token in (
        "Qwen/Qwen3-235B-A22B-Instruct-2507",
        "Qwen/Qwen3-235B-A22B-Instruct-2507-FP8",
        "Qwen/Qwen3-Coder-30B-A3B-Instruct",
        "Qwen/Qwen3-32B",
        "--tensor-parallel-size",
        "--max-model-len",
        "--max-num-seqs",
        "--enable-prefix-caching",
        "--guided-decoding-backend",
        "--port",
    ):
      self.assertIn(required_token, content)

  def test_eval_code_checkpoints_paired_bootstrap_ci_and_auc(self) -> None:
    from maxtext.trainers.post_train.rl.code_rl.scripts import eval_code_checkpoints

    rng = np.random.default_rng(99)
    base = rng.binomial(1, 0.45, size=200).astype(np.float64)
    improved = np.clip(base + rng.binomial(1, 0.15, size=200), 0.0, 1.0)
    ci = eval_code_checkpoints.paired_bootstrap_ci(
        improved, base, num_resamples=2000, seed=42
    )
    self.assertGreater(ci["mean_diff"], 0.05)
    self.assertGreater(ci["ci_low"], 0.0)
    self.assertTrue(ci["significant"])

    auc = eval_code_checkpoints.compute_checkpoint_auc(
        [0, 25, 50, 75, 100, 125, 150],
        [0.40, 0.44, 0.48, 0.50, 0.52, 0.54, 0.56],
    )
    self.assertGreater(auc, 0.40)
    self.assertLess(auc, 0.56)

    endpoints = eval_code_checkpoints.evaluate_hypothesis_endpoints(
        {
            "R1": {
                "problem_scores": base,
                "mean_assistant_tokens": 2400.0,
                "syntax_error_frac": [0.30, 0.25, 0.20],
            },
            "R2": {
                "problem_scores": improved,
                "mean_assistant_tokens": 2100.0,
                "instability_events": 0,
                "syntax_error_frac": [0.30, 0.15, 0.05],
            },
            "R3": {
                "problem_scores": base,
                "mean_assistant_tokens": 2150.0,
                "instability_events": 2,
            },
        },
        num_resamples=2000,
        seed=42,
    )
    self.assertTrue(endpoints["H1_supported"])
    self.assertTrue(endpoints["H2_supported"])
    self.assertTrue(endpoints["H3_supported"])

    # When R2 accuracy collapses below R1, H2_supported must be False even if shorter
    collapsed = eval_code_checkpoints.evaluate_hypothesis_endpoints(
        {
            "R1": {"problem_scores": [1.0] * 20, "mean_assistant_tokens": 2000.0},
            "R2": {"problem_scores": [0.0] * 20, "mean_assistant_tokens": 100.0},
        },
        num_resamples=500,
    )
    self.assertFalse(collapsed["H2_supported"])


if __name__ == "__main__":
  unittest.main()
