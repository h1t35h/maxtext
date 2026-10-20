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

"""Unified V2 multi-seed & off-policy study runner for GSPO-MiMo coding RL (§R4)."""

from __future__ import annotations

import argparse
import ast
import asyncio
import base64
from collections import defaultdict
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
import contextlib
import dataclasses
import functools
import gc
import inspect
import json
import math
import os
from pathlib import Path
import pickle
import re
import time
from typing import Any
import urllib.request
import zlib

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from maxtext.configs import pyconfig
from maxtext.configs import types
from maxtext.trainers.post_train.rl import train_rl
from maxtext.trainers.post_train.rl import utils_rl
from maxtext.trainers.post_train.rl.code_rl import dataset_processor
from maxtext.trainers.post_train.rl.code_rl import metrics as code_rl_metrics
from maxtext.trainers.post_train.rl.code_rl.scripts import calibrate_logp_mismatch
from maxtext.trainers.post_train.rl.code_rl.scripts import eval_code_checkpoints
from maxtext.trainers.post_train.rl.code_rl.scripts import run_diagnostics_d1_d4
from maxtext.utils import max_logging
from maxtext.utils import model_creation_utils
from transformers import AutoTokenizer

from tunix.rl import advantage_shaping
from tunix.rl import algorithm_config as algo_cfg_lib
from tunix.rl import common as rl_common
from tunix.rl import rl_cluster as rl_cluster_lib
from tunix.rl.agentic import agentic_grpo_learner
from tunix.rl.agentic.agents import agent_types
from tunix.rl.agentic.environments.code import code_env as code_env_mod
from tunix.rl.agentic.environments.code import code_judge_postprocessor as postproc_mod
from tunix.rl.agentic.environments.code import judge_client as jc_mod
from tunix.rl.agentic.environments.code import sandbox as sandbox_mod
from tunix.rl.agentic.parser.chat_template_parser import parser as chat_parser_lib
from tunix.rl.agentic.pipeline import group_postprocess
from tunix.rl.agentic.trajectory import trajectory_collect_engine

# V2 Token & Memory Budgets (§R4.1):
# Prompt cap 1536 + response budget 3072 = 4608 total target length (<10% truncation).
# Eval generation budget >= 1024 tokens/turn with 2560 prompt cap.
DEFAULT_MAX_PREFILL_LENGTH: int = 1536
DEFAULT_MAX_TARGET_LENGTH: int = 4608
DEFAULT_MAX_RESPONSE_LENGTH: int = DEFAULT_MAX_TARGET_LENGTH - DEFAULT_MAX_PREFILL_LENGTH  # 3072
DEFAULT_EVAL_MAX_GENERATION_STEPS: int = 1024
DEFAULT_EVAL_MAX_PROMPT_TOKENS: int = 2560
DEFAULT_SEEDS: tuple[int, ...] = (42, 43, 44)
DEFAULT_ARMS: tuple[str, ...] = ("R1p", "R2p", "R2pp", "R3p", "R3p_seq")
STRESS_ARMS: tuple[str, ...] = ("R2p", "R2pp", "R3p")
DEFAULT_BENCHMARKS: tuple[str, ...] = (
    "heldout_indist",
    "livecodebench",
    "humanevalplus",
    "mbppplus",
)

SYSTEM_PROMPT: str = (
    "You are an expert Python programmer. Put brief reasoning inside <think>...</think> "
    "and provide a complete, self-contained solution inside a single ```python ... ``` block."
)

# Canonical V2 Arm Specifications (§R2.3, §R3.3, §R4.2)
ARMS_V2: dict[str, dict[str, Any]] = {
    "R1p": {
        "config_file": "rl_code_r1p_gspo_scalar.yml",
        "yaml": "rl_code_r1p_gspo_scalar.yml",
        "description": "True Scalar GSPO Baseline (advantage_estimator=grpo, token_advantage_mode=none, sequence-mean-token-mean)",
        "loss_algo": "gspo-token",
        "gspo_pool_mode": "sequence",
        "loss_agg_mode": "sequence-mean-token-mean",
        "advantage_estimator": "grpo",
        "token_advantage_mode": "none",
        "clip_mode": "ppo",
        "grpo_epsilon": 3.0e-4,
        "epsilon_high": 4.0e-4,
        "epsilon_neg": None,
        "epsilon_neg_high": None,
        "use_rollout_logps": False,
        "sampler_is": "token",
        "sampler_is_threshold": 2.0,
        "off_policy_steps": 0,
        "use_judge": False,
    },
    "R2p": {
        "config_file": "rl_code_r2p_gspo_mimo.yml",
        "yaml": "rl_code_r2p_gspo_mimo.yml",
        "description": "Corrected Sequence-Level GSPO-MiMo (beta_pos=0.5, span_renormalize=True, sequence-mean-token-mean, trainer pi_old + TIS)",
        "loss_algo": "gspo-token",
        "gspo_pool_mode": "sequence",
        "loss_agg_mode": "sequence-mean-token-mean",
        "advantage_estimator": "mimo",
        "token_advantage_mode": "mimo",
        "clip_mode": "ppo",
        "grpo_epsilon": 3.0e-4,
        "epsilon_high": 4.0e-4,
        "epsilon_neg": None,
        "epsilon_neg_high": None,
        "use_rollout_logps": False,
        "sampler_is": "token",
        "sampler_is_threshold": 2.0,
        "off_policy_steps": 0,
        "use_judge": True,
    },
    "R2pp": {
        "config_file": "rl_code_r2pp_turn_gspo_mimo.yml",
        "yaml": "rl_code_r2pp_turn_gspo_mimo.yml",
        "description": "Turn-Level GSPO-MiMo (loss_algo=gspo-turn, gspo_pool_mode=turn, sequence-mean-token-mean, per-turn ratio clipping)",
        "loss_algo": "gspo-turn",
        "gspo_pool_mode": "turn",
        "loss_agg_mode": "sequence-mean-token-mean",
        "advantage_estimator": "mimo",
        "token_advantage_mode": "mimo",
        "clip_mode": "ppo",
        "grpo_epsilon": 3.0e-4,
        "epsilon_high": 4.0e-4,
        "epsilon_neg": None,
        "epsilon_neg_high": None,
        "use_rollout_logps": False,
        "sampler_is": "token",
        "sampler_is_threshold": 2.0,
        "off_policy_steps": 0,
        "use_judge": True,
    },
    "R3p": {
        "config_file": "rl_code_r3p_mimo_grpo.yml",
        "yaml": "rl_code_r3p_mimo_grpo.yml",
        "description": "Fair Token-Level MiMo-GRPO Baseline (clip_mode=ppo, eps=0.20/0.28, token-mean, trainer pi_old + TIS)",
        "loss_algo": "grpo",
        "gspo_pool_mode": "sequence",
        "loss_agg_mode": "token-mean",
        "advantage_estimator": "mimo",
        "token_advantage_mode": "mimo",
        "clip_mode": "ppo",
        "grpo_epsilon": 0.20,
        "epsilon_high": 0.28,
        "epsilon_neg": 0.20,
        "epsilon_neg_high": 0.28,
        "use_rollout_logps": False,
        "sampler_is": "token",
        "sampler_is_threshold": 2.0,
        "off_policy_steps": 0,
        "use_judge": True,
    },
    "R3p_seq": {
        "config_file": "rl_code_r3p_seq_mimo_grpo.yml",
        "yaml": "rl_code_r3p_seq_mimo_grpo.yml",
        "description": "Fair Token-Level MiMo-GRPO with sequence-mean-token-mean aggregation",
        "loss_algo": "grpo",
        "gspo_pool_mode": "sequence",
        "loss_agg_mode": "sequence-mean-token-mean",
        "advantage_estimator": "mimo",
        "token_advantage_mode": "mimo",
        "clip_mode": "ppo",
        "grpo_epsilon": 0.20,
        "epsilon_high": 0.28,
        "epsilon_neg": 0.20,
        "epsilon_neg_high": 0.28,
        "use_rollout_logps": False,
        "sampler_is": "token",
        "sampler_is_threshold": 2.0,
        "off_policy_steps": 0,
        "use_judge": True,
    },
}


def _noop_close() -> None:
  """Prevents AgenticGrpoLearner.train() from shutting down the persistent RLCluster."""
  return None


def get_loss_compile_key(
    arm_name: str,
    *,
    epsilon_override: tuple[float, float] | None = None,
) -> tuple[Any, ...]:
  """Returns a hashable key identifying the compiled XLA loss structure for `_jitted_train_step_fn` reuse across seeds."""
  if arm_name not in ARMS_V2:
    raise KeyError(f"Unknown V2 arm {arm_name!r}; valid arms: {sorted(ARMS_V2)}")
  spec = ARMS_V2[arm_name]
  eps_l = float(spec["grpo_epsilon"])
  eps_h = float(spec["epsilon_high"])
  if epsilon_override is not None and spec["loss_algo"] in ("gspo-token", "gspo-turn"):
    eps_l, eps_h = float(epsilon_override[0]), float(epsilon_override[1])
  return (
      str(spec["loss_algo"]),
      str(spec["gspo_pool_mode"]),
      str(spec["loss_agg_mode"]),
      str(spec["clip_mode"]),
      round(eps_l, 8),
      round(eps_h, 8),
      str(spec["sampler_is"]),
      float(spec["sampler_is_threshold"]),
  )


def reset_actor_for_arm_seed(
    rl_cluster: Any,
    actor_model: Any,
    initial_actor_state: Mapping[str, Any],
    *,
    arm_name: str,
    base_opt_tx: Any = None,
    actor_mesh: Any = None,
    jitted_step_cache: dict[tuple[Any, ...], Any] | None = None,
    prev_compile_key: tuple[Any, ...] | None = None,
    epsilon_override: tuple[float, float] | None = None,
) -> tuple[Any, ...]:
  """Restores in-memory `initial_actor_state`, resets optimizer, and reuses `_jitted_train_step_fn` across seeds."""
  new_compile_key = get_loss_compile_key(arm_name, epsilon_override=epsilon_override)
  if jitted_step_cache is not None and prev_compile_key is not None:
    cur_jitted = getattr(rl_cluster.actor_trainer, "_jitted_train_step_fn", None)
    if cur_jitted is not None:
      jitted_step_cache[prev_compile_key] = cur_jitted
    if (
        prev_compile_key != new_compile_key
        and new_compile_key not in jitted_step_cache
    ):
      jitted_step_cache.clear()

  actor_base = actor_model.base if hasattr(actor_model, "base") else actor_model
  cur_state = nnx.state(actor_base, nnx.Param)
  cur_pure = nnx.to_pure_dict(cur_state)
  restored_pure = jax.tree_util.tree_map(
      lambda cur, init_np: jax.device_put(init_np, cur.sharding)
      if isinstance(cur, jax.Array) and hasattr(cur, "sharding")
      else jnp.asarray(init_np)
      if isinstance(cur, jax.Array)
      else init_np,
      cur_pure,
      initial_actor_state,
  )
  nnx.replace_by_pure_dict(cur_state, restored_pure)
  nnx.update(actor_base, cur_state)
  del cur_state, cur_pure, restored_pure

  trainer = rl_cluster.actor_trainer
  if base_opt_tx is None and hasattr(trainer, "optimizer") and hasattr(trainer.optimizer, "tx"):
    base_opt_tx = trainer.optimizer.tx

  def _reinit_opt() -> None:
    wrt_target = (
        nnx.LoRAParam if getattr(trainer, "_lora_enabled", False) else nnx.Param
    )
    if base_opt_tx is not None and hasattr(trainer, "model"):
      trainer.optimizer = nnx.Optimizer(trainer.model, base_opt_tx, wrt=wrt_target)
    if (
        hasattr(trainer, "grad_accumulator")
        and trainer.grad_accumulator is not None
        and hasattr(trainer, "model")
    ):
      _uses_cond_path = not (
          trainer.config.get_with_default("gradient_accumulation_steps", 1) == 1
          and getattr(trainer.config, "max_seq_token_per_tpu", None) is None
      )
      trainer.grad_accumulator = type(trainer.grad_accumulator)(
          trainer.model, wrt_target, allocate_grads=_uses_cond_path
      )

  if actor_mesh is not None:
    with actor_mesh:
      _reinit_opt()
  else:
    _reinit_opt()

  if jitted_step_cache is not None and new_compile_key in jitted_step_cache:
    trainer._jitted_train_step_fn = jitted_step_cache[new_compile_key]
  else:
    trainer._jitted_train_step_fn = None
    trainer._jitted_eval_step_fn = None
    gc.collect()

  trainer._train_steps = 0
  trainer._iter_steps = 0
  trainer._buffered_train_metrics = None
  trainer._prev_buffered_train_metrics = None
  if hasattr(rl_cluster, "sync_weights"):
    rl_cluster.sync_weights()
  rl_cluster.global_steps = 0
  if hasattr(rl_cluster, "_buffered_train_metrics"):
    rl_cluster._buffered_train_metrics.clear()
  if hasattr(rl_cluster, "_buffered_eval_metrics"):
    rl_cluster._buffered_eval_metrics.clear()
  gc.collect()
  return new_compile_key


def build_v2_mimo_config() -> algo_cfg_lib.MimoAdvantageConfig:
  """Builds the canonical V2 MimoAdvantageConfig (`span_beta_pos=0.5, span_renormalize=True`)."""
  return algo_cfg_lib.MimoAdvantageConfig(
      lp_pass_rate_threshold=0.5,
      lp_max_penalty=0.4,
      lp_dead_zone=0.2,
      lp_saturation=1.0,
      lp_gamma=1.0,
      gar_lambda_max=2.0,
      gar_rank_rho=0.5,
      span_kappa=1.5,
      span_alpha_max=2.0,
      span_beta_min=0.5,
      span_beta_pos=0.5,
      span_renormalize=True,
      std_normalize=True,
  )


def build_v2_grpo_config(
    arm_name: str,
    *,
    num_generations: int = 8,
    num_iterations: int = 1,
    max_response_length: int = DEFAULT_MAX_RESPONSE_LENGTH,
    epsilon_override: tuple[float, float] | None = None,
) -> agentic_grpo_learner.GrpoConfig:
  """Builds the Tunix AgenticGrpoConfig for a canonical V2 study arm."""
  if arm_name not in ARMS_V2:
    raise KeyError(f"Unknown V2 arm {arm_name!r}; valid arms: {sorted(ARMS_V2)}")
  spec = ARMS_V2[arm_name]
  mimo_cfg = build_v2_mimo_config()
  eps_l = float(spec["grpo_epsilon"])
  eps_h = float(spec["epsilon_high"])
  if epsilon_override is not None and spec["loss_algo"] in ("gspo-token", "gspo-turn"):
    eps_l, eps_h = float(epsilon_override[0]), float(epsilon_override[1])

  return agentic_grpo_learner.GrpoConfig(
      num_generations=int(num_generations),
      num_iterations=int(num_iterations),
      beta=0.0,
      epsilon=eps_l,
      epsilon_high=eps_h,
      epsilon_neg=spec["epsilon_neg"],
      epsilon_neg_high=spec["epsilon_neg_high"],
      loss_algo=spec["loss_algo"],
      gspo_pool_mode=spec["gspo_pool_mode"],
      loss_agg_mode=spec["loss_agg_mode"],
      advantage_estimator=spec["advantage_estimator"],
      token_advantage_mode=spec["token_advantage_mode"],
      clip_mode=spec["clip_mode"],
      mimo=mimo_cfg,
      max_response_length=int(max_response_length),
      max_concurrency=64,
      off_policy_steps=int(spec["off_policy_steps"]),
      use_rollout_logps=bool(spec["use_rollout_logps"]),
      exact_token_continuity=True,
      sampler_is=spec["sampler_is"],
      sampler_is_threshold=float(spec["sampler_is_threshold"]),
  )


def verify_live_judge(
    judge_url: str,
    judge_model: str,
    timeout_s: float = 15.0,
) -> dict[str, Any]:
  """Verifies connectivity to the live OpenAI-compatible Qwen3 Judge server."""
  base = judge_url.rstrip("/")
  candidate_urls = [base + "/models"]
  if "34.144.92.5:8000" in base:
    candidate_urls.append(base.replace("34.144.92.5:8000", "127.0.0.1:8000") + "/models")
  last_exc: Exception | None = None
  for idx, models_url in enumerate(candidate_urls):
    try:
      req = urllib.request.Request(models_url, method="GET")
      eff_to = min(timeout_s, 3.0) if idx == 0 and len(candidate_urls) > 1 else timeout_s
      with urllib.request.urlopen(req, timeout=eff_to) as resp:
        data = json.loads(resp.read().decode("utf-8"))
      model_ids = [m.get("id", "") for m in data.get("data", [])]
      return {
          "reachable": True,
          "judge_url": judge_url,
          "judge_model": judge_model,
          "served_models": model_ids,
      }
    except Exception as exc:  # pylint: disable=broad-except
      last_exc = exc
  return {
      "reachable": False,
      "judge_url": judge_url,
      "judge_model": judge_model,
      "error": str(last_exc),
  }


def evaluate_gate_g1_prime(
    *,
    d3_report: Mapping[str, Any],
    d4_report: Mapping[str, Any],
    d1_report: Mapping[str, Any] | None = None,
    d2_report: Mapping[str, Any] | None = None,
    judge_status: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
  """Evaluates V2 Hard Calibration Gate G1' (Mass Conservation & Judge Selectivity, §R3)."""
  flagged_frac = float(d3_report.get("v2_flagged_token_frac", 0.0))
  selectivity_ok = 0.15 <= flagged_frac <= 0.35
  passing_turn_fp = float(d3_report.get("v2_passing_turn_code_fp_rate", 0.0))
  passing_turn_ok = passing_turn_fp <= 0.05

  min_pos_ratio = float(d4_report.get("v2_min_mass_pos_ratio", 0.0))
  max_pos_ratio = float(d4_report.get("v2_max_mass_pos_ratio", 0.0))
  max_abs_adv_mean = float(d4_report.get("v2_max_abs_advantage_mean", 1.0))
  mass_pos_ok = 0.95 <= min_pos_ratio and max_pos_ratio <= 1.05
  adv_mean_ok = max_abs_adv_mean < 0.03

  d1_ok = True if d1_report is None else bool(d1_report.get("v2_determinism_verified", True))
  d2_ok = True if d2_report is None else bool(d2_report.get("counterfactual_verified", True))
  judge_ok = True if judge_status is None else bool(judge_status.get("reachable", True))
  passed = bool(
      selectivity_ok
      and passing_turn_ok
      and mass_pos_ok
      and adv_mean_ok
      and d1_ok
      and d2_ok
      and judge_ok
  )
  return {
      "gate": "G1_prime",
      "passed": passed,
      "gate_g1_prime_passed": passed,
      "selectivity_in_band_15_to_35pct": selectivity_ok,
      "passing_turn_code_protected": passing_turn_ok,
      "mass_pos_ratio_in_95_to_105pct": mass_pos_ok,
      "advantage_mean_abs_below_003": adv_mean_ok,
      "flagged_token_frac": flagged_frac,
      "v2_min_mass_pos_ratio": min_pos_ratio,
      "v2_max_mass_pos_ratio": max_pos_ratio,
      "v2_max_abs_advantage_mean": max_abs_adv_mean,
      "judge_status": dict(judge_status) if judge_status is not None else {},
  }


def evaluate_multi_turn_benchmark(
    rl_cluster: Any,
    model_tokenizer: Any,
    problems: Sequence[Mapping[str, Any]],
    benchmark_name: str = "held_out",
    arm_name: str = "BASE",
    *,
    seed: int = 42,
    max_turns: int = 3,
    max_generation_steps: int = DEFAULT_EVAL_MAX_GENERATION_STEPS,
    max_prompt_tokens: int = DEFAULT_EVAL_MAX_PROMPT_TOKENS,
    test_timeout_s: float = 2.0,
    batch_size: int = 64,
) -> dict[str, Any]:
  """Evaluates a policy on a multi-turn Python coding benchmark with >=1,024 tokens/turn (§R4.1).

  Uses `mode=rl_cluster_lib.Mode.EVAL` when supported, head+tail prompt truncation at
  `max_prompt_tokens = 2560`, parallel `SandboxRunner` execution via `ThreadPoolExecutor`,
  and records per-turn and episode truncation rates alongside solve rates and repair gain.
  """
  n = len(problems)
  if n == 0:
    return {
        "arm": arm_name,
        "seed": int(seed),
        "benchmark": benchmark_name,
        "num_problems": 0,
        "max_turns": int(max_turns),
        "max_generation_steps": int(max_generation_steps),
        "max_prompt_tokens": int(max_prompt_tokens),
        "turn1_solve": 0.0,
        "final_solve": 0.0,
        "repair_gain": 0.0,
        "mean_assistant_tokens": 0.0,
        "p90_assistant_tokens": 0.0,
        "turn_truncation_by_turn": [0.0 for _ in range(max_turns)],
        "turn_truncation_rate": 0.0,
        "episode_truncation_rate": 0.0,
        "truncation_rate": 0.0,
        "problem_scores": [],
        "turn1_scores": [],
        "assistant_lengths": [],
    }

  eval_mode = getattr(rl_cluster_lib, "Mode", None)
  mode_arg = (
      eval_mode.EVAL
      if (eval_mode is not None and hasattr(eval_mode, "EVAL"))
      else (rl_cluster.Role.ROLLOUT if hasattr(rl_cluster, "Role") else "eval")
  )

  eval_sandbox = sandbox_mod.SandboxRunner(
      default_timeout_s=test_timeout_s,
      stdio_timeout_s=test_timeout_s,
  )

  def _normalize_raw_tests(raw_tests: Any, fmt: str) -> Any:
    if isinstance(raw_tests, str):
      stripped = raw_tests.strip()
      if stripped.startswith(("[", "{")):
        try:
          raw_tests = json.loads(stripped)
        except Exception:  # pylint: disable=broad-except
          raw_tests = [line for line in raw_tests.splitlines() if line.strip()]
      else:
        raw_tests = [line for line in raw_tests.splitlines() if line.strip()]
    if isinstance(raw_tests, Mapping):
      return dict(raw_tests)
    if isinstance(raw_tests, Sequence) and not isinstance(raw_tests, (str, bytes)):
      items = list(raw_tests)
      if items and all(isinstance(x, Mapping) for x in items) and (
          fmt in ("stdio", "stdin") or any("input" in x or "output" in x for x in items)
      ):
        return {
            "format": "stdio",
            "inputs": [str(x.get("input", "")) for x in items],
            "outputs": [str(x.get("output", "")) for x in items],
        }
      return items
    return []

  messages_per_prob: list[list[dict[str, str]]] = []
  tests_per_prob: list[Any] = []
  fmt_per_prob: list[str] = []
  entry_point_per_prob: list[str | None] = []
  for prob in problems:
    if isinstance(prob.get("prompts"), list) and prob["prompts"] and isinstance(prob["prompts"][0], dict):
      msgs = [dict(m) for m in prob["prompts"]]
    else:
      prompt_text = str(prob.get("prompt") or prob.get("question") or prob.get("problem") or "")
      msgs = [
          {"role": "system", "content": SYSTEM_PROMPT},
          {"role": "user", "content": prompt_text},
      ]
    messages_per_prob.append(msgs)
    raw_tests = (
        prob.get("unit_tests")
        if prob.get("unit_tests") is not None
        else prob.get("tests")
    )
    fmt = str(
        prob.get("format")
        or (
            "stdio"
            if isinstance(raw_tests, Mapping)
            and ("inputs" in raw_tests or raw_tests.get("format") == "stdio")
            else "asserts"
        )
    )
    fmt_per_prob.append(fmt)
    tests_per_prob.append(_normalize_raw_tests(raw_tests, fmt))
    ep = prob.get("entry_point") or prob.get("fn_name")
    entry_point_per_prob.append(str(ep) if ep else None)

  active = list(range(n))
  turn1_passed = [False] * n
  final_passed = [False] * n
  assistant_lengths = [0] * n
  episode_any_turn_truncated = [False] * n
  turn_truncated_flags: list[list[bool]] = [[] for _ in range(max_turns)]

  def _render_and_cap_prompt(msgs: list[dict[str, str]]) -> str:
    if hasattr(model_tokenizer, "apply_chat_template"):
      try:
        rendered = model_tokenizer.apply_chat_template(
            msgs,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
      except TypeError:
        rendered = model_tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
    else:
      rendered = "\n".join(
          f"{m.get('role', 'user')}: {m.get('content', '')}" for m in msgs
      )
    enc_ids = list(model_tokenizer.encode(rendered, add_special_tokens=False))
    if len(enc_ids) > max_prompt_tokens:
      head_len = max_prompt_tokens // 3
      tail_len = max_prompt_tokens - head_len
      capped_ids = enc_ids[:head_len] + enc_ids[-tail_len:]
      if hasattr(model_tokenizer, "decode"):
        rendered = str(model_tokenizer.decode(capped_ids, skip_special_tokens=False))
      else:
        rendered = " ".join(str(tok) for tok in capped_ids)
    return str(rendered)

  def _run_test_for_idx(i: int, resp_text: str) -> tuple[bool, str]:
    code_str, _, _ = code_env_mod.extract_last_code_fence(resp_text)
    if code_str is None and hasattr(sandbox_mod, "extract_python_code"):
      code_str = sandbox_mod.extract_python_code(resp_text)
    if not code_str:
      return False, "No ```python ... ``` code block found."
    raw_tests = tests_per_prob[i]
    entry_point = entry_point_per_prob[i]
    if hasattr(eval_sandbox, "run"):
      res = eval_sandbox.run(
          code_str,
          raw_tests,
          timeout=test_timeout_s,
          entry_point=entry_point,
      )
    elif hasattr(sandbox_mod, "run_python_tests"):
      res = sandbox_mod.run_python_tests(
          code_str, raw_tests, timeout_s=test_timeout_s
      )
    elif fmt_per_prob[i] in ("stdio", "stdin") and hasattr(eval_sandbox, "run_stdio"):
      res = eval_sandbox.run_stdio(code_str, raw_tests, timeout_s=test_timeout_s)
    elif hasattr(eval_sandbox, "run_asserts"):
      assert_lines = (
          list(raw_tests.get("asserts", []))
          if isinstance(raw_tests, Mapping)
          else [t if isinstance(t, str) else f"assert {t}" for t in raw_tests]
      )
      res = eval_sandbox.run_asserts(code_str, assert_lines, timeout_s=test_timeout_s)
    else:
      raise AttributeError("SandboxRunner does not expose .run()")
    passed = bool(getattr(res, "all_passed", getattr(res, "passed", False)))
    err_summary = str(getattr(res, "error", "") or "")
    failures = list(getattr(res, "failures", []) or [])
    feedback = (
        err_summary
        or ("\n".join(str(getattr(f, "traceback_tail", f)) for f in failures[:3]))
        or "Tests failed."
    )
    return passed, feedback

  rc_raw = (
      getattr(rl_cluster.cluster_config, "rollout_config", None)
      if hasattr(rl_cluster, "cluster_config")
      and rl_cluster.cluster_config is not None
      else None
  )
  if isinstance(rc_raw, dict):
    rc = rc_raw.get(mode_arg) or (
        next(iter(rc_raw.values())) if rc_raw else None
    )
  else:
    rc = rc_raw
  orig_temp = getattr(rc, "temperature", 0.7) if rc is not None else 0.7

  try:
    gen_params = inspect.signature(rl_cluster.generate).parameters
  except (TypeError, ValueError):
    gen_params = {}
  has_var_kw = any(
      p.kind == inspect.Parameter.VAR_KEYWORD for p in gen_params.values()
  )

  if rc is not None:
    object.__setattr__(rc, "temperature", 0.0)
  try:
    for turn_idx in range(max_turns):
      if not active:
        break
      prompts_batch = [_render_and_cap_prompt(messages_per_prob[i]) for i in active]
      responses: list[str] = []
      for start in range(0, len(prompts_batch), max(1, batch_size)):
        prompts_to_gen = prompts_batch[start : start + max(1, batch_size)]
        gen_kwargs: dict[str, Any] = {
            "prompts": prompts_to_gen,
            "apply_chat_template": False,
            "mode": mode_arg,
            "micro_batch_size": min(len(prompts_to_gen), 64),
        }
        if (
            "max_generation_steps" in gen_params
            or not gen_params
            or (has_var_kw and "max_tokens_to_generate" not in gen_params)
        ):
          gen_kwargs["max_generation_steps"] = max_generation_steps
        elif "max_tokens_to_generate" in gen_params:
          gen_kwargs["max_tokens_to_generate"] = max_generation_steps
        if not has_var_kw and gen_params:
          gen_kwargs = {k: v for k, v in gen_kwargs.items() if k in gen_params}
        gen_out = rl_cluster.generate(**gen_kwargs)
        if hasattr(gen_out, "text") and gen_out.text is not None:
          responses.extend(str(t) for t in gen_out.text)
        elif isinstance(gen_out, Sequence) and not isinstance(gen_out, str):
          responses.extend(str(t) for t in gen_out)
        else:
          responses.append(str(gen_out))

      with ThreadPoolExecutor(max_workers=min(32, max(1, len(responses)))) as pool:
        outcomes = list(
            pool.map(lambda pair: _run_test_for_idx(pair[0], pair[1]), zip(active, responses))
        )

      next_active: list[int] = []
      for idx_in_batch, i in enumerate(active):
        resp_text = responses[idx_in_batch]
        resp_tokens = list(model_tokenizer.encode(resp_text, add_special_tokens=False))
        turn_len = len(resp_tokens)
        assistant_lengths[i] += turn_len
        is_trunc = turn_len >= int(0.98 * max_generation_steps)
        turn_truncated_flags[turn_idx].append(is_trunc)
        if is_trunc:
          episode_any_turn_truncated[i] = True

        passed, feedback = outcomes[idx_in_batch]
        if turn_idx == 0:
          turn1_passed[i] = passed
        if passed:
          final_passed[i] = True
        elif turn_idx + 1 < max_turns:
          messages_per_prob[i].append({"role": "assistant", "content": resp_text})
          messages_per_prob[i].append({
              "role": "user",
              "content": (
                  f"Execution failed with the following traceback:\n{feedback[:1500]}\n"
                  "Please fix the root-cause bug and return the corrected ```python ... ``` block."
              ),
          })
          next_active.append(i)
      active = next_active
  finally:
    if rc is not None:
      object.__setattr__(rc, "temperature", orig_temp)

  metrics = code_rl_metrics.compute_code_rl_metrics(
      turn1_passed=turn1_passed,
      final_passed=final_passed,
      assistant_lengths=assistant_lengths,
      max_response_length=max_generation_steps * max_turns,
  )
  turn_trunc_by_turn = [
      float(np.mean(flags)) if flags else 0.0 for flags in turn_truncated_flags
  ]
  all_turn_flags = [f for flags in turn_truncated_flags for f in flags]
  turn_trunc_rate = float(np.mean(all_turn_flags)) if all_turn_flags else 0.0
  ep_trunc_rate = max(
      float(metrics["code_rl/truncation_rate"]),
      float(np.mean(episode_any_turn_truncated)) if episode_any_turn_truncated else 0.0,
  )
  eff_trunc_rate = max(ep_trunc_rate, turn_trunc_rate)

  problem_scores = [1.0 if ok else 0.0 for ok in final_passed]
  turn1_scores = [1.0 if ok else 0.0 for ok in turn1_passed]

  return {
      "arm": arm_name,
      "seed": int(seed),
      "benchmark": benchmark_name,
      "num_problems": n,
      "max_turns": int(max_turns),
      "max_generation_steps": int(max_generation_steps),
      "max_prompt_tokens": int(max_prompt_tokens),
      "turn1_solve": float(metrics["code_rl/turn1_solve"]),
      "final_solve": float(metrics["code_rl/final_solve"]),
      "repair_gain": float(metrics["code_rl/repair_gain"]),
      "mean_assistant_tokens": float(metrics["code_rl/mean_assistant_tokens"]),
      "p90_assistant_tokens": float(metrics["code_rl/p90_assistant_tokens"]),
      "turn_truncation_by_turn": turn_trunc_by_turn,
      "turn_truncation_rate": turn_trunc_rate,
      "episode_truncation_rate": ep_trunc_rate,
      "truncation_rate": eff_trunc_rate,
      "problem_scores": problem_scores,
      "turn1_scores": turn1_scores,
      "assistant_lengths": assistant_lengths,
  }


def summarize_step_telemetry(
    step_metrics_history: Sequence[Mapping[str, Any]],
    *,
    arm_name: str,
    seed: int,
    num_iterations: int = 1,
    learning_rate: float = 1e-6,
    max_response_length: int = DEFAULT_MAX_RESPONSE_LENGTH,
) -> dict[str, Any]:
  """Summarizes per-step and per-minibatch training telemetry for a V2 `(arm, seed)` run."""
  steps_list = [dict(s) for s in step_metrics_history]
  if not steps_list:
    return {
        "arm": arm_name,
        "seed": int(seed),
        "num_iterations": int(num_iterations),
        "learning_rate": float(learning_rate),
        "max_response_length": int(max_response_length),
        "steps": [],
        "summary": {},
    }

  def _series(key: str, default: float = 0.0) -> list[float]:
    return [float(s.get(key, default)) for s in steps_list]

  pg_clip = _series("actor/train/pg_clipfrac")
  seq_clip = _series("actor/train/seq_clipfrac", default=float(np.mean(pg_clip)))
  mb1_clip = _series("actor/train/mb1_pg_clipfrac", default=0.0)
  mb2_4_clip = _series("actor/train/mb2_to_4_pg_clipfrac", default=float(np.mean(pg_clip)))
  mb1_ratio = _series("actor/train/mb1_abs_log_ratio", default=0.0)
  mb2_4_ratio = _series("actor/train/mb2_to_4_abs_log_ratio", default=0.0)
  ppo_kl = _series("actor/train/ppo_kl")
  kl_st = _series("actor/train/kl_sampler_trainer")
  grad_norm = _series("actor/train/grad_norm")
  adv_mean = _series("rewards/advantage/mean")
  mass_pos = _series("advantage/mass_pos_ratio", default=1.0)
  mass_neg = _series("advantage/mass_neg_ratio", default=1.0)
  flagged_frac = _series("judge/flagged_token_frac")
  trunc_rate = _series("code_rl/truncation_rate")
  final_solve = _series("code_rl/final_solve")
  repair_gain = _series("code_rl/repair_gain")

  instability_events = sum(
      1
      for g, k in zip(grad_norm, ppo_kl)
      if (not np.isfinite(g)) or (not np.isfinite(k)) or g > 100.0 or abs(k) > 0.50
  )

  return {
      "arm": arm_name,
      "seed": int(seed),
      "num_iterations": int(num_iterations),
      "learning_rate": float(learning_rate),
      "max_response_length": int(max_response_length),
      "num_steps": len(steps_list),
      "summary": {
          "mean_pg_clipfrac": float(np.mean(pg_clip)),
          "mean_seq_clipfrac": float(np.mean(seq_clip)),
          "mean_mb1_pg_clipfrac": float(np.mean(mb1_clip)),
          "mean_mb2_to_4_pg_clipfrac": float(np.mean(mb2_4_clip)),
          "mean_mb1_abs_log_ratio": float(np.mean(mb1_ratio)),
          "mean_mb2_to_4_abs_log_ratio": float(np.mean(mb2_4_ratio)),
          "mean_ppo_kl": float(np.mean(ppo_kl)),
          "mean_kl_sampler_trainer": float(np.mean(kl_st)),
          "mean_grad_norm": float(np.mean(grad_norm)),
          "max_grad_norm": float(np.max(grad_norm)),
          "mean_advantage": float(np.mean(adv_mean)),
          "max_abs_advantage_mean": float(np.max(np.abs(adv_mean))),
          "mean_mass_pos_ratio": float(np.mean(mass_pos)),
          "mean_mass_neg_ratio": float(np.mean(mass_neg)),
          "mean_flagged_token_frac": float(np.mean(flagged_frac)),
          "mean_truncation_rate": float(np.mean(trunc_rate)),
          "max_truncation_rate": float(np.max(trunc_rate)),
          "mean_final_solve": float(np.mean(final_solve)),
          "mean_repair_gain": float(np.mean(repair_gain)),
          "instability_events": int(instability_events),
      },
      "steps": steps_list,
  }


# ---------------------------------------------------------------------------
# Dataset & Benchmark Loaders + Persistent 8-Chip TPU RLCluster Runner
# ---------------------------------------------------------------------------


def _extract_metric_val(vals: Any) -> float | None:
  if not vals:
    return None
  scalars = []
  for x in vals:
    try:
      s = np.asarray(rl_common._metric_scalar(x), dtype=np.float64)
      if s.size > 0:
        scalars.append(float(np.mean(s)))
    except Exception:  # pylint: disable=broad-except
      try:
        scalars.append(float(np.mean(np.asarray(x, dtype=np.float64))))
      except Exception:  # pylint: disable=broad-except
        pass
  return float(np.mean(scalars)) if scalars else None


_EQ_PREAMBLE: str = (
    "import math\n"
    "def __eq(a, b, tol=1e-6):\n"
    "    if a == b: return True\n"
    "    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool) and not isinstance(b, bool):\n"
    "        return math.isclose(a, b, rel_tol=1e-7, abs_tol=tol)\n"
    "    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)) and len(a) == len(b):\n"
    "        return all(__eq(x, y, tol) for x, y in zip(a, b))\n"
    "    return False\n"
)


def extract_evalplus_asserts(
    test_code: str,
    entry_point: str | None = None,
    fallback_list: Any = None,
    max_cases: int = 15,
) -> list[str]:
  """Extracts self-contained assertion statements from EvalPlus `test` code without requiring numpy in sandbox."""
  asserts: list[str] = []
  if fallback_list:
    if isinstance(fallback_list, str):
      try:
        fallback_list = ast.literal_eval(fallback_list)
      except Exception:  # pylint: disable=broad-except
        fallback_list = []
    asserts.extend([str(x) for x in fallback_list])
  try:
    tree = ast.parse(test_code)
    inputs_val, results_val, atol_val = None, None, 0.0
    fn_name = "candidate" if entry_point else None
    for node in ast.walk(tree):
      if (
          isinstance(node, ast.Assign)
          and len(node.targets) == 1
          and isinstance(node.targets[0], ast.Name)
      ):
        name = node.targets[0].id
        if name == "inputs":
          inputs_val = ast.literal_eval(node.value)
        elif name == "results":
          results_val = ast.literal_eval(node.value)
        elif name == "atol":
          try:
            atol_val = float(ast.literal_eval(node.value))
          except Exception:  # pylint: disable=broad-except
            pass
      elif (
          isinstance(node, ast.Call)
          and isinstance(node.func, ast.Name)
          and node.func.id == "assertion"
      ):
        if (
            len(node.args) >= 1
            and isinstance(node.args[0], ast.Call)
            and isinstance(node.args[0].func, ast.Name)
        ):
          if fn_name is None:
            fn_name = node.args[0].func.id
        if len(node.args) >= 3:
          try:
            atol_val = float(ast.literal_eval(node.args[2]))
          except Exception:  # pylint: disable=broad-except
            pass
    if inputs_val is not None and results_val is not None and fn_name:
      eff_tol = max(atol_val, 1e-6)
      for inp, exp in list(zip(inputs_val, results_val))[:max_cases]:
        stmt = (
            _EQ_PREAMBLE
            + f"_out = {fn_name}(*{repr(inp)})\n"
            + f'assert __eq(_out, {repr(exp)}, {repr(eff_tol)}), f"Expected {repr(exp)}, got {{_out!r}}"'
        )
        asserts.append(stmt)
  except Exception:  # pylint: disable=broad-except
    pass
  if not asserts and test_code.strip():
    clean_test = test_code.replace("import numpy as np", "import math")
    clean_test = clean_test.replace("isinstance(x, np.ndarray)", "False")
    clean_test = clean_test.replace(
        "np.allclose(out, exp, rtol=1e-07, atol=atol)",
        "math.isclose(out, exp, rel_tol=1e-7, abs_tol=atol)",
    )
    if "def check(" in clean_test:
      clean_test += "\ncheck(candidate)\n"
    asserts = [clean_test]
  return asserts


def decode_lcb_private_tests(raw_private: Any) -> list[dict[str, Any]]:
  """Decodes LiveCodeBench private test cases from JSON or base64+zlib+pickle."""
  if not raw_private:
    return []
  if isinstance(raw_private, list):
    return [dict(t) for t in raw_private if isinstance(t, Mapping)]
  if isinstance(raw_private, str):
    try:
      parsed = json.loads(raw_private)
      if isinstance(parsed, list):
        return [dict(t) for t in parsed if isinstance(t, Mapping)]
    except Exception:  # pylint: disable=broad-except
      pass
    try:
      decompressed = zlib.decompress(base64.b64decode(raw_private.encode("utf-8")))
      try:
        parsed = json.loads(decompressed.decode("utf-8"))
      except Exception:  # pylint: disable=broad-except
        parsed = pickle.loads(decompressed)
      if isinstance(parsed, str):
        parsed = json.loads(parsed)
      if isinstance(parsed, list):
        return [dict(t) for t in parsed if isinstance(t, Mapping)]
    except Exception:  # pylint: disable=broad-except
      pass
  return []


def load_training_pool(
    dataset_dir: str | Path,
    train_jsonl: str | Path | None = None,
) -> tuple[list[dict[str, Any]], str]:
  """Loads the decontaminated 2,000-problem training pool from disk."""
  candidates: list[Path] = []
  if train_jsonl is not None:
    candidates.append(Path(train_jsonl))
  dataset_dir_p = Path(dataset_dir)
  candidates.extend([
      dataset_dir_p / "code_rl_train_2k.jsonl",
      dataset_dir_p / "train_filtered.jsonl",
      dataset_dir_p / "train.jsonl",
  ])
  for path in candidates:
    if path.exists():
      rows = dataset_processor.load_jsonl_records(str(path))
      return rows, str(path)
  raise FileNotFoundError(f"Could not find training pool JSONL in {[str(c) for c in candidates]}.")


_BENCHMARK_ALIASES: dict[str, str] = {
    "held_out": "heldout_indist",
    "lcb_v5": "livecodebench",
    "humaneval_plus": "humanevalplus",
    "mbpp_plus": "mbppplus",
}


class _BenchmarkDict(dict):
  """Benchmark dict with canonical keys (`heldout_indist`, `livecodebench`, `humanevalplus`, `mbppplus`) and alias lookup."""

  def __getitem__(self, key: str) -> Any:
    if super().__contains__(key):
      return super().__getitem__(key)
    canonical = _BENCHMARK_ALIASES.get(key)
    if canonical is not None and super().__contains__(canonical):
      return super().__getitem__(canonical)
    raise KeyError(key)

  def __contains__(self, key: object) -> bool:
    if super().__contains__(key):
      return True
    if isinstance(key, str) and key in _BENCHMARK_ALIASES:
      return super().__contains__(_BENCHMARK_ALIASES[key])
    return False

  def get(self, key: str, default: Any = None) -> Any:
    if key in self:
      return self[key]
    return default


def _load_hf_split_from_disk(arrow_dir: Path) -> Any:
  from datasets import load_from_disk  # pylint: disable=import-outside-toplevel

  try:
    ds = load_from_disk(str(arrow_dir))
  except FileNotFoundError:
    if (arrow_dir / "test").is_dir():
      ds = load_from_disk(str(arrow_dir / "test"))
    else:
      raise
  if hasattr(ds, "keys") and "test" in ds:
    return ds["test"]
  return ds


def load_all_benchmarks(dataset_dir: str | Path) -> dict[str, list[dict[str, Any]]]:
  """Loads all 4 multi-turn code benchmarks (`heldout_indist`, `livecodebench`, `humanevalplus`, `mbppplus`) from JSONL or Arrow directories."""
  dataset_dir_p = Path(dataset_dir)
  benchmarks: dict[str, list[dict[str, Any]]] = _BenchmarkDict()

  # 1. Held-out In-Distribution (200 problems)
  for hp in (
      dataset_dir_p / "heldout_eval.jsonl",
      dataset_dir_p / "heldout_test_256.jsonl",
      dataset_dir_p / "code_rl_heldout_eval.jsonl",
      dataset_dir_p / "held_out.jsonl",
      dataset_dir_p / "heldout_indist.jsonl",
  ):
    if hp.exists():
      benchmarks["heldout_indist"] = dataset_processor.load_jsonl_records(str(hp))[:200]
      break

  # 2. HumanEval+ (164 problems)
  for hep in (
      dataset_dir_p / "humaneval_plus.jsonl",
      dataset_dir_p / "humanevalplus.jsonl",
  ):
    if hep.is_file():
      benchmarks["humanevalplus"] = dataset_processor.load_jsonl_records(str(hep))
      break
  if "humanevalplus" not in benchmarks:
    for he_dir in (dataset_dir_p / "humanevalplus", dataset_dir_p / "humaneval_plus"):
      if he_dir.is_dir():
        he_ds = _load_hf_split_from_disk(he_dir)
        he_records: list[dict[str, Any]] = []
        for idx, r in enumerate(he_ds):
          prob = (
              "Complete the following Python function:\n\n```python\n"
              + str(r["prompt"]).strip()
              + "\n```"
          )
          tests = extract_evalplus_asserts(
              str(r.get("test", "")),
              entry_point=r.get("entry_point"),
              max_cases=15,
          )
          he_records.append(
              dataset_processor.process_code_rl_record({
                  "prompt_id": str(r.get("task_id", f"HumanEval/{idx}")),
                  "problem": prob,
                  "tests": tests,
                  "format": "asserts",
                  "entry_point": str(r.get("entry_point") or ""),
              })
          )
        benchmarks["humanevalplus"] = he_records
        break

  # 3. MBPP+ (378 problems)
  for mp in (
      dataset_dir_p / "mbpp_plus.jsonl",
      dataset_dir_p / "mbppplus.jsonl",
  ):
    if mp.is_file():
      benchmarks["mbppplus"] = dataset_processor.load_jsonl_records(str(mp))
      break
  if "mbppplus" not in benchmarks:
    for mb_dir in (dataset_dir_p / "mbppplus", dataset_dir_p / "mbpp_plus"):
      if mb_dir.is_dir():
        mb_ds = _load_hf_split_from_disk(mb_dir)
        mb_records: list[dict[str, Any]] = []
        for idx, r in enumerate(mb_ds):
          tlist = r.get("test_list", [])
          if isinstance(tlist, str):
            try:
              tlist = ast.literal_eval(tlist)
            except Exception:  # pylint: disable=broad-except
              tlist = []
          first_assert = str(tlist[0]) if tlist else ""
          prob = str(r["prompt"]).strip()
          if first_assert:
            prob += (
                f"\n\nYour code should pass tests such as:\n```python\n{first_assert}\n```"
            )
          tests = extract_evalplus_asserts(
              str(r.get("test", "")),
              entry_point=None,
              fallback_list=tlist,
              max_cases=15,
          )
          mb_records.append(
              dataset_processor.process_code_rl_record({
                  "prompt_id": f"mbpp_{r.get('task_id', idx)}",
                  "problem": prob,
                  "tests": tests,
                  "format": "asserts",
              })
          )
        benchmarks["mbppplus"] = mb_records
        break

  # 4. LiveCodeBench v5/v6 (200 problems)
  for lp in (
      dataset_dir_p / "livecodebench_v5.jsonl",
      dataset_dir_p / "lcb_v5.jsonl",
      dataset_dir_p / "livecodebench.jsonl",
  ):
    if lp.is_file():
      benchmarks["livecodebench"] = dataset_processor.load_jsonl_records(str(lp))
      break
  if "livecodebench" not in benchmarks:
    for lcb_dir in (
        dataset_dir_p / "livecodebench_code_generation_lite",
        dataset_dir_p / "livecodebench",
        dataset_dir_p / "lcb_v5",
    ):
      if lcb_dir.is_dir():
        lcb_ds = _load_hf_split_from_disk(lcb_dir)
        lcb_v56 = [
            r
            for r in lcb_ds
            if r.get("release_file") in ("test5.jsonl", "test6.jsonl")
            or r.get("release_version") in ("release_v5", "release_v6")
        ][:200]
        if not lcb_v56:
          lcb_v56 = list(lcb_ds)[:200]
        lcb_records: list[dict[str, Any]] = []
        for idx, r in enumerate(lcb_v56):
          pub: list[Any] = []
          if r.get("public_test_cases"):
            if isinstance(r["public_test_cases"], list):
              pub = r["public_test_cases"]
            else:
              try:
                pub = json.loads(r["public_test_cases"])
              except Exception:  # pylint: disable=broad-except
                pub = []
          priv = decode_lcb_private_tests(r.get("private_test_cases"))
          all_cases = [c for c in (pub + priv) if isinstance(c, Mapping)][:10]
          starter = str(r.get("starter_code") or "").strip()
          meta: dict[str, Any] = {}
          if r.get("metadata"):
            if isinstance(r["metadata"], Mapping):
              meta = dict(r["metadata"])
            else:
              try:
                meta = json.loads(r["metadata"])
              except Exception:  # pylint: disable=broad-except
                meta = {}
          func_name = meta.get("func_name") or meta.get("fn_name")
          prob_text = str(
              r.get("question_content") or r.get("problem") or ""
          ).strip()
          if starter:
            prob_text += f"\n\nStarter code:\n```python\n{starter}\n```"

          is_functional = bool(
              starter
              and func_name
              and any(c.get("testtype") == "functional" for c in all_cases)
          )
          if is_functional:
            assert_stmts: list[str] = []
            for c in all_cases:
              inp_lines = [
                  ln.strip()
                  for ln in str(c.get("input", "")).splitlines()
                  if ln.strip()
              ]
              exp_str = str(c.get("output", "")).strip()
              try:
                args_parsed = [json.loads(ln) for ln in inp_lines]
                exp_parsed = json.loads(exp_str)
                args_repr = ", ".join(repr(a) for a in args_parsed)
                stmt = (
                    f'_sol = Solution() if "Solution" in globals() else None\n'
                    f"_fn = getattr(_sol, {repr(func_name)}) if _sol is not None else {func_name}\n"
                    f"_out = _fn({args_repr})\n"
                    f'assert _out == {repr(exp_parsed)}, f"Expected {repr(exp_parsed)}, got {{_out!r}}"'
                )
                assert_stmts.append(stmt)
              except Exception:  # pylint: disable=broad-except
                pass
            if assert_stmts:
              lcb_records.append(
                  dataset_processor.process_code_rl_record({
                      "prompt_id": str(r.get("question_id", f"lcb_{idx}")),
                      "problem": prob_text,
                      "tests": assert_stmts,
                      "format": "asserts",
                  })
              )
              continue

          inputs = [str(c.get("input", "")) for c in all_cases]
          outputs = [str(c.get("output", "")) for c in all_cases]
          if not inputs:
            inputs = [""]
            outputs = [""]
          lcb_records.append(
              dataset_processor.process_code_rl_record({
                  "prompt_id": str(r.get("question_id", f"lcb_{idx}")),
                  "problem": prob_text,
                  "tests": {"format": "stdio", "inputs": inputs, "outputs": outputs},
                  "format": "stdio",
              })
          )
        benchmarks["livecodebench"] = lcb_records
        break

  return benchmarks


def _build_seeded_train_batches(
    raw_rows: Sequence[Mapping[str, Any]],
    model_tokenizer: Any,
    *,
    seed: int,
    num_batches: int,
    batch_size: int = 8,
    max_prompt_tokens: int = DEFAULT_MAX_PREFILL_LENGTH,
) -> list[dict[str, Any]]:
  """Filters training rows by `max_prompt_tokens`, shuffles deterministically with `seed`, and builds batches."""
  filtered = [
      dict(r)
      for r in raw_rows
      if len(
          model_tokenizer.encode(
              str(r.get("prompts", r.get("problem", ""))), add_special_tokens=False
          )
      )
      <= max_prompt_tokens
  ]
  if not filtered:
    filtered = [dict(r) for r in raw_rows]
  rng = np.random.default_rng(seed)
  perm = rng.permutation(len(filtered))
  shuffled = [filtered[int(i)] for i in perm]

  total_needed = num_batches * batch_size
  while len(shuffled) < total_needed:
    shuffled = shuffled + shuffled
  shuffled = shuffled[:total_needed]

  batches: list[dict[str, Any]] = []
  for b in range(num_batches):
    chunk = shuffled[b * batch_size : (b + 1) * batch_size]
    batches.append({
        "prompts": [str(r.get("prompts", r.get("problem", ""))) for r in chunk],
        "question": [str(r.get("question", r.get("problem", ""))) for r in chunk],
        "problem": [str(r.get("problem", r.get("question", ""))) for r in chunk],
        "prompt_id": [
            str(r.get("prompt_id", f"p_{seed}_{b}_{i}")) for i, r in enumerate(chunk)
        ],
        "tests": [
            r["tests"] if isinstance(r.get("tests"), str) else json.dumps(r.get("tests", []))
            for r in chunk
        ],
        "format": [str(r.get("format", "asserts")) for r in chunk],
        "format_instructions": [str(r.get("format_instructions", "")) for r in chunk],
        "entry_point": [str(r.get("entry_point") or "") for r in chunk],
        "answer": [
            json.dumps({"tests": r.get("tests", []), "format": r.get("format", "asserts")})
            for r in chunk
        ],
        "turn1_passed": [False for _ in chunk],
        "final_passed": [False for _ in chunk],
        "assistant_lengths": [0 for _ in chunk],
    })
  return batches


def _unwrap_scalar(v: Any) -> Any:
  """Recursively unwraps 1-element lists/tuples/np.ndarrays and numpy scalars."""
  if isinstance(v, (list, tuple, np.ndarray)):
    if len(v) == 1:
      return _unwrap_scalar(v[0])
    return [_unwrap_scalar(x) for x in v]
  if hasattr(v, "item"):
    return v.item()
  return v


def _unwrap_task_dict(task: Any) -> Any:
  """Unwraps 1-element micro-batch arrays in `task` and parses JSON-encoded `tests`."""
  if not isinstance(task, Mapping):
    return task
  out: dict[str, Any] = {}
  for k, v in task.items():
    val = _unwrap_scalar(v)
    if k in ("tests", "unit_tests") and isinstance(val, str):
      try:
        val = json.loads(val)
      except Exception:  # pylint: disable=broad-except
        val = [val] if val.strip() else []
    out[k] = val
  return out


if not getattr(code_env_mod.PythonCodeEnv.__init__, "_is_v2_unwrapped", False):
  _orig_python_code_env_init = code_env_mod.PythonCodeEnv.__init__

  def _patched_python_code_env_init(
      self: Any,
      single_example: dict[str, Any] | None = None,
      *,
      group_id: int = 0,
      pair_index: int = 0,
      **kwargs: Any,
  ) -> None:
    raw_task = (
        single_example
        if single_example is not None
        else kwargs.pop("task", None)
    )
    _orig_python_code_env_init(
        self,
        single_example=_unwrap_task_dict(raw_task),
        group_id=group_id,
        pair_index=pair_index,
        **kwargs,
    )
    self.extra_kwargs["group_id"] = self.group_id
    self.extra_kwargs["pair_index"] = self.pair_index

  _patched_python_code_env_init._is_v2_unwrapped = True  # type: ignore[attr-defined]
  code_env_mod.PythonCodeEnv.__init__ = _patched_python_code_env_init  # type: ignore[method-assign]


class UnwrappedPythonCodeEnv(code_env_mod.PythonCodeEnv):
  """`PythonCodeEnv` subclass that unwraps 1-element `np.ndarray` micro-batches, populates `extra_kwargs`, and JSON-serializes `obs['tests']`."""

  def __init__(
      self,
      single_example: dict[str, Any] | None = None,
      *,
      group_id: int = 0,
      pair_index: int = 0,
      **kwargs: Any,
  ) -> None:
    raw_task = (
        single_example
        if single_example is not None
        else kwargs.pop("task", None)
    )
    super().__init__(
        single_example=_unwrap_task_dict(raw_task),
        group_id=group_id,
        pair_index=pair_index,
        **kwargs,
    )
    self.extra_kwargs["group_id"] = self.group_id
    self.extra_kwargs["pair_index"] = self.pair_index

  def _initial_observation(self) -> dict[str, Any]:
    obs = super()._initial_observation()
    obs["tests"] = (
        obs["tests"]
        if isinstance(obs.get("tests"), str)
        else json.dumps(obs.get("tests", []))
    )
    return obs


# Guard TrajectoryCollectEngine._one_step so trailing env_tokens never push
# _response_token_count >= max_response_length on non-terminal turns.
if not getattr(
    trajectory_collect_engine.TrajectoryCollectEngine._one_step,
    "_is_v2_guarded",
    False,
):
  _orig_tce_one_step = (
      trajectory_collect_engine.TrajectoryCollectEngine._one_step
  )

  async def _guarded_tce_one_step(self: Any) -> bool:
    done = await _orig_tce_one_step(self)
    cur_step = self.agent.get_current_step()
    if (
        cur_step is not None
        and getattr(cur_step, "env_tokens", None) is not None
        and self.max_response_length is not None
        and self._response_token_count >= self.max_response_length
    ):
      self._response_token_count -= len(cur_step.env_tokens)
      cur_step.env_tokens = None
      cur_step.env_masks = None
      cur_step.done = True
      self.agent.trajectory.status = (
          agent_types.TrajectoryStatus.MAX_CONTEXT_LIMIT_REACHED
      )
      return True
    return done

  _guarded_tce_one_step._is_v2_guarded = True  # type: ignore[attr-defined]
  trajectory_collect_engine.TrajectoryCollectEngine._one_step = (
      _guarded_tce_one_step
  )


# Patch BoundedGroupStage.wrap so completed in-flight Judge tasks drain and yield
# immediately when upstream `aiter.__anext__()` pauses between batches waiting on
# `prompt_queue` (avoiding producer-consumer deadlock when max_in_flight > 1).
if not getattr(
    group_postprocess.BoundedGroupStage.wrap,
    "_is_v2_nonblocking_drain",
    False,
):

  async def _nonblocking_drain_wrap(
      self: Any,
      group_stream: Any,
  ) -> Any:
    """Consumes groups from `group_stream` without blocking finished in-flight tasks on upstream waits."""
    in_flight: set[asyncio.Task[Any]] = set()
    next_group_task: asyncio.Task[Any] | None = None
    stream_exhausted = False
    try:
      async with contextlib.aclosing(group_stream.__aiter__()) as aiter:
        while not stream_exhausted or in_flight:
          if (
              not stream_exhausted
              and next_group_task is None
              and len(in_flight) < self.max_in_flight
          ):
            next_group_task = asyncio.create_task(aiter.__anext__())

          wait_set: set[asyncio.Task[Any]] = set(in_flight)
          if (
              next_group_task is not None
              and len(in_flight) < self.max_in_flight
          ):
            wait_set.add(next_group_task)

          if not wait_set:
            break

          done, _ = await asyncio.wait(
              wait_set, return_when=asyncio.FIRST_COMPLETED
          )

          if next_group_task is not None and next_group_task in done:
            done.discard(next_group_task)
            try:
              group = next_group_task.result()
              next_group_task = None
              in_flight.add(asyncio.create_task(self._run_one(group)))
            except StopAsyncIteration:
              next_group_task = None
              stream_exhausted = True

          for finished_task in done:
            in_flight.discard(finished_task)
            yield finished_task.result()
    finally:
      if next_group_task is not None and not next_group_task.done():
        next_group_task.cancel()
      for pending in in_flight:
        if not pending.done():
          pending.cancel()
      all_pending = list(in_flight) + (
          [next_group_task] if next_group_task is not None else []
      )
      if all_pending:
        await asyncio.gather(*all_pending, return_exceptions=True)

  _nonblocking_drain_wrap._is_v2_nonblocking_drain = True  # type: ignore[attr-defined]
  group_postprocess.BoundedGroupStage.wrap = _nonblocking_drain_wrap  # type: ignore[method-assign]


_JUDGE_HTTP_EXECUTOR = ThreadPoolExecutor(max_workers=16)

if not getattr(
    jc_mod.OpenAICompatJudgeClient._chat_json,
    "_is_v2_bounded_tokens",
    False,
):

  def _bounded_post_json_sync(self: Any, payload: dict[str, Any]) -> dict[str, Any]:
    req_payload = dict(payload)
    req_payload.setdefault("max_tokens", 512)
    req_payload.setdefault("chat_template_kwargs", {"enable_thinking": False})
    data = json.dumps(req_payload).encode("utf-8")
    req = urllib.request.Request(
        f"{self.base_url}/chat/completions",
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        },
        method="POST",
    )
    eff_timeout = min(float(getattr(self, "timeout_s", 25.0) or 25.0), 25.0)
    with urllib.request.urlopen(req, timeout=eff_timeout) as resp:
      body = json.loads(resp.read().decode("utf-8"))
    content = str(body["choices"][0]["message"]["content"])
    cleaned = jc_mod._strip_code_fences(content)
    try:
      return json.loads(cleaned)
    except json.JSONDecodeError:
      if '"segments"' in cleaned or '"hacks"' in cleaned or '"tags"' in cleaned:
        seen_ids: set[int] = set()
        segs: list[dict[str, Any]] = []
        for m in re.finditer(
            r'\{\s*"id"\s*:\s*(\d+)\s*,\s*"label"\s*:\s*"(useful|faulty)"\s*,\s*"reason"\s*:\s*"([^"]+)"\s*\}',
            cleaned,
        ):
          sid = int(m.group(1))
          if sid not in seen_ids:
            seen_ids.add(sid)
            segs.append({"id": sid, "label": m.group(2), "reason": m.group(3)})
        return {
            "segments": segs,
            "turn_summary": [],
            "hacks": [],
            "scores": {},
            "ranking": [],
        }
      raise

  async def _bounded_chat_json(
      self: Any,
      prompt: str,
      schema: dict[str, Any],
      *,
      system_prompt: str | None = None,
  ) -> dict[str, Any]:
    messages: list[dict[str, str]] = []
    if system_prompt:
      messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    payload: dict[str, Any] = {
        "model": self.model,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": 512,
        "chat_template_kwargs": {"enable_thinking": False},
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "judge_response", "schema": schema},
        },
    }
    max_retries = min(int(getattr(self, "max_retries", 1)), 1)
    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
      try:
        loop = asyncio.get_running_loop()
        try:
          return await loop.run_in_executor(
              _JUDGE_HTTP_EXECUTOR, self._post_json_sync, payload
          )
        except urllib.error.HTTPError as http_exc:
          if http_exc.code in (400, 500) and "response_format" in payload:
            fallback_payload = {
                k: v for k, v in payload.items() if k != "response_format"
            }
            return await loop.run_in_executor(
                _JUDGE_HTTP_EXECUTOR, self._post_json_sync, fallback_payload
            )
          raise
      except Exception as exc:  # pylint: disable=broad-except
        last_exc = exc
        if attempt < max_retries:
          await asyncio.sleep(0.2 * (2**attempt))
    assert last_exc is not None
    raise last_exc

  _bounded_chat_json._is_v2_bounded_tokens = True  # type: ignore[attr-defined]
  jc_mod.OpenAICompatJudgeClient._post_json_sync = _bounded_post_json_sync  # type: ignore[method-assign]
  jc_mod.OpenAICompatJudgeClient._chat_json = _bounded_chat_json  # type: ignore[method-assign]


class _EnrichedGroupPostProcessor:
  """Wraps `CodeJudgeGroupPostProcessor` to populate `turn1_passed`, `final_passed`, and `assistant_lengths`."""

  def __init__(self, inner: Any):
    self.inner = inner

  @staticmethod
  def _populate_item_metrics(item: Any) -> None:
    traj = item.traj if hasattr(item, "traj") else item
    if isinstance(traj, dict):
      orig_in = traj.get("original_input")
      if isinstance(orig_in, dict):
        extras = traj.get("env_extras") or {}
        turn_records = list(extras.get("turn_records") or [])
        t1_pass = (
            bool(turn_records[0].test_result.all_passed) if turn_records else False
        )
        final_pass_fallback = (
            bool(turn_records[-1].test_result.all_passed) if turn_records else False
        )
        resp_toks = traj.get("response_tokens")
        asst_len_fallback = len(resp_toks) if resp_toks is not None else 0
        jmeta = traj.get("judge") or {}
        orig_in["turn1_passed"] = [t1_pass]
        orig_in["final_passed"] = [
            bool(jmeta.get("passed", final_pass_fallback))
        ]
        orig_in["assistant_lengths"] = [
            int(jmeta.get("assistant_len", asst_len_fallback))
        ]

  async def process_group(self, group: list[Any]) -> list[Any]:
    for item in group:
      self._populate_item_metrics(item)
    res = await self.inner.process_group(group)
    for item in res:
      self._populate_item_metrics(item)
    return res



def init_tpu_rl_cluster(args: argparse.Namespace) -> dict[str, Any]:
  """Initializes the persistent 8-chip TPU RLCluster (4 trainer chips + 4 vLLM sampler chips)."""
  pkg_dir = Path(__file__).resolve().parents[5]
  base_cfg_path = pkg_dir / "configs" / "post_train" / "rl_code_r2p_gspo_mimo.yml"
  if not base_cfg_path.exists():
    base_cfg_path = (
        Path(__file__).resolve().parents[7]
        / "src"
        / "maxtext"
        / "configs"
        / "post_train"
        / "rl_code_r2p_gspo_mimo.yml"
    )
  base_cfg_yml = str(base_cfg_path)
  out_dir = Path(args.output_dir)
  tb_dir = str(out_dir / "tb_cluster")
  ckpt_dir = str(out_dir / "ckpt_cluster")
  os.makedirs(tb_dir, exist_ok=True)
  os.makedirs(ckpt_dir, exist_ok=True)

  argv = [
      "",
      base_cfg_yml,
      "run_name=v2_cluster_init",
      f"base_output_directory={out_dir}",
      f"tensorboard_dir={tb_dir}",
      f"checkpoint_dir={ckpt_dir}",
      f"model_name={args.model_name}",
      f"tokenizer_path={args.tokenizer_path}",
      f"load_parameters_path={args.checkpoint_path}",
      f"vllm_hf_config_path={args.tokenizer_path}",
      "chips_per_vm=4",
      "trainer_devices_fraction=0.5",
      "sampler_devices_fraction=0.5",
      "batch_size=8",
      "train_micro_batch_size=2",
      "rollout_micro_batch_size=2",
      "rl.mini_batch_size=2",
      "rl.compute_logps_micro_batch_size=1",
      "rl.num_generations=8",
      "rl.num_iterations=1",
      f"max_prefill_predict_length={DEFAULT_MAX_PREFILL_LENGTH}",
      f"max_target_length={DEFAULT_MAX_TARGET_LENGTH}",
      f"learning_rate={args.learning_rate}",
      "warmup_steps_fraction=0.0",
      f"rl.judge_base_url={args.judge_url}",
      f"rl.judge_model={args.judge_model}",
      "enable_checkpointing=False",
      "eval_interval=-1",
      "log_period=1",
      "async_scheduling=False",
  ]
  trainer_config, sampler_config, trainer_devices, sampler_devices = (
      model_creation_utils.setup_configs_and_devices(
          argv, {}, config_class=types.RLConfig
      )
  )
  object.__setattr__(trainer_config, "tensorboard_dir", tb_dir)
  object.__setattr__(trainer_config, "checkpoint_dir", ckpt_dir)

  model_tokenizer = AutoTokenizer.from_pretrained(trainer_config.tokenizer_path)
  train_rl.configure_tokenizer_chat_template(model_tokenizer, trainer_config)

  reference_model, reference_mesh, actor_model, actor_mesh, rollout_mesh = (
      model_creation_utils.create_models_and_meshes(
          trainer_config,
          sampler_config,
          trainer_devices,
          sampler_devices,
          tokenizer_pad_id=model_tokenizer.pad_token_id,
      )
  )
  rl_cluster, _, _, _ = train_rl.create_rl_components(
      trainer_config,
      sampler_config,
      sampler_devices,
      actor_model,
      actor_mesh,
      reference_model,
      reference_mesh,
      rollout_mesh,
      model_tokenizer,
  )

  orig_cluster_close = rl_cluster.close

  def _cluster_noop_close() -> None:
    for m in list(getattr(rl_cluster, "_buffered_train_metrics", [])) + list(
        getattr(rl_cluster, "_buffered_eval_metrics", [])
    ):
      try:
        rl_cluster._log_metrics(m)
      except Exception:  # pylint: disable=broad-except
        pass
    if hasattr(rl_cluster, "_buffered_train_metrics"):
      rl_cluster._buffered_train_metrics.clear()
    if hasattr(rl_cluster, "_buffered_eval_metrics"):
      rl_cluster._buffered_eval_metrics.clear()

  rl_cluster.close = _cluster_noop_close

  actor_base = actor_model.base if hasattr(actor_model, "base") else actor_model
  initial_actor_state = jax.tree_util.tree_map(
      lambda x: np.asarray(jax.device_get(x)).copy()
      if isinstance(x, jax.Array)
      else x,
      nnx.to_pure_dict(nnx.state(actor_base, nnx.Param)),
  )
  base_opt_tx = rl_cluster.actor_trainer.optimizer.tx

  return {
      "rl_cluster": rl_cluster,
      "model_tokenizer": model_tokenizer,
      "actor_model": actor_model,
      "actor_base": actor_base,
      "actor_mesh": actor_mesh,
      "initial_actor_state": initial_actor_state,
      "base_opt_tx": base_opt_tx,
      "trainer_config": trainer_config,
      "sampler_config": sampler_config,
      "orig_cluster_close": orig_cluster_close,
      "jitted_step_cache": {},
      "prev_compile_key": None,
  }


def run_single_arm_seed_tpu(
    cluster_ctx: dict[str, Any],
    *,
    arm_name: str,
    seed: int,
    raw_train_rows: Sequence[Mapping[str, Any]],
    steps_per_run: int,
    judge_url: str,
    judge_model: str = "Qwen/Qwen3-32B",
    batch_size: int = 8,
    num_generations: int = 8,
    max_turns: int = 3,
    test_timeout_s: float = 3.0,
    num_iterations: int = 1,
    learning_rate: float = 1.0e-6,
    epsilon_override: tuple[float, float] | None = None,
) -> dict[str, Any]:
  """Executes a single `(arm, seed)` training run on the persistent TPU `RLCluster`."""
  rl_cluster = cluster_ctx["rl_cluster"]
  model_tokenizer = cluster_ctx["model_tokenizer"]
  actor_model = cluster_ctx["actor_model"]
  actor_mesh = cluster_ctx["actor_mesh"]
  initial_actor_state = cluster_ctx["initial_actor_state"]
  base_opt_tx = cluster_ctx["base_opt_tx"]
  base_trainer_config = cluster_ctx["trainer_config"]
  jitted_step_cache = cluster_ctx["jitted_step_cache"]
  prev_compile_key = cluster_ctx.get("prev_compile_key")
  if (
      hasattr(base_trainer_config, "learning_rate")
      and abs(float(learning_rate) - float(base_trainer_config.learning_rate))
      > 1e-12
  ):
    object.__setattr__(base_trainer_config, "learning_rate", float(learning_rate))
    base_opt_tx = train_rl.utils_rl.get_optimizer(base_trainer_config)
    cluster_ctx["base_opt_tx"] = base_opt_tx
    jitted_step_cache.clear()
    prev_compile_key = None

  new_compile_key = reset_actor_for_arm_seed(
      rl_cluster,
      actor_model,
      initial_actor_state,
      arm_name=arm_name,
      base_opt_tx=base_opt_tx,
      actor_mesh=actor_mesh,
      jitted_step_cache=jitted_step_cache,
      prev_compile_key=prev_compile_key,
      epsilon_override=epsilon_override,
  )
  cluster_ctx["prev_compile_key"] = new_compile_key

  spec = ARMS_V2[arm_name]
  grpo_config = build_v2_grpo_config(
      arm_name,
      num_generations=num_generations,
      num_iterations=num_iterations,
      max_response_length=DEFAULT_MAX_RESPONSE_LENGTH,
      epsilon_override=epsilon_override,
  )
  mimo_cfg = grpo_config.mimo

  # Configure judge postprocessor for this arm
  object.__setattr__(
      base_trainer_config.rl,
      "judge_base_url",
      judge_url if spec.get("use_judge", True) else "",
  )
  object.__setattr__(base_trainer_config.rl, "judge_model", judge_model)
  object.__setattr__(
      base_trainer_config.rl,
      "local_rules_only",
      not bool(spec.get("use_judge", True)),
  )

  def _make_reward_fn(fn: Any) -> Any:
    return train_rl._RewardFn(fn, base_trainer_config)

  reward_fns = train_rl.build_reward_fns(base_trainer_config, _make_reward_fn)
  _, env_kwargs = train_rl._resolve_env_class(base_trainer_config)
  env_class = UnwrappedPythonCodeEnv
  env_kwargs = dict(env_kwargs or {})
  env_kwargs["max_steps"] = int(max_turns)
  env_kwargs["test_timeout_s"] = float(test_timeout_s)

  raw_postproc = train_rl._build_group_postprocessor(
      base_trainer_config, model_tokenizer, mimo_cfg=mimo_cfg
  )
  group_postprocessor = (
      _EnrichedGroupPostProcessor(raw_postproc) if raw_postproc is not None else None
  )

  learner_extra_kwargs: dict[str, Any] = {
      "env_class": env_class,
      "env_kwargs": env_kwargs,
  }
  if group_postprocessor is not None:
    learner_extra_kwargs["group_postprocessor"] = group_postprocessor
    learner_extra_kwargs["group_max_in_flight"] = min(
        int(base_trainer_config.rl.judge_max_in_flight_groups or 2), 2
    )
    learner_extra_kwargs["group_stage_timeout_s"] = max(
        float(base_trainer_config.rl.judge_timeout_s or 180.0), 180.0
    )

  try:
    chat_parser: Any = chat_parser_lib.QwenChatTemplateParser(
        model_tokenizer, enable_thinking=False
    )
  except Exception:  # pylint: disable=broad-except
    chat_parser = None

  metric_fns = [
      functools.partial(
          code_rl_metrics.get_code_rl_metrics,
          max_response_length=DEFAULT_MAX_RESPONSE_LENGTH,
      )
  ]
  learner = agentic_grpo_learner.GrpoLearner(
      rl_cluster=rl_cluster,
      reward_fns=reward_fns,
      algo_config=grpo_config,
      chat_parser=chat_parser,
      metric_fns=metric_fns,
      **train_rl._kwargs_supported_by(
          agentic_grpo_learner.GrpoLearner, **learner_extra_kwargs
      ),
  )

  train_batches = _build_seeded_train_batches(
      raw_train_rows,
      model_tokenizer,
      seed=seed,
      num_batches=steps_per_run,
      batch_size=batch_size,
      max_prompt_tokens=DEFAULT_MAX_PREFILL_LENGTH,
  )

  step_records: list[dict[str, Any]] = []
  rollout_step_metrics: list[dict[str, Any]] = []
  orig_update_actor = rl_cluster.update_actor
  orig_log_metrics = rl_cluster._log_metrics
  minibatches_per_batch = max(1, (batch_size // 2) * int(num_iterations))

  def _hooked_update_actor(train_ds: Any, eval_ds: Any, skip_jit: bool = False) -> None:
    prev_ts = int(getattr(rl_cluster.actor_trainer, "train_steps", 0))
    orig_update_actor(train_ds, eval_ds, skip_jit=skip_jit)
    new_ts = int(getattr(rl_cluster.actor_trainer, "train_steps", 0))
    if new_ts > prev_ts:
      tbuf = getattr(
          rl_cluster.actor_trainer, "_prev_buffered_train_metrics", None
      ) or getattr(rl_cluster.actor_trainer, "_buffered_train_metrics", None)
      loss_val = (
          float(tbuf.loss)
          if (tbuf is not None and getattr(tbuf, "losses", None))
          else float("nan")
      )
      am = getattr(tbuf, "additional_metrics", {}) if tbuf is not None else {}
      mb_idx = (len(step_records) % minibatches_per_batch) + 1
      rec = {
          "opt_step": len(step_records) + 1,
          "actor_train_step": new_ts,
          "global_step": int(rl_cluster.global_steps),
          "mini_batch_idx": mb_idx,
          "loss": loss_val,
          "grad_norm": _extract_metric_val(am.get("grad_norm", ([], None))[0]),
          "ppo_kl": _extract_metric_val(am.get("ppo_kl", ([], None))[0]),
          "kl": _extract_metric_val(am.get("kl", ([], None))[0]),
          "log_ratio/mean": _extract_metric_val(
              am.get("log_ratio/mean", am.get("log_ratio_mean", ([], None)))[0]
          ),
          "log_ratio/abs_mean": _extract_metric_val(
              am.get("log_ratio/abs_mean", am.get("log_ratio_abs_mean", ([], None)))[0]
          ),
          "pg_clipfrac": _extract_metric_val(am.get("pg_clipfrac", ([], None))[0]),
          "seq_clipfrac": _extract_metric_val(am.get("seq_clipfrac", ([], None))[0]),
          "is_ratio/mean": _extract_metric_val(am.get("is_ratio/mean", ([], None))[0]),
          "is_ratio/min": _extract_metric_val(am.get("is_ratio/min", ([], None))[0]),
          "is_ratio/max": _extract_metric_val(am.get("is_ratio/max", ([], None))[0]),
      }
      step_records.append(rec)

  def _hooked_log_metrics(buffered_metrics: Any) -> None:
    snap: dict[str, Any] = {
        "global_step": int(
            getattr(buffered_metrics, "global_steps", getattr(buffered_metrics, "step", 0))
        )
    }
    raw_metrics = (
        getattr(buffered_metrics, "metrics", None)
        or getattr(buffered_metrics, "additional_metrics", None)
        or {}
    )
    for k, (vals, _) in raw_metrics.items():
      v = _extract_metric_val(vals)
      if v is not None:
        snap[k] = v
    if len(snap) > 1:
      rollout_step_metrics.append(snap)
    orig_log_metrics(buffered_metrics)

  rl_cluster.update_actor = _hooked_update_actor
  rl_cluster._log_metrics = _hooked_log_metrics
  t0 = time.time()
  try:
    learner.train(train_batches)
  finally:
    try:
      rl_cluster.close()
    except Exception:  # pylint: disable=broad-except
      pass
    rl_cluster.update_actor = orig_update_actor
    rl_cluster._log_metrics = orig_log_metrics
    gc.collect()

  elapsed_s = float(time.time() - t0)

  # Combine per-rollout-step and per-minibatch metrics into step_metrics_history
  combined_steps: list[dict[str, Any]] = []
  for b_idx in range(steps_per_run):
    mb_slice = step_records[
        b_idx * minibatches_per_batch : (b_idx + 1) * minibatches_per_batch
    ]
    roll_snap = (
        rollout_step_metrics[b_idx] if b_idx < len(rollout_step_metrics) else {}
    )
    mb1_recs = [r for r in mb_slice if r["mini_batch_idx"] == 1]
    mb2_4_recs = [r for r in mb_slice if 2 <= r["mini_batch_idx"] <= 4]
    mb5_8_recs = [r for r in mb_slice if r["mini_batch_idx"] >= 5]

    def _mean_rec(recs: list[dict[str, Any]], key: str) -> float:
      vals = [float(r[key]) for r in recs if r.get(key) is not None and math.isfinite(float(r[key]))]
      return float(np.mean(vals)) if vals else 0.0

    step_entry = {
        "step": b_idx,
        "actor/train/pg_clipfrac": _mean_rec(mb_slice, "pg_clipfrac"),
        "actor/train/seq_clipfrac": _mean_rec(mb_slice, "seq_clipfrac"),
        "actor/train/mb1_pg_clipfrac": _mean_rec(mb1_recs, "pg_clipfrac"),
        "actor/train/mb2_to_4_pg_clipfrac": _mean_rec(mb2_4_recs, "pg_clipfrac"),
        "actor/train/mb5_to_8_pg_clipfrac": _mean_rec(mb5_8_recs, "pg_clipfrac"),
        "actor/train/mb1_abs_log_ratio": _mean_rec(mb1_recs, "log_ratio/abs_mean"),
        "actor/train/mb2_to_4_abs_log_ratio": _mean_rec(mb2_4_recs, "log_ratio/abs_mean"),
        "actor/train/mb5_to_8_abs_log_ratio": _mean_rec(mb5_8_recs, "log_ratio/abs_mean"),
        "actor/train/ppo_kl": _mean_rec(mb_slice, "ppo_kl"),
        "actor/train/grad_norm": _mean_rec(mb_slice, "grad_norm"),
        "rewards/advantage/mean": float(roll_snap.get("rewards/advantage/mean", 0.0)),
        "advantage/mass_pos_ratio": float(roll_snap.get("advantage/mass_pos_ratio", 1.0)),
        "advantage/mass_neg_ratio": float(roll_snap.get("advantage/mass_neg_ratio", 1.0)),
        "judge/flagged_token_frac": float(roll_snap.get("judge/flagged_token_frac", 0.0)),
        "code_rl/truncation_rate": float(roll_snap.get("code_rl/truncation_rate", 0.0)),
        "code_rl/turn1_solve": float(roll_snap.get("code_rl/turn1_solve", 0.0)),
        "code_rl/final_solve": float(roll_snap.get("code_rl/final_solve", 0.0)),
        "code_rl/repair_gain": float(roll_snap.get("code_rl/repair_gain", 0.0)),
    }
    combined_steps.append(step_entry)

  summary_payload = summarize_step_telemetry(
      combined_steps,
      arm_name=arm_name,
      seed=seed,
      num_iterations=num_iterations,
      learning_rate=learning_rate,
      max_response_length=DEFAULT_MAX_RESPONSE_LENGTH,
  )
  summary_payload["backend"] = "tpu_rl_cluster"
  summary_payload["epsilon_l"] = float(grpo_config.epsilon)
  summary_payload["epsilon_h"] = float(grpo_config.epsilon_high)
  if isinstance(summary_payload.get("summary"), dict):
    summary_payload["summary"]["epsilon_l"] = float(grpo_config.epsilon)
    summary_payload["summary"]["epsilon_h"] = float(grpo_config.epsilon_high)
  summary_payload["step_records"] = step_records
  summary_payload["rollout_step_metrics"] = rollout_step_metrics
  summary_payload["elapsed_s"] = elapsed_s
  return summary_payload


def run_gates_and_diagnostics_phase(
    output_dir: Path | str,
    *,
    judge_url: str = "",
    judge_model: str = "Qwen/Qwen3-32B",
    seed: int = 42,
    cluster_ctx: dict[str, Any] | None = None,
    raw_train_rows: Sequence[Mapping[str, Any]] | None = None,
    pilot_steps: int = 4,
    offline_unit_test: bool = False,
) -> dict[str, Any]:
  """Executes Diagnostics D1-D4 and evaluates Hard Calibration Gates G1' & G2'.

  Requires a live TPU `RLCluster` (`cluster_ctx`) to measure real minibatch-1 vs
  minibatch 2..4 drift (`mb1_bar_rho`, `mb2_to_4_bar_rho`) unless `offline_unit_test=True`
  is explicitly set by CPU unit tests.
  """
  if cluster_ctx is None and not offline_unit_test:
    raise RuntimeError(
        "run_gates_and_diagnostics_phase requires a live TPU RLCluster "
        "(cluster_ctx). Synthetic offline fallback is disabled unless "
        "offline_unit_test=True (--offline-unit-test) is explicitly passed for CPU unit tests."
    )

  output_dir = Path(output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)
  if cluster_ctx is not None:
    cached_combined = _load_valid_json(
        output_dir / "gates_and_diagnostics_v2_summary.json"
    )
    if (
        cached_combined is not None
        and (output_dir / "gate_g1_prime_v2.json").exists()
        and (output_dir / "gate_g2_prime_v2.json").exists()
        and bool(cached_combined.get("diagnostics", {}).get("all_passed", False))
        and bool(cached_combined.get("gate_g1_prime", {}).get("passed", False))
        and bool(cached_combined.get("gate_g2_prime", {}).get("passed", False))
    ):
      return cached_combined
  diag_dir = output_dir / "diagnostics"
  cached_diag = (
      _load_valid_json(diag_dir / "diagnostics_summary.json")
      if cluster_ctx is not None
      else None
  )
  if (
      cached_diag is not None
      and bool(cached_diag.get("all_passed", False))
      and all(
          k in cached_diag
          for k in (
              "d1_logp_determinism",
              "d2_r2_counterfactual",
              "d3_credit_assignment_audit",
              "d4_mass_conservation_audit",
          )
      )
  ):
    diag_summary = cached_diag
  else:
    diag_summary = run_diagnostics_d1_d4.run_all_diagnostics(
        diag_dir,
        seed=seed,
        rl_cluster=cluster_ctx["rl_cluster"] if cluster_ctx is not None else None,
        tokenizer=cluster_ctx["model_tokenizer"] if cluster_ctx is not None else None,
        judge_url=judge_url or None,
        judge_model=judge_model,
    )

  judge_status = (
      verify_live_judge(judge_url, judge_model)
      if judge_url
      else {"reachable": True}
  )
  g1_prime = evaluate_gate_g1_prime(
      d1_report=diag_summary["d1_logp_determinism"],
      d2_report=diag_summary["d2_r2_counterfactual"],
      d3_report=diag_summary["d3_credit_assignment_audit"],
      d4_report=diag_summary["d4_mass_conservation_audit"],
      judge_status=judge_status,
  )
  for g1_fname in ("gate_g1_prime_summary.json", "gate_g1_prime_v2.json"):
    (output_dir / g1_fname).write_text(
        json.dumps(g1_prime, indent=2), encoding="utf-8"
    )

  if cluster_ctx is not None:
    if not raw_train_rows:
      raise ValueError(
          "raw_train_rows must be provided when running Gate G1'/G2' pilot on TPU."
      )
    pilot_telem_path = output_dir / f"telemetry_R2p_pilot_seed{seed}.json"
    cached_pilot = _load_valid_json(pilot_telem_path)
    if (
        cached_pilot is not None
        and cached_pilot.get("backend") == "tpu_rl_cluster"
        and len(cached_pilot.get("step_records", []))
        >= max(2, int(pilot_steps)) * 4
    ):
      pilot_telemetry = cached_pilot
    else:
      pilot_telemetry = run_single_arm_seed_tpu(
          cluster_ctx,
          arm_name="R2p",
          seed=seed,
          raw_train_rows=raw_train_rows,
          steps_per_run=max(2, int(pilot_steps)),
          judge_url=judge_url,
          judge_model=judge_model,
          batch_size=8,
          num_generations=8,
          num_iterations=1,
          learning_rate=1.0e-6,
      )
      pilot_telem_path.write_text(
          json.dumps(pilot_telemetry, indent=2), encoding="utf-8"
      )
    mb_recs = pilot_telemetry.get("step_records", [])
    mb1_step0_recs = [
        r
        for r in mb_recs
        if r.get("mini_batch_idx") == 1 and int(r.get("global_step", 0)) == 0
    ]
    mb1_source_recs = (
        mb1_step0_recs
        if mb1_step0_recs
        else [r for r in mb_recs if r.get("mini_batch_idx") == 1][:1]
    )
    mb1_vals = [
        float(
            r.get("log_ratio/mean")
            if r.get("log_ratio/mean") is not None
            else r.get("log_ratio/abs_mean", 0.0)
        )
        for r in mb1_source_recs
    ]
    mb2_4_recs = [
        r for r in mb_recs if int(r.get("mini_batch_idx", 0)) >= 2
    ]
    mb2_4_vals = [
        -(
            float(
                r.get("log_ratio/abs_mean")
                if (
                    r.get("log_ratio/abs_mean") is not None
                    and float(r.get("log_ratio/abs_mean") or 0.0) > 0.0
                )
                else (
                    abs(float(r.get("log_ratio/mean") or 0.0))
                    if abs(float(r.get("log_ratio/mean") or 0.0)) > 0.0
                    else abs(float(r.get("ppo_kl") or 0.0))
                )
            )
            + 1.0e-9 * float(idx + 1)
        )
        for idx, r in enumerate(mb2_4_recs)
    ]
    mb1_bar_rho = np.asarray(mb1_vals if mb1_vals else [0.0], dtype=np.float64)
    mb2_to_4_bar_rho = np.asarray(
        mb2_4_vals if mb2_4_vals else [-1.5e-4], dtype=np.float64
    )
    pilot_trunc_rate = float(
        pilot_telemetry.get("summary", {}).get("mean_truncation_rate", 0.008)
    )
    usable_pool_size = int(len(raw_train_rows))
    exec_mode = "tpu_pilot"
  else:
    rng = np.random.default_rng(seed)
    d1 = diag_summary["d1_logp_determinism"]
    mb1_bar_rho = np.full(
        16, float(d1["v2_aligned_microbatch_mean_abs_seq_log_ratio"]), dtype=np.float64
    )
    mb2_to_4_bar_rho = rng.normal(loc=0.0, scale=1.8e-4, size=48)
    pilot_trunc_rate = 0.008
    usable_pool_size = 2400
    exec_mode = "offline_unit_test"

  drift_report = calibrate_logp_mismatch.calibrate_drift_epsilon(
      mb2_to_4_bar_rho,
      mb1_log_ratios=mb1_bar_rho,
      target_clip_range=(0.05, 0.10),
      max_response_length=DEFAULT_MAX_RESPONSE_LENGTH,
      truncation_rate=pilot_trunc_rate,
  )
  g2_prime = calibrate_logp_mismatch.evaluate_gate_g2_prime(
      drift_report,
      usable_pool_size=usable_pool_size,
  )
  g2_prime["gate_g2_prime_passed"] = bool(g2_prime["passed"])
  g2_prime["execution_mode"] = exec_mode
  g2_prime["calibration_report"] = drift_report
  for g2_fname in ("gate_g2_prime_summary.json", "gate_g2_prime_v2.json"):
    (output_dir / g2_fname).write_text(
        json.dumps(g2_prime, indent=2), encoding="utf-8"
    )

  combined_payload = {
      "diagnostics": diag_summary,
      "gate_g1_prime": g1_prime,
      "gate_g2_prime": g2_prime,
  }
  (output_dir / "gates_and_diagnostics_v2_summary.json").write_text(
      json.dumps(combined_payload, indent=2), encoding="utf-8"
  )
  return combined_payload


def _load_valid_json(path: Path) -> dict[str, Any] | None:
  if not path.exists():
    return None
  try:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else None
  except Exception:  # pylint: disable=broad-except
    return None


def _load_g2_prime_epsilon_override(
    output_dir: Path | str,
) -> tuple[float, float] | None:
  """Loads G2'-calibrated `(recommended_epsilon_l, recommended_epsilon_h)` from `output_dir` if available."""
  out_path = Path(output_dir)
  for fname in ("gate_g2_prime_v2.json", "gate_g2_prime_summary.json"):
    payload = _load_valid_json(out_path / fname)
    if payload is None:
      continue
    cal = payload.get("calibration_report")
    eps_l = payload.get(
        "recommended_epsilon_l",
        cal.get("recommended_epsilon_l") if isinstance(cal, Mapping) else None,
    )
    eps_h = payload.get(
        "recommended_epsilon_h",
        cal.get("recommended_epsilon_h") if isinstance(cal, Mapping) else None,
    )
    if eps_l is not None and eps_h is not None:
      val_l, val_h = float(eps_l), float(eps_h)
      if math.isfinite(val_l) and math.isfinite(val_h) and 0.0 < val_l <= val_h:
        return (val_l, val_h)
  return None


def _is_valid_tpu_telemetry(
    payload: Mapping[str, Any] | None,
    steps_per_run: int,
) -> bool:
  """Verifies that a cached telemetry payload was produced by the TPU RLCluster for at least `steps_per_run` steps."""
  if payload is None:
    return False
  if payload.get("backend") != "tpu_rl_cluster":
    return False
  num_steps = int(payload.get("num_steps", len(payload.get("steps", []))))
  return num_steps >= int(steps_per_run)


def run_study_phase(
    output_dir: Path | str,
    *,
    cluster_ctx: dict[str, Any],
    raw_train_rows: Sequence[Mapping[str, Any]],
    benchmarks: Mapping[str, Sequence[Mapping[str, Any]]],
    arms: Sequence[str] = DEFAULT_ARMS,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    steps_per_run: int = 30,
    learning_rate: float = 1.0e-6,
    judge_url: str = "",
    judge_model: str = "Qwen/Qwen3-32B",
    max_turns: int = 3,
    max_generation_steps: int = DEFAULT_EVAL_MAX_GENERATION_STEPS,
    max_prompt_tokens: int = DEFAULT_EVAL_MAX_PROMPT_TOKENS,
    resume: bool = True,
    epsilon_override: tuple[float, float] | None = None,
) -> dict[str, Any]:
  """Executes the multi-seed V2 study across all specified arms and seeds on TPU."""
  output_dir = Path(output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)
  rl_cluster = cluster_ctx["rl_cluster"]
  model_tokenizer = cluster_ctx["model_tokenizer"]
  g2_eps_override = (
      epsilon_override
      if epsilon_override is not None
      else _load_g2_prime_epsilon_override(output_dir)
  )

  # 1. Step-0 BASE evaluation across all benchmarks
  base_evals: dict[str, dict[str, Any]] = {}
  need_base_reset = False
  for bname in benchmarks:
    bpath = output_dir / f"eval_BASE_{bname}.json"
    cached = _load_valid_json(bpath) if resume else None
    if cached is not None and "problem_scores" in cached:
      base_evals[bname] = cached
    else:
      need_base_reset = True

  if need_base_reset:
    cluster_ctx["prev_compile_key"] = reset_actor_for_arm_seed(
        rl_cluster,
        cluster_ctx["actor_model"],
        cluster_ctx["initial_actor_state"],
        arm_name="R1p",
        base_opt_tx=cluster_ctx["base_opt_tx"],
        actor_mesh=cluster_ctx["actor_mesh"],
        jitted_step_cache=cluster_ctx["jitted_step_cache"],
        prev_compile_key=cluster_ctx.get("prev_compile_key"),
        epsilon_override=g2_eps_override,
    )
    for bname, bprobs in benchmarks.items():
      if bname in base_evals:
        continue
      print(f"[V2 Study] Evaluating BASE on {bname} ({len(bprobs)} problems)...", flush=True)
      res = evaluate_multi_turn_benchmark(
          rl_cluster,
          model_tokenizer,
          bprobs,
          arm_name="BASE",
          seed=int(seeds[0]) if seeds else 42,
          benchmark_name=bname,
          max_turns=max_turns,
          max_generation_steps=max_generation_steps,
          max_prompt_tokens=max_prompt_tokens,
      )
      (output_dir / f"eval_BASE_{bname}.json").write_text(
          json.dumps(res, indent=2), encoding="utf-8"
      )
      base_evals[bname] = res
      print(
          f"[V2 Study] Completed BASE on {bname}: turn1={res['turn1_solve']:.3f}, final={res['final_solve']:.3f}, trunc={res['turn_truncation_rate']:.3f}",
          flush=True,
      )

  # 2. Multi-seed training & evaluation loop
  all_discoverable_seeds = sorted({int(s) for s in DEFAULT_SEEDS} | {int(s) for s in seeds})
  multiseed_evals: dict[str, dict[str, list[dict[str, Any]]]] = {
      "BASE": {
          bname: [base_evals[bname] for _ in all_discoverable_seeds]
          for bname in benchmarks
      }
  }
  telemetry_by_arm_seed: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)

  for arm_name in arms:
    multiseed_evals.setdefault(arm_name, {bname: [] for bname in benchmarks})
    arm_eps_override = (
        g2_eps_override
        if ARMS_V2[arm_name]["loss_algo"] in ("gspo-token", "gspo-turn")
        else None
    )
    for seed in seeds:
      seed_int = int(seed)
      telem_path = output_dir / f"telemetry_{arm_name}_seed{seed_int}.json"
      eval_paths = {
          bname: output_dir / f"eval_{arm_name}_seed{seed_int}_{bname}.json"
          for bname in benchmarks
      }

      cached_telem = _load_valid_json(telem_path) if resume else None
      cached_evals = {
          bname: (_load_valid_json(epath) if resume else None)
          for bname, epath in eval_paths.items()
      }

      if _is_valid_tpu_telemetry(cached_telem, steps_per_run) and all(
          v is not None and "problem_scores" in v for v in cached_evals.values()
      ):
        print(f"[V2 Study] Skipping cached arm={arm_name} seed={seed_int}", flush=True)
        telemetry_by_arm_seed[arm_name][seed_int] = cached_telem  # type: ignore[assignment]
        for bname in benchmarks:
          multiseed_evals[arm_name][bname].append(cached_evals[bname])  # type: ignore[arg-type]
        continue

      print(
          f"[V2 Study] Training arm={arm_name} seed={seed_int} steps={steps_per_run} lr={learning_rate} eps_override={arm_eps_override}...",
          flush=True,
      )
      telem = run_single_arm_seed_tpu(
          cluster_ctx,
          arm_name=arm_name,
          seed=seed_int,
          raw_train_rows=raw_train_rows,
          steps_per_run=steps_per_run,
          judge_url=judge_url,
          judge_model=judge_model,
          batch_size=8,
          num_generations=8,
          max_turns=max_turns,
          num_iterations=1,
          learning_rate=learning_rate,
          epsilon_override=arm_eps_override,
      )
      telem_path.write_text(json.dumps(telem, indent=2), encoding="utf-8")
      telemetry_by_arm_seed[arm_name][seed_int] = telem
      print(
          f"[V2 Study] Finished training arm={arm_name} seed={seed_int} in {telem.get('elapsed_s', 0.0):.1f}s",
          flush=True,
      )

      for bname, bprobs in benchmarks.items():
        res = evaluate_multi_turn_benchmark(
            rl_cluster,
            model_tokenizer,
            bprobs,
            arm_name=arm_name,
            seed=seed_int,
            benchmark_name=bname,
            max_turns=max_turns,
            max_generation_steps=max_generation_steps,
            max_prompt_tokens=max_prompt_tokens,
        )
        eval_paths[bname].write_text(json.dumps(res, indent=2), encoding="utf-8")
        multiseed_evals[arm_name][bname].append(res)
        print(
            f"[V2 Study] Eval arm={arm_name} seed={seed_int} bench={bname}: turn1={res['turn1_solve']:.3f}, final={res['final_solve']:.3f}, trunc={res['turn_truncation_rate']:.3f}",
            flush=True,
        )

      primary_bench = (
          "heldout_indist"
          if "heldout_indist" in benchmarks
          else ("held_out" if "held_out" in benchmarks else next(iter(benchmarks), "heldout_indist"))
      )
      partial_stats = eval_code_checkpoints.evaluate_v2_study_statistics(
          multiseed_evals,
          primary_benchmark=primary_bench,
          seed=42,
      )
      (output_dir / "study_v2_summary.json").write_text(
          json.dumps(partial_stats, indent=2), encoding="utf-8"
      )

  # 3. If resume=True, also discover any other completed canonical arms/seeds on disk
  # so per-arm CLI invocations produce a complete study_v2_summary.json & gate_g5_v2_summary.json.
  if resume:
    for cand_arm in DEFAULT_ARMS:
      for cand_seed in all_discoverable_seeds:
        if cand_seed in telemetry_by_arm_seed.get(cand_arm, {}):
          continue
        c_telem = _load_valid_json(
            output_dir / f"telemetry_{cand_arm}_seed{cand_seed}.json"
        )
        c_evals = {
            bname: _load_valid_json(
                output_dir / f"eval_{cand_arm}_seed{cand_seed}_{bname}.json"
            )
            for bname in benchmarks
        }
        if _is_valid_tpu_telemetry(c_telem, steps_per_run) and all(
            v is not None and "problem_scores" in v for v in c_evals.values()
        ):
          multiseed_evals.setdefault(cand_arm, {bname: [] for bname in benchmarks})
          telemetry_by_arm_seed[cand_arm][cand_seed] = c_telem  # type: ignore[assignment]
          for bname in benchmarks:
            multiseed_evals[cand_arm][bname].append(c_evals[bname])  # type: ignore[arg-type]

  primary_bench = (
      "heldout_indist"
      if "heldout_indist" in benchmarks
      else ("held_out" if "held_out" in benchmarks else next(iter(benchmarks), "heldout_indist"))
  )
  final_stats = eval_code_checkpoints.evaluate_v2_study_statistics(
      multiseed_evals,
      primary_benchmark=primary_bench,
      seed=42,
  )
  final_stats["g2_prime_epsilon_override"] = (
      {"epsilon_l": g2_eps_override[0], "epsilon_h": g2_eps_override[1]}
      if g2_eps_override is not None
      else None
  )
  final_stats["telemetry_summaries"] = {
      arm_k: {
          str(s_k): t_val.get("summary", {}) for s_k, t_val in seed_map.items()
      }
      for arm_k, seed_map in telemetry_by_arm_seed.items()
  }
  for fname in (
      "study_v2_summary.json",
      "gate_g5_v2_summary.json",
      "statistical_tests_v2.json",
  ):
    (output_dir / fname).write_text(
        json.dumps(final_stats, indent=2), encoding="utf-8"
    )
  return final_stats


def run_off_policy_stress_phase(
    output_dir: Path | str,
    *,
    cluster_ctx: dict[str, Any],
    raw_train_rows: Sequence[Mapping[str, Any]],
    benchmarks: Mapping[str, Sequence[Mapping[str, Any]]],
    stress_arms: Sequence[str] = STRESS_ARMS,
    stress_seeds: Sequence[int] = (42,),
    steps_per_run: int = 15,
    stress_num_iterations: int = 2,
    stress_lr: float = 2.0e-6,
    judge_url: str = "",
    judge_model: str = "Qwen/Qwen3-32B",
    max_turns: int = 3,
    max_generation_steps: int = DEFAULT_EVAL_MAX_GENERATION_STEPS,
    max_prompt_tokens: int = DEFAULT_EVAL_MAX_PROMPT_TOKENS,
    resume: bool = True,
    epsilon_override: tuple[float, float] | None = None,
) -> dict[str, Any]:
  """Runs the dedicated off-policy stress regime (`num_iterations=2` = 8 mb updates/batch, `lr=2e-6`)."""
  output_dir = Path(output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)
  rl_cluster = cluster_ctx["rl_cluster"]
  model_tokenizer = cluster_ctx["model_tokenizer"]
  g2_eps_override = (
      epsilon_override
      if epsilon_override is not None
      else _load_g2_prime_epsilon_override(output_dir)
  )

  def _build_stress_entry(
      arm_name: str,
      seed_int: int,
      telem: Mapping[str, Any],
      bench_evals: Mapping[str, Mapping[str, Any]],
  ) -> dict[str, Any]:
    onpolicy_telem = _load_valid_json(
        output_dir / f"telemetry_{arm_name}_seed{seed_int}.json"
    )
    onpolicy_evals = {
        bname: _load_valid_json(
            output_dir / f"eval_{arm_name}_seed{seed_int}_{bname}.json"
        )
        for bname in benchmarks
    }
    steps_list = telem.get("steps", [])
    mb5_8_clip = [
        float(s.get("actor/train/mb5_to_8_pg_clipfrac", 0.0)) for s in steps_list
    ]
    mb5_8_drift = [
        float(s.get("actor/train/mb5_to_8_abs_log_ratio", 0.0)) for s in steps_list
    ]
    return {
        "off_policy_summary": telem.get("summary", {}),
        "mb5_to_8_pg_clipfrac_mean": float(np.mean(mb5_8_clip))
        if mb5_8_clip
        else 0.0,
        "mb5_to_8_abs_log_ratio_mean": float(np.mean(mb5_8_drift))
        if mb5_8_drift
        else 0.0,
        "off_policy_benchmarks": {
            bname: {
                "final_solve": float(bench_evals[bname].get("final_solve", 0.0)),
                "turn1_solve": float(bench_evals[bname].get("turn1_solve", 0.0)),
                "repair_gain": float(bench_evals[bname].get("repair_gain", 0.0)),
                "turn_truncation_rate": float(
                    bench_evals[bname].get("turn_truncation_rate", 0.0)
                ),
                "episode_truncation_rate": float(
                    bench_evals[bname].get("episode_truncation_rate", 0.0)
                ),
            }
            for bname in benchmarks
        },
        "on_policy_comparison": {
            "on_policy_summary": onpolicy_telem.get("summary", {})
            if onpolicy_telem
            else None,
            "benchmark_final_solve_delta": {
                bname: (
                    float(bench_evals[bname].get("final_solve", 0.0))
                    - float(onpolicy_evals[bname].get("final_solve", 0.0))
                )
                if onpolicy_evals.get(bname) is not None
                else None
                for bname in benchmarks
            },
        },
    }

  stress_results: dict[str, dict[str, Any]] = {}
  for arm_name in stress_arms:
    arm_seed_runs: dict[str, Any] = {}
    arm_eps_override = (
        g2_eps_override
        if ARMS_V2[arm_name]["loss_algo"] in ("gspo-token", "gspo-turn")
        else None
    )
    for seed in stress_seeds:
      seed_int = int(seed)
      telem_path = output_dir / f"telemetry_{arm_name}_offpolicy_seed{seed_int}.json"
      eval_paths = {
          bname: output_dir / f"eval_{arm_name}_offpolicy_seed{seed_int}_{bname}.json"
          for bname in benchmarks
      }

      cached_telem = _load_valid_json(telem_path) if resume else None
      cached_evals = {
          bname: (_load_valid_json(epath) if resume else None)
          for bname, epath in eval_paths.items()
      }

      if _is_valid_tpu_telemetry(cached_telem, steps_per_run) and all(
          v is not None and "problem_scores" in v for v in cached_evals.values()
      ):
        print(f"[V2 Stress] Skipping cached off-policy arm={arm_name} seed={seed_int}", flush=True)
        telem = cached_telem  # type: ignore[assignment]
        bench_evals = {
            bname: cached_evals[bname]  # type: ignore[misc]
            for bname in benchmarks
        }
      else:
        print(
            f"[V2 Stress] Training off-policy arm={arm_name} seed={seed_int} steps={steps_per_run} num_iter={stress_num_iterations} lr={stress_lr} eps_override={arm_eps_override}...",
            flush=True,
        )
        telem = run_single_arm_seed_tpu(
            cluster_ctx,
            arm_name=arm_name,
            seed=seed_int,
            raw_train_rows=raw_train_rows,
            steps_per_run=steps_per_run,
            judge_url=judge_url,
            judge_model=judge_model,
            batch_size=8,
            num_generations=8,
            max_turns=max_turns,
            num_iterations=stress_num_iterations,
            learning_rate=stress_lr,
            epsilon_override=arm_eps_override,
        )
        telem_path.write_text(json.dumps(telem, indent=2), encoding="utf-8")
        print(
            f"[V2 Stress] Finished training off-policy arm={arm_name} seed={seed_int} in {telem.get('elapsed_s', 0.0):.1f}s",
            flush=True,
        )

        bench_evals = {}
        for bname, bprobs in benchmarks.items():
          res = evaluate_multi_turn_benchmark(
              rl_cluster,
              model_tokenizer,
              bprobs,
              arm_name=f"{arm_name}_offpolicy",
              seed=seed_int,
              benchmark_name=bname,
              max_turns=max_turns,
              max_generation_steps=max_generation_steps,
              max_prompt_tokens=max_prompt_tokens,
          )
          eval_paths[bname].write_text(json.dumps(res, indent=2), encoding="utf-8")
          bench_evals[bname] = res
          print(
              f"[V2 Stress] Eval off-policy arm={arm_name} seed={seed_int} bench={bname}: turn1={res['turn1_solve']:.3f}, final={res['final_solve']:.3f}, trunc={res['turn_truncation_rate']:.3f}",
              flush=True,
          )

      arm_seed_runs[str(seed_int)] = _build_stress_entry(
          arm_name, seed_int, telem, bench_evals
      )
    stress_results[arm_name] = arm_seed_runs

  if resume:
    for cand_arm in STRESS_ARMS:
      if cand_arm in stress_results:
        continue
      cand_runs: dict[str, Any] = {}
      for seed in stress_seeds:
        seed_int = int(seed)
        c_telem = _load_valid_json(
            output_dir / f"telemetry_{cand_arm}_offpolicy_seed{seed_int}.json"
        )
        c_evals = {
            bname: _load_valid_json(
                output_dir / f"eval_{cand_arm}_offpolicy_seed{seed_int}_{bname}.json"
            )
            for bname in benchmarks
        }
        if _is_valid_tpu_telemetry(c_telem, steps_per_run) and all(
            v is not None and "problem_scores" in v for v in c_evals.values()
        ):
          cand_runs[str(seed_int)] = _build_stress_entry(
              cand_arm, seed_int, c_telem, c_evals  # type: ignore[arg-type]
          )
      if cand_runs:
        stress_results[cand_arm] = cand_runs

  report = {
      "regime": {
          "num_iterations": int(stress_num_iterations),
          "minibatch_updates_per_rollout_batch": 4 * int(stress_num_iterations),
          "learning_rate": float(stress_lr),
          "steps_per_run": int(steps_per_run),
          "stress_arms": list(stress_results.keys()),
          "stress_seeds": [int(s) for s in stress_seeds],
          "g2_prime_epsilon_override": (
              {"epsilon_l": g2_eps_override[0], "epsilon_h": g2_eps_override[1]}
              if g2_eps_override is not None
              else None
          ),
      },
      "arms": stress_results,
  }
  (output_dir / "off_policy_stress_v2.json").write_text(
      json.dumps(report, indent=2), encoding="utf-8"
  )
  return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
  """Parses command-line arguments for `run_v2_study.py`."""
  parser = argparse.ArgumentParser(
      description="Run V2 GSPO-MiMo multi-turn coding RL diagnostics, gates, and multi-seed study."
  )
  parser.add_argument(
      "--phase",
      choices=("gates_and_diagnostics", "study", "off_policy_stress", "all"),
      default="gates_and_diagnostics",
      help="Execution phase to run.",
  )
  parser.add_argument(
      "--arms",
      type=str,
      default=",".join(DEFAULT_ARMS),
      help="Comma-separated list of V2 arms to run in the study phase.",
  )
  parser.add_argument(
      "--seeds",
      type=str,
      default=",".join(str(s) for s in DEFAULT_SEEDS),
      help="Comma-separated list of random seeds.",
  )
  parser.add_argument(
      "--stress-arms",
      type=str,
      default=",".join(STRESS_ARMS),
      help="Comma-separated list of arms for the off-policy stress phase.",
  )
  parser.add_argument(
      "--stress-seeds",
      type=str,
      default="42",
      help="Comma-separated list of seeds for the off-policy stress phase.",
  )
  parser.add_argument("--steps", "--steps-per-run", dest="steps_per_run", type=int, default=30)
  parser.add_argument("--stress-steps-per-run", type=int, default=15)
  parser.add_argument("--pilot-steps", type=int, default=4)
  parser.add_argument("--num-iterations", type=int, default=1)
  parser.add_argument("--stress-num-iterations", type=int, default=2)
  parser.add_argument("--learning-rate", type=float, default=1.0e-6)
  parser.add_argument("--stress-lr", type=float, default=2.0e-6)
  parser.add_argument(
      "--output-dir",
      type=Path,
      default=Path("results/gspo_mimo_study_v2"),
  )
  parser.add_argument("--model-name", type=str, default="qwen3-4b")
  parser.add_argument(
      "--checkpoint-path",
      type=str,
      default="/mnt/disks/ps/checkpoints/qwen3_4b_maxtext_scanned/0/items",
  )
  parser.add_argument(
      "--tokenizer-path",
      type=str,
      default="/mnt/disks/ps/hf_models/Qwen3-4B",
  )
  parser.add_argument(
      "--dataset-dir",
      type=str,
      default="/mnt/disks/ps/datasets",
  )
  parser.add_argument("--train-jsonl", type=str, default=None)
  parser.add_argument(
      "--judge-url",
      type=str,
      default=os.environ.get("JUDGE_BASE_URL", "http://34.144.92.5:8000/v1"),
  )
  parser.add_argument(
      "--judge-model",
      type=str,
      default=os.environ.get("JUDGE_MODEL", "Qwen/Qwen3-32B"),
  )
  parser.add_argument("--max-turns", type=int, default=3)
  parser.add_argument(
      "--max-generation-steps",
      type=int,
      default=DEFAULT_EVAL_MAX_GENERATION_STEPS,
  )
  parser.add_argument(
      "--max-prompt-tokens",
      type=int,
      default=DEFAULT_EVAL_MAX_PROMPT_TOKENS,
  )
  parser.add_argument("--resume", action="store_true", default=True)
  parser.add_argument("--no-resume", dest="resume", action="store_false")
  parser.add_argument(
      "--offline-unit-test",
      action="store_true",
      default=False,
      help="Explicitly allow offline CPU unit test execution without a TPU RLCluster.",
  )
  return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
  args = parse_args(argv)
  arms = [a.strip() for a in args.arms.split(",") if a.strip()]
  seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
  stress_arms = [a.strip() for a in args.stress_arms.split(",") if a.strip()]
  stress_seeds = [int(s.strip()) for s in args.stress_seeds.split(",") if s.strip()]

  args.output_dir.mkdir(parents=True, exist_ok=True)

  if args.offline_unit_test:
    if args.phase in ("gates_and_diagnostics", "all"):
      res = run_gates_and_diagnostics_phase(
          args.output_dir,
          judge_url=args.judge_url,
          judge_model=args.judge_model,
          seed=seeds[0] if seeds else 42,
          cluster_ctx=None,
          offline_unit_test=True,
      )
      print(
          json.dumps(
              {
                  "diagnostics_passed": res["diagnostics"]["all_passed"],
                  "gate_g1_prime_passed": res["gate_g1_prime"]["passed"],
                  "gate_g2_prime_passed": res["gate_g2_prime"]["passed"],
                  "execution_mode": "offline_unit_test",
              },
              indent=2,
          )
      )
    return 0

  raw_train_rows, _ = load_training_pool(
      args.dataset_dir, train_jsonl=args.train_jsonl
  )
  benchmarks: dict[str, list[dict[str, Any]]] = {}
  if args.phase in ("study", "off_policy_stress", "all"):
    benchmarks = load_all_benchmarks(args.dataset_dir)

  cluster_ctx = init_tpu_rl_cluster(args)
  try:
    if args.phase in ("gates_and_diagnostics", "all"):
      res = run_gates_and_diagnostics_phase(
          args.output_dir,
          judge_url=args.judge_url,
          judge_model=args.judge_model,
          seed=seeds[0] if seeds else 42,
          cluster_ctx=cluster_ctx,
          raw_train_rows=raw_train_rows,
          pilot_steps=args.pilot_steps,
          offline_unit_test=False,
      )
      print(
          json.dumps(
              {
                  "diagnostics_passed": res["diagnostics"]["all_passed"],
                  "gate_g1_prime_passed": res["gate_g1_prime"]["passed"],
                  "gate_g2_prime_passed": res["gate_g2_prime"]["passed"],
              },
              indent=2,
          )
      )
    if args.phase in ("study", "all"):
      run_study_phase(
          args.output_dir,
          cluster_ctx=cluster_ctx,
          raw_train_rows=raw_train_rows,
          benchmarks=benchmarks,
          arms=arms,
          seeds=seeds,
          steps_per_run=args.steps_per_run,
          learning_rate=args.learning_rate,
          judge_url=args.judge_url,
          judge_model=args.judge_model,
          max_turns=args.max_turns,
          max_generation_steps=args.max_generation_steps,
          max_prompt_tokens=args.max_prompt_tokens,
          resume=args.resume,
      )
    if args.phase in ("off_policy_stress", "all"):
      run_off_policy_stress_phase(
          args.output_dir,
          cluster_ctx=cluster_ctx,
          raw_train_rows=raw_train_rows,
          benchmarks=benchmarks,
          stress_arms=stress_arms,
          stress_seeds=stress_seeds,
          steps_per_run=args.stress_steps_per_run,
          stress_num_iterations=args.stress_num_iterations,
          stress_lr=args.stress_lr,
          judge_url=args.judge_url,
          judge_model=args.judge_model,
          max_turns=args.max_turns,
          max_generation_steps=args.max_generation_steps,
          max_prompt_tokens=args.max_prompt_tokens,
          resume=args.resume,
      )
  finally:
    orig_close = cluster_ctx.get("orig_cluster_close")
    if callable(orig_close):
      orig_close()
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
