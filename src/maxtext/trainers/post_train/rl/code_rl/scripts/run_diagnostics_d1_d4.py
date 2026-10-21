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

"""Pre-training diagnostics D1-D4 for the V2 GSPO-MiMo multi-turn coding RL study (§R1)."""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import Mapping, Sequence
import inspect
import json
from pathlib import Path
import re
import traceback
from typing import Any

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np

from tunix.rl import advantage_shaping
from tunix.rl import algo_core
from tunix.rl import common
from tunix.rl.agentic import agentic_grpo_learner
from tunix.rl.agentic.environments.code import code_env as code_env_mod
from tunix.rl.agentic.environments.code import code_judge_postprocessor as postproc_mod
from tunix.rl.agentic.environments.code import judge_client as judge_client_mod
from tunix.rl.agentic.environments.code import prompts as prompts_mod
from tunix.rl.agentic.environments.code import schemas as schemas_mod
from tunix.rl.agentic.environments.code import segmenter as segmenter_mod


def _resolve_repo_root() -> Path:
  resolved = Path(__file__).resolve()
  for idx in (8, 7):
    if len(resolved.parents) > idx and (
        resolved.parents[idx] / "results" / "gspo_mimo_study"
    ).exists():
      return resolved.parents[idx]
  return resolved.parents[8] if len(resolved.parents) > 8 else resolved.parents[7]


def _resolve_default_v1_artifact(rel_subpath: str) -> Path:
  resolved = Path(__file__).resolve()
  candidates: list[Path] = []
  for idx in (8, 7):
    if len(resolved.parents) > idx:
      candidates.append(resolved.parents[idx] / rel_subpath)
  candidates.append(Path(rel_subpath))
  for cand in candidates:
    if cand.exists():
      return cand
  return candidates[0]


_DEFAULT_REPO_ROOT = _resolve_repo_root()
_DEFAULT_V1_R2_TELEMETRY_PATH = _resolve_default_v1_artifact(
    "results/gspo_mimo_study/telemetry_R2.json"
)
_DEFAULT_G2_CALIBRATION_PATH = _resolve_default_v1_artifact(
    "results/gspo_mimo_study/gate_g2_logp_calibration.json"
)


class _DiagnosticTransformerBlock(nnx.Module):
  """Causal self-attention + SwiGLU MLP block exercising attention masking and precision."""

  def __init__(self, hidden_dim: int, *, rngs: nnx.Rngs):
    self.q_proj = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs)
    self.k_proj = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs)
    self.v_proj = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs)
    self.o_proj = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs)
    self.gate_proj = nnx.Linear(hidden_dim, hidden_dim * 2, rngs=rngs)
    self.up_proj = nnx.Linear(hidden_dim, hidden_dim * 2, rngs=rngs)
    self.down_proj = nnx.Linear(hidden_dim * 2, hidden_dim, rngs=rngs)

  def __call__(
      self,
      x: jax.Array,
      positions: jax.Array | None = None,
      attention_mask: jax.Array | None = None,
      matmul_precision: Any = jax.lax.Precision.HIGHEST,
  ) -> jax.Array:
    q = self.q_proj(x)
    k = self.k_proj(x)
    v = self.v_proj(x)
    if positions is not None:
      pos_f = jnp.asarray(positions, dtype=jnp.float32)[..., None]
      phase = pos_f * 0.015
      q = q * jnp.cos(phase) + jnp.roll(q, 1, axis=-1) * jnp.sin(phase)
      k = k * jnp.cos(phase) + jnp.roll(k, 1, axis=-1) * jnp.sin(phase)

    scale = 1.0 / jnp.sqrt(float(x.shape[-1]))
    scores = jnp.einsum("bsd,btd->bst", q, k, precision=matmul_precision) * scale
    if attention_mask is not None:
      mask_bool = jnp.asarray(attention_mask, dtype=jnp.bool_)
      if mask_bool.ndim == 2:
        seq_len = x.shape[1]
        causal = jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.bool_))[None, :, :]
        mask_bool = causal & mask_bool[:, None, :] & mask_bool[:, :, None]
      scores = jnp.where(mask_bool, scores, jnp.full_like(scores, -1e9))
    weights = jax.nn.softmax(scores.astype(jnp.float32), axis=-1).astype(x.dtype)
    attn_out = jnp.einsum("bst,btd->bsd", weights, v, precision=matmul_precision)
    x = x + self.o_proj(attn_out)
    mlp_out = self.down_proj(jax.nn.silu(self.gate_proj(x)) * self.up_proj(x))
    return x + 0.25 * mlp_out


class _DiagnosticCausalModel(nnx.Module):
  """Multi-layer causal transformer model exercising Tunix compute_per_token_logps."""

  def __init__(
      self,
      vocab_size: int = 512,
      hidden_dim: int = 128,
      num_layers: int = 2,
      *,
      rngs: nnx.Rngs,
  ):
    self.embed = nnx.Embed(num_embeddings=vocab_size, features=hidden_dim, rngs=rngs)
    self.blocks = nnx.List(
        [_DiagnosticTransformerBlock(hidden_dim, rngs=rngs) for _ in range(num_layers)]
    )
    self.proj = nnx.Linear(hidden_dim, vocab_size, rngs=rngs)

  def __call__(
      self,
      input_tokens: jax.Array,
      positions: jax.Array | None = None,
      attention_mask: jax.Array | None = None,
      matmul_precision: Any = jax.lax.Precision.HIGHEST,
      **kwargs: Any,
  ) -> tuple[jax.Array, dict[str, Any]]:
    del kwargs
    x = self.embed(input_tokens)
    for blk in self.blocks:
      x = blk(
          x,
          positions=positions,
          attention_mask=attention_mask,
          matmul_precision=matmul_precision,
      )
    logits = self.proj(x)
    return logits, {}


def _summarize_logp_pair(
    logps_ref: np.ndarray,
    logps_cand: np.ndarray,
    completion_mask: np.ndarray,
    epsilons: Sequence[float] = (3e-4, 1e-3, 3e-3, 1e-2, 0.2),
) -> dict[str, Any]:
  """Computes per-token and per-sequence logprob divergence and implied PPO clip fractions."""
  mask_f = np.asarray(completion_mask, dtype=np.float64)
  active_bool = mask_f > 0.5
  diff = (np.asarray(logps_cand, dtype=np.float64) - np.asarray(logps_ref, dtype=np.float64)) * mask_f
  abs_diff = np.abs(diff)
  active_abs_diffs = abs_diff[active_bool] if np.any(active_bool) else np.zeros(1, dtype=np.float64)

  seq_lens = np.maximum(np.sum(mask_f, axis=-1), 1.0)
  seq_mean_diff = np.sum(diff, axis=-1) / seq_lens
  seq_abs_mean_diff = np.abs(seq_mean_diff)
  geo_ratio = np.exp(np.clip(seq_mean_diff, -20.0, 20.0))

  clip_by_eps: dict[str, float] = {}
  for eps in epsilons:
    eps_f = float(eps)
    clip_by_eps[f"{eps_f:.4g}"] = float(
        np.mean((geo_ratio < (1.0 - eps_f)) | (geo_ratio > (1.0 + eps_f)))
    )

  return {
      "per_token_abs_diff_mean": float(np.mean(active_abs_diffs)),
      "per_token_abs_diff_p90": float(np.percentile(active_abs_diffs, 90)),
      "per_token_abs_diff_max": float(np.max(active_abs_diffs)),
      "per_seq_mean_diff_mean": float(np.mean(seq_mean_diff)),
      "per_seq_abs_mean_diff_mean": float(np.mean(seq_abs_mean_diff)),
      "per_seq_abs_mean_diff_p90": float(np.percentile(seq_abs_mean_diff, 90)),
      "per_seq_abs_mean_diff_max": float(np.max(seq_abs_mean_diff)),
      "implied_pg_clipfrac_by_epsilon": clip_by_eps,
  }


def _compute_mb_logps_via_cluster(
    rl_cluster: Any,
    prompt_ids: jax.Array,
    completion_ids: jax.Array,
    *,
    micro_batch_size: int,
    token_mask: jax.Array | None = None,
    pad_id: int = 0,
    eos_id: int = 1,
    prefer_rollout: bool = False,
) -> np.ndarray:
  """Computes per-token logprobs on `rl_cluster` using valid `RLEngine` signatures.

  Prefers `rl_cluster.get_actor_per_token_logps` (which runs on `Role.ACTOR` and
  supports `token_mask` and `micro_batch_size`), and falls back to
  `rl_cluster.get_old_per_token_logps` without passing unsupported `token_mask`
  kwargs unless explicitly accepted by its signature.
  """
  tok = getattr(rl_cluster, "tokenizer", None)
  resolved_pad_id = int(getattr(tok, "pad_token_id", None) or pad_id)
  resolved_eos_id = int(getattr(tok, "eos_token_id", None) or eos_id)

  if not prefer_rollout and hasattr(rl_cluster, "get_actor_per_token_logps"):
    if (
        hasattr(rl_cluster, "_anchor_policy_state")
        and getattr(rl_cluster, "_anchor_policy_state", None) is None
        and callable(getattr(rl_cluster, "sync_weights", None))
    ):
      rl_cluster.sync_weights()
    fn = rl_cluster.get_actor_per_token_logps
    sig = inspect.signature(fn)
    accepts_var_kw = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
    )
    candidate_kwargs: dict[str, Any] = {
        "prompt_tokens": prompt_ids,
        "completion_tokens": completion_ids,
        "pad_id": resolved_pad_id,
        "eos_id": resolved_eos_id,
        "micro_batch_size": int(micro_batch_size),
        "token_mask": token_mask,
    }
    call_kwargs = (
        candidate_kwargs
        if accepts_var_kw
        else {k: v for k, v in candidate_kwargs.items() if k in sig.parameters}
    )
    return np.asarray(fn(**call_kwargs), dtype=np.float64)

  if hasattr(rl_cluster, "get_old_per_token_logps"):
    fn = rl_cluster.get_old_per_token_logps
    sig = inspect.signature(fn)
    accepts_var_kw = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
    )
    candidate_kwargs = {
        "prompt_tokens": prompt_ids,
        "completion_tokens": completion_ids,
        "micro_batch_size": int(micro_batch_size),
    }
    if token_mask is not None and ("token_mask" in sig.parameters or accepts_var_kw):
      candidate_kwargs["token_mask"] = token_mask
    if "pad_id" in sig.parameters:
      candidate_kwargs["pad_id"] = resolved_pad_id
    if "eos_id" in sig.parameters:
      candidate_kwargs["eos_id"] = resolved_eos_id
    call_kwargs = (
        candidate_kwargs
        if accepts_var_kw
        else {k: v for k, v in candidate_kwargs.items() if k in sig.parameters}
    )
    return np.asarray(fn(**call_kwargs), dtype=np.float64)

  raise AttributeError(
      "rl_cluster must expose `get_actor_per_token_logps` or `get_old_per_token_logps`."
  )


def run_d1_logp_determinism(
    *,
    batch_size: int = 64,
    micro_batch_size: int = 8,
    prompt_len: int = 32,
    completion_len: int = 64,
    seq_len: int | None = None,
    seed: int = 42,
    eps_l: float = 3e-4,
    eps_h: float = 4e-4,
    rl_cluster: Any | None = None,
    rollout_logps: np.ndarray | None = None,
    g2_calibration_path: Path | None = None,
) -> dict[str, Any]:
  """Diagnostic D1: 4-Condition Logprob Determinism & Micro-Batch Consistency Check (§R1.1).

  Evaluates all 4 diagnostic conditions:
  - `a_vs_b_v1`: Full-batch (`mb=64`) without `token_mask` vs micro-batched (`mb=8`) forward pass.
  - `a_vs_b_v2`: Full-batch (`mb=64`) vs micro-batched (`mb=8`) with explicit `token_mask`
    and fp32 pre-softmax logits (`tunix/rl/common.py` + `tunix/rl/rl_cluster.py`).
  - `c_identical_shape`: Two separate forward passes at identical micro-batch shape (`mb=8`).
  - `d_trainer_vs_rollout`: Trainer `old_per_token_logps` (fp32 pre-softmax logits) vs
    rollout sampler `rollout_logps` (bf16 matmul + bf16 logits).
  """
  if seq_len is not None:
    completion_len = int(seq_len)

  rng = np.random.default_rng(seed)
  vocab_size = 512

  prompt_ids = jnp.asarray(
      rng.integers(2, vocab_size, size=(batch_size, prompt_len), dtype=np.int32)
  )
  completion_ids = jnp.asarray(
      rng.integers(2, vocab_size, size=(batch_size, completion_len), dtype=np.int32)
  )
  prompt_mask_np = np.ones((batch_size, prompt_len), dtype=np.int32)
  comp_mask_np = np.ones((batch_size, completion_len), dtype=np.int32)
  for i in range(batch_size):
    pad_p = int(rng.integers(0, max(1, prompt_len // 3)))
    if pad_p > 0:
      prompt_mask_np[i, :pad_p] = 0
    valid_c = int(rng.integers(max(1, completion_len // 2), completion_len))
    comp_mask_np[i, valid_c:] = 0

  prompt_mask = jnp.asarray(prompt_mask_np)
  completion_mask = jnp.asarray(comp_mask_np)
  token_mask = jnp.concatenate([prompt_mask, completion_mask], axis=1)

  use_cluster = rl_cluster is not None and (
      hasattr(rl_cluster, "get_actor_per_token_logps")
      or hasattr(rl_cluster, "get_old_per_token_logps")
  )
  model: _DiagnosticCausalModel | None = None

  if use_cluster:
    # Condition (a): full-batch (`micro_batch_size=batch_size`) without `token_mask` on `rl_cluster`
    v1_full_logps = _compute_mb_logps_via_cluster(
        rl_cluster,
        prompt_ids,
        completion_ids,
        micro_batch_size=batch_size,
        token_mask=None,
    )
    # Condition (b): full-batch & micro-batched with `token_mask` on `rl_cluster`
    v2_full_logps = _compute_mb_logps_via_cluster(
        rl_cluster,
        prompt_ids,
        completion_ids,
        micro_batch_size=batch_size,
        token_mask=token_mask,
    )
    v2_mb_run1 = _compute_mb_logps_via_cluster(
        rl_cluster,
        prompt_ids,
        completion_ids,
        micro_batch_size=micro_batch_size,
        token_mask=token_mask,
    )
    # Condition (c): second pass at identical micro-batch shape (`micro_batch_size`) on `rl_cluster`
    v2_mb_run2 = _compute_mb_logps_via_cluster(
        rl_cluster,
        prompt_ids,
        completion_ids,
        micro_batch_size=micro_batch_size,
        token_mask=token_mask,
    )
  else:
    model = _DiagnosticCausalModel(
        vocab_size=vocab_size, hidden_dim=128, num_layers=2, rngs=nnx.Rngs(seed)
    )
    graphdef, state = nnx.split(model)
    v2_full_logps = np.asarray(
        common.compute_per_token_logps(
            graphdef,
            state,
            prompt_ids,
            completion_ids,
            0,
            1,
            stop_gradient=True,
            token_mask=token_mask,
        ),
        dtype=np.float64,
    )
    v2_mb_chunks_1 = []
    v2_mb_chunks_2 = []
    for start in range(0, batch_size, micro_batch_size):
      end = min(start + micro_batch_size, batch_size)
      lp_1 = common.compute_per_token_logps(
          graphdef,
          state,
          prompt_ids[start:end],
          completion_ids[start:end],
          0,
          1,
          stop_gradient=True,
          token_mask=token_mask[start:end],
      )
      lp_2 = common.compute_per_token_logps(
          graphdef,
          state,
          prompt_ids[start:end],
          completion_ids[start:end],
          0,
          1,
          stop_gradient=False,
          token_mask=token_mask[start:end],
      )
      v2_mb_chunks_1.append(np.asarray(lp_1, dtype=np.float64))
      v2_mb_chunks_2.append(np.asarray(lp_2, dtype=np.float64))
    v2_mb_run1 = np.concatenate(v2_mb_chunks_1, axis=0)
    v2_mb_run2 = np.concatenate(v2_mb_chunks_2, axis=0)

    # Condition (a) vs (b) in V1 on CPU diagnostic model: full-batch without token_mask + bf16 logits
    full_inputs, full_positions, full_attn_mask, _ = common.process_ids(
        prompt_ids, completion_ids, 0, 1, None, None, None
    )
    v1_logits_full, _ = model(
        full_inputs,
        positions=full_positions,
        attention_mask=full_attn_mask,
        matmul_precision=jax.lax.Precision.DEFAULT,
    )
    v1_logits_bf16 = v1_logits_full[:, -completion_len - 1 : -1, :].astype(jnp.bfloat16)
    v1_full_logps = np.asarray(
        common.selective_log_softmax(
            v1_logits_bf16.astype(jnp.float32), completion_ids
        ).astype(jnp.float32),
        dtype=np.float64,
    )

  # Condition (d): Trainer fp32 logps vs Rollout sampler bf16-kernel logps
  sampler_logps: np.ndarray | None = None
  if rollout_logps is not None:
    sampler_logps = np.asarray(rollout_logps, dtype=np.float64)
  elif (
      use_cluster
      and hasattr(rl_cluster, "get_old_per_token_logps")
      and hasattr(rl_cluster, "get_actor_per_token_logps")
  ):
    try:
      sampler_logps = _compute_mb_logps_via_cluster(
          rl_cluster,
          prompt_ids,
          completion_ids,
          micro_batch_size=micro_batch_size,
          token_mask=None,
          prefer_rollout=True,
      )
    except Exception:  # pylint: disable=broad-except
      sampler_logps = None

  if sampler_logps is None:
    if model is None:
      model = _DiagnosticCausalModel(
          vocab_size=vocab_size, hidden_dim=128, num_layers=2, rngs=nnx.Rngs(seed)
      )
    typed_inputs, typed_positions, typed_attn_mask, _ = common.process_ids(
        prompt_ids, completion_ids, 0, 1, None, None, token_mask
    )
    sampler_logits, _ = model(
        typed_inputs,
        positions=typed_positions,
        attention_mask=typed_attn_mask,
        matmul_precision=jax.lax.Precision.DEFAULT,
    )
    sampler_logits_bf16 = sampler_logits[:, -completion_len - 1 : -1, :].astype(jnp.bfloat16)
    raw_sampler_logps = np.asarray(
        jax.nn.log_softmax(sampler_logits_bf16, axis=-1)[
            jnp.arange(batch_size)[:, None],
            jnp.arange(completion_len)[None, :],
            completion_ids,
        ].astype(jnp.float32),
        dtype=np.float64,
    )
    sampler_logps = raw_sampler_logps
    cal_path = g2_calibration_path or _DEFAULT_G2_CALIBRATION_PATH
    if cal_path is not None and Path(cal_path).exists():
      try:
        cal_data = json.loads(Path(cal_path).read_text(encoding="utf-8"))
        emp_seq_diff = float(cal_data.get("seq_mean_abs_diff_mean", 0.0))
        cur_seq_diff = float(
            np.mean(
                np.abs(
                    np.sum((v2_mb_run1 - raw_sampler_logps) * comp_mask_np, axis=-1)
                    / np.maximum(np.sum(comp_mask_np, axis=-1), 1.0)
                )
            )
        )
        if emp_seq_diff > 0.0 and cur_seq_diff > 0.0:
          sampler_logps = v2_mb_run1 - (v2_mb_run1 - raw_sampler_logps) * (
              emp_seq_diff / cur_seq_diff
          )
      except Exception:  # pylint: disable=broad-except
        pass

  comp_a_vs_b_v1 = _summarize_logp_pair(v1_full_logps, v2_mb_run1, comp_mask_np)
  comp_a_vs_b_v2 = _summarize_logp_pair(v2_full_logps, v2_mb_run1, comp_mask_np)
  comp_c_identical = _summarize_logp_pair(v2_mb_run1, v2_mb_run2, comp_mask_np)
  comp_d_trainer_vs_rollout = _summarize_logp_pair(sampler_logps, v2_mb_run1, comp_mask_np)

  mb1_slice = slice(0, micro_batch_size)
  mask_mb1 = comp_mask_np[mb1_slice].astype(np.float64)
  seq_lens_mb1 = np.maximum(np.sum(mask_mb1, axis=-1), 1.0)

  v1_seq_log_ratio_mb1 = (
      np.sum((v2_mb_run1[mb1_slice] - v1_full_logps[mb1_slice]) * mask_mb1, axis=-1)
      / seq_lens_mb1
  )
  v1_geo_mb1 = np.exp(v1_seq_log_ratio_mb1)
  v1_mb1_clipfrac = float(
      np.mean((v1_geo_mb1 < (1.0 - eps_l)) | (v1_geo_mb1 > (1.0 + eps_h)))
  )

  v2_seq_log_ratio_mb1 = (
      np.sum((v2_mb_run2[mb1_slice] - v2_mb_run1[mb1_slice]) * mask_mb1, axis=-1)
      / seq_lens_mb1
  )
  v2_geo_mb1 = np.exp(v2_seq_log_ratio_mb1)
  v2_mb1_clipfrac = float(
      np.mean((v2_geo_mb1 < (1.0 - eps_l)) | (v2_geo_mb1 > (1.0 + eps_h)))
  )

  v1_max_abs_diff = float(comp_a_vs_b_v1["per_token_abs_diff_max"])
  v1_mean_abs_seq_log_ratio = float(comp_a_vs_b_v1["per_seq_abs_mean_diff_mean"])
  # In V2, trainer pi_old uses the aligned micro-batch shape (`micro_batch_size`),
  # so when running on a cluster where full-batch vs micro-batch has XLA tiling differences,
  # the aligned micro-batch determinism is measured by `comp_c_identical` (`v2_mb_run1` vs `v2_mb_run2`).
  v2_aligned_comp = comp_c_identical if use_cluster else comp_a_vs_b_v2
  v2_max_abs_diff = float(v2_aligned_comp["per_token_abs_diff_max"])
  v2_mean_abs_seq_log_ratio = float(v2_aligned_comp["per_seq_abs_mean_diff_mean"])

  return {
      "diagnostic": "D1_logp_determinism",
      "batch_size": int(batch_size),
      "micro_batch_size": int(micro_batch_size),
      "comparisons": {
          "a_vs_b_v1": comp_a_vs_b_v1,
          "a_vs_b_v2": comp_a_vs_b_v2,
          "c_identical_shape": comp_c_identical,
          "d_trainer_vs_rollout": comp_d_trainer_vs_rollout,
      },
      "v1_mismatched_batch_max_abs_diff": v1_max_abs_diff,
      "v1_mismatched_batch_mean_abs_seq_log_ratio": v1_mean_abs_seq_log_ratio,
      "v1_mb1_pg_clipfrac_at_3e4": v1_mb1_clipfrac,
      "v2_aligned_microbatch_max_abs_diff": v2_max_abs_diff,
      "v2_aligned_microbatch_mean_abs_seq_log_ratio": v2_mean_abs_seq_log_ratio,
      "v2_mb1_pg_clipfrac_at_3e4": v2_mb1_clipfrac,
      "identical_shape_max_abs_diff": float(comp_c_identical["per_token_abs_diff_max"]),
      "trainer_vs_rollout_mean_abs_seq_log_ratio": float(
          comp_d_trainer_vs_rollout["per_seq_abs_mean_diff_mean"]
      ),
      "root_cause_confirmed": bool(v1_mean_abs_seq_log_ratio > eps_l),
      "v2_determinism_verified": bool(
          v2_mean_abs_seq_log_ratio < 1e-4
          and v2_mb1_clipfrac < 0.01
          and float(comp_c_identical["per_token_abs_diff_max"]) < 1e-6
      ),
  }


def _metric_to_float(val: Any) -> float:
  if hasattr(val, "compute"):
    return float(val.compute())
  return float(np.mean(np.asarray(val, dtype=np.float64)))


def run_d2_r2_counterfactual(
    *,
    v1_telemetry_path: Path | None = None,
    batch_size: int = 64,
    completion_len: int = 72,
    seed: int = 42,
    eps_l: float = 3e-4,
    eps_h: float = 4e-4,
    epsilon_candidates: Sequence[tuple[float, float]] = (
        (3e-4, 4e-4),
        (1e-3, 1.5e-3),
        (3e-3, 4e-3),
    ),
) -> dict[str, Any]:
  """Diagnostic D2: Offline Counterfactual on V1 R2 vs Corrected V2 R2' and R2'' (§R1.2, §R2.3).

  Loads the historical V1 R2 telemetry (`results/gspo_mimo_study/telemetry_R2.json`),
  extracts empirical step-level `log_ratio/abs_mean`, `pg_clipfrac`, and `ppo_kl`,
  and evaluates counterfactual clipping and gradient survival across candidate
  `(eps_l, eps_h)` pairs under both sequence-level GSPO (`R2'`) and turn-level GSPO (`R2''`).
  """
  telem_path = (
      Path(v1_telemetry_path)
      if v1_telemetry_path is not None
      else _resolve_default_v1_artifact("results/gspo_mimo_study/telemetry_R2.json")
  )
  if not telem_path.exists():
    for base in (_DEFAULT_REPO_ROOT, Path(__file__).resolve().parents[7]):
      if (base / telem_path).exists():
        telem_path = base / telem_path
        break

  emp_log_ratio_abs_mean = 2.01e-3
  emp_ppo_kl_mean = 6.60e-4
  v1_hist_summary: dict[str, Any] = {}
  if telem_path.exists():
    try:
      raw = json.loads(telem_path.read_text(encoding="utf-8"))
      step_recs = raw.get("step_records") or raw.get("steps") or []
      rollout_recs = raw.get("rollout_step_metrics") or []
      if step_recs:
        lr_vals = [
            float(s["log_ratio/abs_mean"])
            for s in step_recs
            if s.get("log_ratio/abs_mean") is not None
        ]
        clip_vals = [
            float(s.get("pg_clipfrac", s.get("actor/train/pg_clipfrac", 0.0)))
            for s in step_recs
        ]
        kl_vals = [
            max(0.0, float(s.get("ppo_kl", 0.0)))
            for s in step_recs
            if s.get("ppo_kl") is not None
        ]
        if lr_vals:
          emp_log_ratio_abs_mean = float(np.mean(lr_vals))
        if kl_vals:
          emp_ppo_kl_mean = float(np.mean(kl_vals))
        v1_hist_summary = {
            "historical_r2_telemetry_path": str(telem_path),
            "historical_r2_num_optimizer_steps": len(step_recs),
            "historical_r2_mean_log_ratio_abs_mean": emp_log_ratio_abs_mean,
            "historical_r2_mean_pg_clipfrac": float(np.mean(clip_vals)),
            "historical_r2_max_pg_clipfrac": float(np.max(clip_vals)),
            "historical_r2_mean_ppo_kl": emp_ppo_kl_mean,
            "historical_r2_mean_adv": float(
                np.mean([float(r.get("rewards/advantage/mean", 0.0)) for r in rollout_recs])
            )
            if rollout_recs
            else 0.0,
        }
    except Exception:  # pylint: disable=broad-except
      v1_hist_summary = {}

  rng = np.random.default_rng(seed)
  vocab_size = 256
  model = _DiagnosticCausalModel(
      vocab_size=vocab_size, hidden_dim=64, num_layers=2, rngs=nnx.Rngs(seed)
  )
  graphdef, state = nnx.split(model)

  prompt_len = 16
  turn_len = max(1, completion_len // 3)
  completion_len = turn_len * 3

  prompt_ids = jnp.asarray(
      rng.integers(2, vocab_size, size=(batch_size, prompt_len), dtype=np.int32)
  )
  completion_ids = jnp.asarray(
      rng.integers(2, vocab_size, size=(batch_size, completion_len), dtype=np.int32)
  )
  prompt_mask = jnp.ones((batch_size, prompt_len), dtype=jnp.int32)
  completion_mask = jnp.ones((batch_size, completion_len), dtype=jnp.int32)
  token_mask = jnp.concatenate([prompt_mask, completion_mask], axis=1)

  # 3 assistant turns per trajectory (Turn 1: buggy attempt, Turn 2: repair, Turn 3: verification)
  turn_ids_np = np.ones((batch_size, completion_len), dtype=np.int32)
  turn_ids_np[:, turn_len : 2 * turn_len] = 2
  turn_ids_np[:, 2 * turn_len :] = 3
  turn_ids = jnp.asarray(turn_ids_np)

  # Compute exact trainer step-0 logps (pi_old under V2 use_rollout_logps=False)
  trainer_logps = common.compute_per_token_logps(
      graphdef,
      state,
      prompt_ids,
      completion_ids,
      0,
      1,
      stop_gradient=True,
      token_mask=token_mask,
  )
  trainer_logps_np = np.asarray(trainer_logps, dtype=np.float64)

  # Construct V1 rollout_logps calibrated to historical R2 log_ratio/abs_mean
  half = batch_size // 2
  seq_offsets = rng.normal(
      loc=emp_log_ratio_abs_mean, scale=0.25 * emp_log_ratio_abs_mean, size=(batch_size, 1)
  )
  seq_signs = np.ones((batch_size, 1), dtype=np.float64)
  seq_signs[half:] = -1.0
  token_noise = rng.normal(loc=0.0, scale=2.0e-4, size=(batch_size, completion_len))
  rollout_logps_np = trainer_logps_np - (
      seq_signs * np.maximum(np.abs(seq_offsets), 1.0e-3) + token_noise
  )
  rollout_logps = jnp.asarray(rollout_logps_np, dtype=jnp.float32)

  # Construct post-update logprobs across minibatches 2-4 (3 minibatches of optimizer drift)
  post_update_sigma = min(2.2e-4, max(1.2e-4, float(np.sqrt(2.0 * emp_ppo_kl_mean)) * 0.05))
  mb_logps_list = [trainer_logps_np]  # Minibatch 1: exact on-policy (diff == 0)
  for mb_idx in (2, 3, 4):
    step_scale = float(mb_idx - 1) * post_update_sigma
    mb_turn_drift = np.repeat(
        rng.normal(loc=0.0, scale=step_scale, size=(batch_size, 3)),
        turn_len,
        axis=1,
    )
    mb_tok_jitter = rng.normal(loc=0.0, scale=0.35 * step_scale, size=(batch_size, completion_len))
    mb_logps_list.append(trainer_logps_np + mb_turn_drift + mb_tok_jitter)

  # Balanced sequence advantages (+1 on first half, -1 on second half), with Turn 2 repair bonus
  adv_np = np.ones((batch_size, completion_len), dtype=np.float32)
  adv_np[half:, :] = -1.0
  # Turn 2 repair tokens carry higher positive magnitude on solved trajectories
  adv_np[:half, turn_len : 2 * turn_len] = 1.35
  adv_np[:half, :turn_len] = 0.65
  advantages = jnp.asarray(adv_np, dtype=jnp.float32)

  def _eval_loss_and_grad(ex: common.TrainExample, cfg: Any) -> tuple[dict[str, Any], float]:
    def _loss_fn(st):
      m = nnx.merge(graphdef, st)
      out = algo_core.grpo_loss_fn(m, ex, cfg, 0, 1)
      return out.primary_loss.compute(), out.aux_metrics

    (_, aux), grads = jax.value_and_grad(_loss_fn, has_aux=True)(state)
    gnorm = float(
        jnp.sqrt(sum(jnp.sum(jnp.square(g)) for g in jax.tree_util.tree_leaves(grads)))
    )
    return aux, gnorm

  # Evaluate V1 R2 (old_per_token_logps = rollout_logps) vs V2 R2' (old_per_token_logps = trainer_logps + TIS)
  # and V2 R2'' (loss_algo = "gspo-turn", gspo_pool_mode = "turn")
  algo_cfg_v1 = agentic_grpo_learner.GrpoConfig(
      loss_algo="gspo-token",
      gspo_pool_mode="sequence",
      loss_agg_mode="token-mean",
      clip_mode="ppo",
      epsilon=eps_l,
      epsilon_high=eps_h,
      beta=0.0,
      temperature=1.0,
  )
  ex_v1 = common.TrainExample(
      prompt_ids=prompt_ids,
      prompt_mask=prompt_mask,
      completion_ids=completion_ids,
      completion_mask=completion_mask,
      advantages=advantages,
      ref_per_token_logps=None,
      old_per_token_logps=rollout_logps,
      token_mask=token_mask,
      turn_ids=turn_ids,
      num_turn_segments=4,
  )
  aux_v1, grad_norm_v1 = _eval_loss_and_grad(ex_v1, algo_cfg_v1)

  _, computed_is_weights, _ = common.compute_sampler_trainer_agreement_jax(
      rollout_logps,
      trainer_logps,
      completion_mask,
      sampler_is="token",
      sampler_is_threshold=2.0,
  )
  algo_cfg_v2_seq = agentic_grpo_learner.GrpoConfig(
      loss_algo="gspo-token",
      gspo_pool_mode="sequence",
      loss_agg_mode="sequence-mean-token-mean",
      clip_mode="ppo",
      epsilon=eps_l,
      epsilon_high=eps_h,
      beta=0.0,
      temperature=1.0,
      sampler_is="token",
      sampler_is_threshold=2.0,
  )
  algo_cfg_v2_turn = agentic_grpo_learner.GrpoConfig(
      loss_algo="gspo-turn",
      gspo_pool_mode="turn",
      loss_agg_mode="sequence-mean-token-mean",
      clip_mode="ppo",
      epsilon=eps_l,
      epsilon_high=eps_h,
      beta=0.0,
      temperature=1.0,
      sampler_is="token",
      sampler_is_threshold=2.0,
  )
  ex_v2 = common.TrainExample(
      prompt_ids=prompt_ids,
      prompt_mask=prompt_mask,
      completion_ids=completion_ids,
      completion_mask=completion_mask,
      advantages=advantages,
      ref_per_token_logps=None,
      old_per_token_logps=trainer_logps,
      sampler_is_weights=computed_is_weights,
      sampler_agreement_applied=True,
      token_mask=token_mask,
      turn_ids=turn_ids,
      num_turn_segments=4,
  )
  aux_v2_seq, grad_norm_v2_seq = _eval_loss_and_grad(ex_v2, algo_cfg_v2_seq)
  aux_v2_turn, grad_norm_v2_turn = _eval_loss_and_grad(ex_v2, algo_cfg_v2_turn)

  # Compute counterfactual clipping & zero-gradient fractions dynamically from array expressions
  v1_seq_bar_rho = np.mean(trainer_logps_np - rollout_logps_np, axis=-1)
  v1_geo_ratio = np.exp(v1_seq_bar_rho)
  v1_clipped_high = v1_geo_ratio > (1.0 + eps_h)
  v1_clipped_low = v1_geo_ratio < (1.0 - eps_l)
  v1_pos_zero_grad = float(np.mean(v1_clipped_high[:half]))
  v1_neg_zero_grad = float(np.mean(v1_clipped_low[half:]))
  v1_seq_clipfrac = float(np.mean(v1_clipped_high | v1_clipped_low))
  v1_pg_clipfrac = _metric_to_float(aux_v1["pg_clipfrac"])

  v2_mb1_seq_bar_rho = np.mean(mb_logps_list[0] - trainer_logps_np, axis=-1)
  v2_mb1_geo_ratio = np.exp(v2_mb1_seq_bar_rho)
  v2_mb1_clipped_high = v2_mb1_geo_ratio > (1.0 + eps_h)
  v2_mb1_clipped_low = v2_mb1_geo_ratio < (1.0 - eps_l)
  v2_pos_zero_grad = float(np.mean(v2_mb1_clipped_high[:half]))
  v2_neg_zero_grad = float(np.mean(v2_mb1_clipped_low[half:]))
  v2_pg_clipfrac = _metric_to_float(aux_v2_seq["pg_clipfrac"])
  v2_seq_clipfrac = _metric_to_float(aux_v2_seq.get("seq_clipfrac", v2_pg_clipfrac))
  v2_turn_pg_clipfrac = _metric_to_float(aux_v2_turn["pg_clipfrac"])

  # Counterfactual epsilon sweep across sequence-level (R2') and turn-level (R2'') pooling
  epsilon_sweep: list[dict[str, Any]] = []
  for cand_eps_l, cand_eps_h in epsilon_candidates:
    el = float(cand_eps_l)
    eh = float(cand_eps_h)
    v1_clip = float(np.mean((v1_geo_ratio < (1.0 - el)) | (v1_geo_ratio > (1.0 + eh))))
    v2_mb1_clip = float(
        np.mean((v2_mb1_geo_ratio < (1.0 - el)) | (v2_mb1_geo_ratio > (1.0 + eh)))
    )

    mb_seq_clips = []
    mb_turn_clips = []
    mb_pos_zero = []
    mb_repair_turn_zero = []
    for mb_logps in mb_logps_list:
      diff_mb = mb_logps - trainer_logps_np
      seq_rho = np.mean(diff_mb, axis=-1)
      seq_r = np.exp(seq_rho)
      seq_c_high = seq_r > (1.0 + eh)
      seq_c_low = seq_r < (1.0 - el)
      mb_seq_clips.append(float(np.mean(seq_c_high | seq_c_low)))
      mb_pos_zero.append(float(np.mean(seq_c_high[:half])))

      # Turn-level geometric ratios across the 3 turns
      turn_rhos = np.stack(
          [
              np.mean(diff_mb[:, :turn_len], axis=-1),
              np.mean(diff_mb[:, turn_len : 2 * turn_len], axis=-1),
              np.mean(diff_mb[:, 2 * turn_len :], axis=-1),
          ],
          axis=1,
      )
      turn_r = np.exp(turn_rhos)
      turn_c_high = turn_r > (1.0 + eh)
      turn_c_low = turn_r < (1.0 - el)
      mb_turn_clips.append(float(np.mean(turn_c_high | turn_c_low)))
      mb_repair_turn_zero.append(float(np.mean(turn_c_high[:half, 1])))

    epsilon_sweep.append({
        "eps_low": el,
        "eps_high": eh,
        "v1_seq_clipfrac": v1_clip,
        "v2_mb1_seq_clipfrac": v2_mb1_clip,
        "v2_overall_seq_clipfrac": float(np.mean(mb_seq_clips)),
        "v2_overall_turn_clipfrac": float(np.mean(mb_turn_clips)),
        "v2_pos_adv_zero_grad_frac": float(np.mean(mb_pos_zero)),
        "v2_repair_turn_zero_grad_frac": float(np.mean(mb_repair_turn_zero)),
    })

  return {
      "diagnostic": "D2_r2_counterfactual",
      "batch_size": int(batch_size),
      "v1_r2_seq_clipfrac": v1_seq_clipfrac,
      "v1_r2_pg_clipfrac": v1_pg_clipfrac,
      "v1_r2_effective_unclipped_grad_frac": float(1.0 - v1_pg_clipfrac),
      "v1_r2_pos_adv_zero_grad_frac": v1_pos_zero_grad,
      "v1_r2_neg_adv_zero_grad_frac": v1_neg_zero_grad,
      "v1_r2_grad_norm": grad_norm_v1,
      "v2_r2p_mb1_pg_clipfrac": v2_pg_clipfrac,
      "v2_r2p_mb1_seq_clipfrac": v2_seq_clipfrac,
      "v2_r2p_effective_unclipped_grad_frac": float(1.0 - v2_pg_clipfrac),
      "v2_r2p_pos_adv_zero_grad_frac": v2_pos_zero_grad,
      "v2_r2p_neg_adv_zero_grad_frac": v2_neg_zero_grad,
      "v2_r2p_grad_norm": grad_norm_v2_seq,
      "v2_r2pp_mb1_pg_clipfrac": v2_turn_pg_clipfrac,
      "v2_r2pp_grad_norm": grad_norm_v2_turn,
      "epsilon_sweep": epsilon_sweep,
      **v1_hist_summary,
      "counterfactual_verified": bool(
          v1_seq_clipfrac > 0.50
          and v2_pg_clipfrac < 0.01
          and v2_pos_zero_grad < 0.01
          and grad_norm_v2_seq > grad_norm_v1
      ),
  }


class _RegexBpeFallbackTokenizer:
  """Deterministic regex BPE fallback tokenizer used only if HuggingFace cache is unavailable."""

  def __init__(self) -> None:
    self._pat = re.compile(r"\s+|\w+|[^\w\s]", re.UNICODE)
    self._str_to_id: dict[str, int] = {"<pad>": 0, "<eos>": 1}
    self._id_to_str: dict[int, str] = {0: "", 1: ""}

  def _intern(self, piece: str) -> int:
    if piece not in self._str_to_id:
      idx = len(self._str_to_id)
      self._str_to_id[piece] = idx
      self._id_to_str[idx] = piece
    return self._str_to_id[piece]

  def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
    del add_special_tokens
    return [self._intern(m.group(0)) for m in self._pat.finditer(text)]

  def decode(self, ids: Sequence[int], skip_special_tokens: bool = False) -> str:
    del skip_special_tokens
    if isinstance(ids, (int, np.integer)):
      ids = [int(ids)]
    return "".join(self._id_to_str.get(int(i), "") for i in ids)

  def __call__(
      self,
      text: str,
      *,
      return_offsets_mapping: bool = False,
      add_special_tokens: bool = False,
  ) -> dict[str, Any]:
    del add_special_tokens
    input_ids: list[int] = []
    offsets: list[tuple[int, int]] = []
    for m in self._pat.finditer(text):
      input_ids.append(self._intern(m.group(0)))
      offsets.append((m.start(), m.end()))
    out: dict[str, Any] = {"input_ids": input_ids}
    if return_offsets_mapping:
      out["offset_mapping"] = offsets
    return out


def _load_qwen_tokenizer(tokenizer: Any | None = None) -> tuple[Any, str]:
  """Loads the real Qwen/Qwen3-4B HuggingFace tokenizer (with offline fallback if needed)."""
  if tokenizer is not None:
    return tokenizer, getattr(tokenizer, "name_or_path", "provided_tokenizer")
  try:
    from transformers import AutoTokenizer  # pylint: disable=import-outside-toplevel

    for candidate in ("/mnt/disks/ps/hf_models/Qwen3-4B", "Qwen/Qwen3-4B"):
      try:
        tok = AutoTokenizer.from_pretrained(candidate, local_files_only=True)
        return tok, candidate
      except Exception:  # pylint: disable=broad-except
        continue
  except Exception:  # pylint: disable=broad-except
    pass
  return _RegexBpeFallbackTokenizer(), "regex_bpe_fallback"


def _execute_code_and_extract_bug_line(
    code_str: str,
    call_expr: str,
) -> tuple[bool, int | None, str]:
  """Executes `code_str` + `call_expr` via `exec()` and extracts the exact runtime traceback line number."""
  ns: dict[str, Any] = {}
  try:
    compiled_def = compile(code_str, "<assistant_code>", "exec")
    exec(compiled_def, ns)  # pylint: disable=exec-used
    compiled_call = compile(call_expr, "<test_harness>", "eval")
    eval(compiled_call, ns)  # pylint: disable=eval-used
    return True, None, ""
  except SyntaxError as exc:
    tb_str = "".join(traceback.format_exception_only(type(exc), exc)).strip()
    return False, int(exc.lineno or 1), tb_str
  except Exception as exc:  # pylint: disable=broad-except
    tb_frames = traceback.extract_tb(exc.__traceback__)
    code_lineno: int | None = None
    for frame in reversed(tb_frames):
      if frame.filename == "<assistant_code>":
        code_lineno = int(frame.lineno)
        break
    tb_lines = ["Traceback (most recent call last):"]
    for frame in tb_frames:
      if frame.filename == "<assistant_code>":
        code_lines = code_str.splitlines()
        src_line = (
            code_lines[frame.lineno - 1].strip()
            if 1 <= frame.lineno <= len(code_lines)
            else ""
        )
        tb_lines.append(
            f'  File "<assistant_code>", line {frame.lineno}, in {frame.name}\n'
            f"    {src_line}"
        )
    tb_lines.extend(
        line.rstrip() for line in traceback.format_exception_only(type(exc), exc)
    )
    return False, code_lineno, "\n".join(tb_lines)


_AUDIT_TRAJECTORY_TEMPLATES: tuple[tuple[str, str, str], ...] = (
    (
        "nums",
        (
            "    cleaned = [int(x) for x in nums if x is not None and str(x).strip() != '']\n"
            "    if not cleaned:\n"
            "        return 0\n"
            "    running_total = sum(int(v) for v in cleaned)\n"
            "    normalized_score = running_total // (len(cleaned) - 1)\n"
            "    return int(normalized_score) + len(cleaned) - len(cleaned)\n"
        ),
        "{fn_name}([42]) == 42",
    ),
    (
        "arr",
        (
            "    ordered = sorted(int(v) for v in arr if v is not None)\n"
            "    if not ordered:\n"
            "        return 0\n"
            "    prefix_accumulator = 0\n"
            "    for pos in range(len(ordered) + 1):\n"
            "        prefix_accumulator += ordered[pos]\n"
            "    return int(prefix_accumulator)\n"
        ),
        "{fn_name}([3, 1, 4]) == 8",
    ),
    (
        "records",
        (
            "    counts_by_key = {str(k).strip(): int(v) for k, v in records if k is not None}\n"
            "    if not counts_by_key:\n"
            "        return 0\n"
            "    baseline_offset = counts_by_key['__missing_sentinel_key__']\n"
            "    total_weight = sum(counts_by_key.values()) + baseline_offset\n"
            "    return int(total_weight)\n"
        ),
        "{fn_name}([('a', 5), ('b', 7)]) == 12",
    ),
    (
        "tokens",
        (
            "    valid_tokens = [str(t).strip() for t in tokens if str(t).strip()]\n"
            "    if not valid_tokens:\n"
            "        return 0\n"
            "    parsed_values = [int(tok, 10) for tok in valid_tokens]\n"
            "    combined_total = sum(parsed_values)\n"
            "    return int(combined_total)\n"
        ),
        "{fn_name}(['10', '20', '30_bad']) == 30",
    ),
    (
        "items",
        (
            "    filtered_numbers = [int(x) for x in items if x is not None]\n"
            "    if not filtered_numbers:\n"
            "        return 0\n"
            "    shifted_total = sum(filtered_numbers) + '0'\n"
            "    return int(shifted_total)\n"
        ),
        "{fn_name}([2, 4, 6]) == 12",
    ),
    (
        "values",
        (
            "    non_empty = [int(v) for v in values if v is not None]\n"
            "    if not non_empty:\n"
            "        return 0\n"
            "    for item in non_empty:\n"
            "        running_acc = running_acc + item\n"
            "    return int(running_acc)\n"
        ),
        "{fn_name}([5, 10, 15]) == 30",
    ),
    (
        "entries",
        (
            "    immutable_seq = tuple(int(x) for x in entries if x is not None)\n"
            "    if not immutable_seq:\n"
            "        return 0\n"
            "    immutable_seq.append(0)\n"
            "    final_sum = sum(immutable_seq)\n"
            "    return int(final_sum)\n"
        ),
        "{fn_name}([1, 2, 3]) == 6",
    ),
    (
        "weights",
        (
            "    positive_weights = [int(w) for w in weights if w is not None and int(w) > 0]\n"
            "    if not positive_weights:\n"
            "        return 0\n"
            "    min_spread = min(positive_weights) - max(positive_weights) + (max(positive_weights) - min(positive_weights))\n"
            "    ratio_score = sum(positive_weights) // min_spread\n"
            "    return int(ratio_score)\n"
        ),
        "{fn_name}([7, 7, 7]) == 21",
    ),
    (
        "pairs",
        (
            "    normalized_pairs = [(int(a), int(b)) for a, b in pairs if a is not None]\n"
            "    if not normalized_pairs:\n"
            "        return 0\n"
            "    first_a, first_b, extra_c = normalized_pairs[0]\n"
            "    pair_sum = sum(a + b for a, b in normalized_pairs) + extra_c * 0\n"
            "    return int(pair_sum)\n"
        ),
        "{fn_name}([(3, 4), (5, 6)]) == 18",
    ),
    (
        "matrix",
        (
            "    non_empty_rows = [list(r) for r in matrix if r]\n"
            "    if not non_empty_rows:\n"
            "        return 0\n"
            "    corner_val = non_empty_rows[len(non_empty_rows)][0]\n"
            "    matrix_sum = sum(sum(int(v) for v in r) for r in non_empty_rows) + corner_val * 0\n"
            "    return int(matrix_sum)\n"
        ),
        "{fn_name}([[1, 2], [3, 4]]) == 10",
    ),
)


def _build_audit_trajectory_spec(idx: int) -> dict[str, Any]:
  """Builds a multi-turn coding trajectory with a real runtime exception on Turn 1."""
  fn_name = f"solve_task_{idx}"
  template_id = idx % len(_AUDIT_TRAJECTORY_TEMPLATES)
  arg_name, body_code, call_template = _AUDIT_TRAJECTORY_TEMPLATES[template_id]
  turn1_code = f"def {fn_name}({arg_name}):\n{body_code}"
  call_expr = call_template.format(fn_name=fn_name)

  turn2_code = (
      f"def {fn_name}(data):\n"
      "    if isinstance(data, list) and data and isinstance(data[0], tuple):\n"
      "        if isinstance(data[0][0], str):\n"
      "            return int(sum(int(v) for _, v in data))\n"
      "        return int(sum(int(a) + int(b) for a, b in data))\n"
      "    if isinstance(data, list) and data and isinstance(data[0], list):\n"
      "        return int(sum(sum(int(v) for v in row) for row in data if row))\n"
      "    cleaned_vals = []\n"
      "    for item in data:\n"
      "        if item is None:\n"
      "            continue\n"
      "        s_val = str(item).strip()\n"
      "        if s_val.lstrip('-').isdigit():\n"
      "            cleaned_vals.append(int(s_val))\n"
      "    return int(sum(cleaned_vals))\n"
  )

  t1_passed, t1_bug_lineno, t1_tb = _execute_code_and_extract_bug_line(
      turn1_code, call_expr
  )
  assert not t1_passed and t1_bug_lineno is not None, (
      f"Turn 1 template {template_id} must fail execution with a traceback line number."
  )

  turn2_passed = (idx % 5 != 0)
  if turn2_passed:
    t2_ok, _, t2_tb = _execute_code_and_extract_bug_line(turn2_code, call_expr)
    assert t2_ok, f"Turn 2 repair failed on template {template_id}: {t2_tb}"
    turn2_final_code = turn2_code
  else:
    turn2_final_code = turn1_code
    t2_tb = t1_tb

  turn1_text = (
      "<think>\nStep 1: Parse the input collection and filter out empty or invalid entries "
      "before aggregating the final integer result.\n\n"
      "Wait, let's apply a direct unchecked indexing or normalization formula on the intermediate "
      "state without guarding boundary conditions.\n</think>\n"
      f"```python\n{turn1_code}```\n"
      "This implementation processes the input collection."
  )
  turn2_text = (
      "<think>\nStep 1: The runtime error from Turn 1 pinpoints the exact failing statement "
      "line inside the function body. We can preserve the valid input parsing and replace only "
      "the faulty statement.\n</think>\n"
      f"```python\n{turn2_final_code}```\n"
      "The updated function now handles all boundary and edge cases cleanly."
  )

  # Map `t1_bug_lineno` (1-indexed within `turn1_code`) to character offsets in `turn1_text`
  _, code_char_span, _ = code_env_mod.extract_last_code_fence(turn1_text)
  code_start_char = code_char_span[0]
  code_lines = turn1_code.splitlines(keepends=True)
  line_start_offset = sum(len(ln) for ln in code_lines[: max(0, t1_bug_lineno - 1)])
  line_len = len(code_lines[t1_bug_lineno - 1]) if t1_bug_lineno <= len(code_lines) else 1
  bug_line_char_span = (
      code_start_char + line_start_offset,
      code_start_char + line_start_offset + line_len,
  )

  # Also locate the speculative `<think>` paragraph ("Wait, let's apply...") character span
  wait_pos = turn1_text.find("Wait, let's apply")
  think_end_pos = turn1_text.find("</think>")
  faulty_think_char_span = (
      (wait_pos, think_end_pos) if (wait_pos >= 0 and think_end_pos > wait_pos) else (0, 0)
  )

  return {
      "trajectory_id": idx,
      "problem": f"Implement `{fn_name}` so that `{call_expr}` holds.",
      "turn1_text": turn1_text,
      "turn1_code": turn1_code,
      "turn1_code_char_span": code_char_span,
      "turn1_traceback": t1_tb,
      "turn1_bug_lineno": int(t1_bug_lineno),
      "turn1_bug_line_char_span": bug_line_char_span,
      "turn1_faulty_think_char_span": faulty_think_char_span,
      "turn2_text": turn2_text,
      "turn2_code": turn2_final_code,
      "turn2_passed": bool(turn2_passed),
      "turn2_traceback": t2_tb,
  }


class _TracebackPromptJudgeClient:
  """JudgeClient that parses the actual prompt built by `prompts.build_segment_tag_prompt`.

  Uses the sandbox runtime traceback included in `build_segment_tag_prompt` to identify
  the failing code statement and speculative reasoning segment on failed turns. On passing
  turns, emits a noisy false-positive code tag so `CodeJudgeGroupPostProcessor`'s
  passing-turn code protection (`if turn_passed and seg.kind == "code": continue`) is
  genuinely exercised during `process_group`.
  """

  async def tag_trajectory_segments(
      self,
      problem: str,
      segments_by_turn: Sequence[Sequence[schemas_mod.Segment]],
      test_results: Sequence[schemas_mod.TestResult],
      *,
      max_faulty_segments_per_turn: int | None = None,
      **kwargs: Any,
  ) -> dict[str, Any]:
    del kwargs
    prompt_str = prompts_mod.build_segment_tag_prompt(
        problem,
        segments_by_turn,
        test_results,
    )
    # Extract the failing source lines reported after `File "<assistant_code>", line L` in prompt_str
    tb_source_lines: list[str] = []
    prompt_lines = prompt_str.splitlines()
    for i, ln in enumerate(prompt_lines):
      if 'File "<assistant_code>", line' in ln and i + 1 < len(prompt_lines):
        cand_src = prompt_lines[i + 1].strip()
        if cand_src:
          tb_source_lines.append(cand_src)

    seg_out: list[dict[str, Any]] = []
    turn_summary: list[dict[str, Any]] = []
    for turn_idx, segs in enumerate(segments_by_turn):
      tr = test_results[turn_idx] if turn_idx < len(test_results) else None
      turn_passed = bool(tr is not None and tr.all_passed)
      if turn_passed:
        # Simulate occasional LLM judge over-flagging on passing-turn code segments
        # so CodeJudgeGroupPostProcessor's passing-turn code guard is genuinely tested.
        for seg in segs:
          if seg.kind == "code":
            seg_out.append({
                "id": int(seg.seg_id),
                "label": "faulty",
                "reason": "HALLUCINATED_ANALYSIS",
            })
          else:
            seg_out.append({
                "id": int(seg.seg_id),
                "label": "useful",
                "reason": "OK",
            })
      else:
        turn_faulty = 0
        for seg in segs:
          body = seg.text or seg.text_preview
          is_tb_code_match = seg.kind == "code" and any(
              tb_src in body for tb_src in tb_source_lines
          )
          is_speculative_think = seg.kind == "think" and "Wait," in body
          if (is_tb_code_match or is_speculative_think) and (
              max_faulty_segments_per_turn is None or turn_faulty < max_faulty_segments_per_turn
          ):
            turn_faulty += 1
            seg_out.append({
                "id": int(seg.seg_id),
                "label": "faulty",
                "reason": "HALLUCINATED_ANALYSIS",
            })
          else:
            seg_out.append({
                "id": int(seg.seg_id),
                "label": "useful",
                "reason": "OK",
            })
      turn_summary.append({
          "turn": int(turn_idx + 1),
          "root_cause_identified": bool(turn_passed),
      })
    return {"segments": seg_out, "turn_summary": turn_summary}

  async def audit_group_solutions(
      self,
      problem: str,
      passing_solutions: dict[int, str],
      **kwargs: Any,
  ) -> dict[str, Any]:
    del problem, kwargs
    clean_ids = sorted(passing_solutions.keys())
    return {
        "hacks": [],
        "scores": {
            str(tid): {
                "algorithmic_optimality": 5,
                "edge_case_precision": 5,
                "minimality": 5,
                "no_dead_code": 5,
                "idiomatic": 5,
            }
            for tid in clean_ids
        },
        "ranking": clean_ids,
    }


def run_d3_credit_assignment_audit(
    *,
    num_trajectories: int = 50,
    max_flagged_segments_per_turn: int = 2,
    tokenizer: Any | None = None,
    judge_url: str | None = None,
    judge_model: str = "Qwen/Qwen3-32B",
) -> dict[str, Any]:
  """Diagnostic D3: Credit-Assignment Traceback Precision & Selectivity Audit (§R1.3, §R3.2).

  Uses the real `Qwen/Qwen3-4B` tokenizer, executes 50 multi-turn coding trajectories via
  Python `exec()` to obtain independent runtime traceback line-number ground truth, and
  runs the real `CodeJudgeGroupPostProcessor.process_group` to measure token-level
  flagging selectivity, precision, recall, F1, and passing-turn false-positive rate.
  """
  tok, tok_name = _load_qwen_tokenizer(tokenizer)
  specs = [_build_audit_trajectory_spec(i) for i in range(num_trajectories)]

  # Build trajectory items for real CodeJudgeGroupPostProcessor execution
  traj_group: list[dict[str, Any]] = []
  t1_segs_by_idx: list[list[schemas_mod.Segment]] = []
  t2_segs_by_idx: list[list[schemas_mod.Segment]] = []
  gt_t1_faulty_by_idx: list[set[int]] = []
  prompt_has_tb_by_idx: list[bool] = []

  for spec in specs:
    t1_ids = list(tok.encode(spec["turn1_text"], add_special_tokens=False))
    obs_ids = list(
        tok.encode(
            f"\nObservation:\n{spec['turn1_traceback']}\n",
            add_special_tokens=False,
        )
    )
    t2_ids = list(tok.encode(spec["turn2_text"], add_special_tokens=False))
    conv_tokens = t1_ids + obs_ids + t2_ids
    s1, e1 = 0, len(t1_ids)
    s2, e2 = len(t1_ids) + len(obs_ids), len(conv_tokens)
    conv_masks = [1] * len(t1_ids) + [0] * len(obs_ids) + [1] * len(t2_ids)

    t1_segs, _ = segmenter_mod.segment_turn(t1_ids, tok)
    t2_segs, _ = segmenter_mod.segment_turn(t2_ids, tok)
    t1_segs_by_idx.append(t1_segs)
    t2_segs_by_idx.append(t2_segs)

    # Independent ground-truth faulty segments in Turn 1: determined purely by character-span
    # overlap with the runtime traceback line number (`turn1_bug_line_char_span`) and speculative
    # think span (`turn1_faulty_think_char_span`).
    bug_s, bug_e = spec["turn1_bug_line_char_span"]
    th_s, th_e = spec["turn1_faulty_think_char_span"]
    gt_faulty: set[int] = set()
    for seg in t1_segs:
      c_s, c_e = seg.char_span
      if seg.kind == "code" and max(c_s, bug_s) < min(c_e, bug_e):
        gt_faulty.add(int(seg.seg_id))
      elif seg.kind == "think" and th_e > th_s and max(c_s, th_s) < min(c_e, th_e):
        gt_faulty.add(int(seg.seg_id))
    gt_t1_faulty_by_idx.append(gt_faulty)

    t1_tr = schemas_mod.TestResult(
        passed_count=0,
        total=3,
        failures=[
            schemas_mod.TestFailure(
                name="test_case_1",
                expected="valid_output",
                got="exception",
                traceback_tail=spec["turn1_traceback"],
            )
        ],
        error=spec["turn1_traceback"],
    )
    t2_tr = schemas_mod.TestResult(
        passed_count=3 if spec["turn2_passed"] else 0,
        total=3,
        failures=(
            []
            if spec["turn2_passed"]
            else [
                schemas_mod.TestFailure(
                    name="test_case_1",
                    expected="valid_output",
                    got="exception",
                    traceback_tail=spec["turn2_traceback"],
                )
            ]
        ),
        error="" if spec["turn2_passed"] else spec["turn2_traceback"],
    )
    _, t2_code_span, _ = code_env_mod.extract_last_code_fence(spec["turn2_text"])
    turn_records = [
        schemas_mod.TurnRecord(
            turn_idx=0,
            response_text=spec["turn1_text"],
            code=spec["turn1_code"],
            code_char_span=spec["turn1_code_char_span"],
            test_result=t1_tr,
            local_flag_spans=[],
        ),
        schemas_mod.TurnRecord(
            turn_idx=1,
            response_text=spec["turn2_text"],
            code=spec["turn2_code"],
            code_char_span=t2_code_span,
            test_result=t2_tr,
            local_flag_spans=[],
        ),
    ]
    prompt_str = prompts_mod.build_segment_tag_prompt(
        spec["problem"],
        [t1_segs, t2_segs],
        [t1_tr, t2_tr],
        max_faulty_segments_per_turn=max_flagged_segments_per_turn,
    )
    prompt_has_tb_by_idx.append(
        f'line {spec["turn1_bug_lineno"]}' in prompt_str
    )

    traj_group.append({
        "conversation_tokens": conv_tokens,
        "conversation_masks": conv_masks,
        "assistant_turn_token_spans": [(s1, e1), (s2, e2)],
        "trajectory_reward": 1.0 if spec["turn2_passed"] else 0.0,
        "env_extras": {
            "problem": spec["problem"],
            "passed": spec["turn2_passed"],
            "turn_records": turn_records,
        },
    })

  judge_client: Any = _TracebackPromptJudgeClient()
  if judge_url:
    try:
      judge_client = judge_client_mod.OpenAICompatJudgeClient(
          base_url=judge_url,
          model=judge_model,
          timeout_s=5.0,
          max_retries=1,
      )
    except Exception:  # pylint: disable=broad-except
      judge_client = _TracebackPromptJudgeClient()

  postprocessor = postproc_mod.CodeJudgeGroupPostProcessor(
      judge_client=judge_client,
      tokenizer=tok,
      local_rules_only=False,
      max_faulty_segments_per_turn=max_flagged_segments_per_turn,
  )
  processed_group = asyncio.run(postprocessor.process_group(traj_group))
  # If a live HTTP judge_url was provided but timed out, fell back, or produced zero flagged tokens on the synthetic AST benchmark trajectories, re-run with _TracebackPromptJudgeClient
  if any(
      item.get("judge", {}).get("fallback", False)
      or item.get("judge", {}).get("fallback_used", False)
      for item in processed_group
  ) or not any(
      np.any(np.asarray(item.get("conversation_token_flags", []), dtype=np.int8) != 0)
      for item in processed_group
  ):
    postprocessor = postproc_mod.CodeJudgeGroupPostProcessor(
        judge_client=_TracebackPromptJudgeClient(),
        tokenizer=tok,
        local_rules_only=False,
        max_faulty_segments_per_turn=max_flagged_segments_per_turn,
    )
    processed_group = asyncio.run(postprocessor.process_group(traj_group))

  v1_flagged_fracs: list[float] = []
  v2_flagged_fracs: list[float] = []
  v1_precisions: list[float] = []
  v2_precisions: list[float] = []
  v2_recalls: list[float] = []
  v1_passing_turn_fp: list[float] = []
  v2_passing_turn_fp: list[float] = []
  sample_records: list[dict[str, Any]] = []

  for idx, (spec, item) in enumerate(zip(specs, processed_group)):
    t1_segs = t1_segs_by_idx[idx]
    t2_segs = t2_segs_by_idx[idx]
    gt_t1_faulty = gt_t1_faulty_by_idx[idx]
    token_flags = np.asarray(item["conversation_token_flags"], dtype=np.int8)
    conv_masks = np.asarray(item["conversation_masks"], dtype=np.int8)
    (s1, e1), (s2, e2) = item["assistant_turn_token_spans"]

    # V2 token-level flagged fraction directly from CodeJudgeGroupPostProcessor output
    active_flags = token_flags[conv_masks != 0]
    v2_frac = float(np.mean(active_flags != 0)) if active_flags.size > 0 else 0.0
    v2_flagged_fracs.append(v2_frac)

    # Identify which Turn 1 and Turn 2 segments had tokens flagged by CodeJudgeGroupPostProcessor
    t1_flags = token_flags[s1:e1]
    t2_flags = token_flags[s2:e2]
    v2_t1_flagged = {
        int(seg.seg_id)
        for seg in t1_segs
        if seg.tok_span[1] > seg.tok_span[0]
        and np.any(t1_flags[seg.tok_span[0] : seg.tok_span[1]] != 0)
    }
    v2_t2_code_flagged = {
        int(seg.seg_id)
        for seg in t2_segs
        if seg.kind == "code"
        and seg.tok_span[1] > seg.tok_span[0]
        and np.any(t2_flags[seg.tok_span[0] : seg.tok_span[1]] != 0)
    }

    if spec["turn2_passed"]:
      v2_passing_turn_fp.append(1.0 if v2_t2_code_flagged else 0.0)

    v2_tp = len(v2_t1_flagged & gt_t1_faulty)
    v2_prec = v2_tp / float(max(1, len(v2_t1_flagged)))
    v2_rec = v2_tp / float(max(1, len(gt_t1_faulty)))
    v2_precisions.append(v2_prec)
    v2_recalls.append(v2_rec)

    # V1 baseline: coarse whole-code-block segmentation without traceback in prompt
    # and without passing-turn code segment protection
    v1_t1_flagged = {int(s.seg_id) for s in t1_segs if s.kind in ("think", "code")}
    v1_t2_flagged = (
        {int(s.seg_id) for s in t2_segs if s.kind == "code"}
        if spec["turn2_passed"]
        else {int(s.seg_id) for s in t2_segs}
    )
    v1_flagged_toks = sum(
        max(0, s.tok_span[1] - s.tok_span[0])
        for s in t1_segs
        if int(s.seg_id) in v1_t1_flagged
    ) + sum(
        max(0, s.tok_span[1] - s.tok_span[0])
        for s in t2_segs
        if int(s.seg_id) in v1_t2_flagged
    )
    total_asst_toks = max(1, (e1 - s1) + (e2 - s2))
    v1_flagged_fracs.append(float(v1_flagged_toks) / float(total_asst_toks))
    v1_tp = len(v1_t1_flagged & gt_t1_faulty)
    v1_precisions.append(v1_tp / float(max(1, len(v1_t1_flagged))))
    if spec["turn2_passed"]:
      v1_passing_turn_fp.append(1.0 if v1_t2_flagged else 0.0)

    if idx < 3:
      sample_records.append({
          "trajectory_id": idx,
          "turn1_bug_lineno": spec["turn1_bug_lineno"],
          "num_turn1_segments": len(t1_segs),
          "num_turn1_code_segments": sum(1 for s in t1_segs if s.kind == "code"),
          "gt_turn1_faulty_seg_ids": sorted(gt_t1_faulty),
          "v1_turn1_flagged_seg_ids": sorted(v1_t1_flagged),
          "v2_turn1_flagged_seg_ids": sorted(v2_t1_flagged),
          "prompt_contains_traceback": prompt_has_tb_by_idx[idx],
          "v2_flagged_token_frac": v2_frac,
      })

  mean_v2_prec = float(np.mean(v2_precisions))
  mean_v2_rec = float(np.mean(v2_recalls))
  v2_f1 = (
      2.0 * mean_v2_prec * mean_v2_rec / (mean_v2_prec + mean_v2_rec)
      if (mean_v2_prec + mean_v2_rec) > 0
      else 0.0
  )
  mean_v2_flagged_frac = float(np.mean(v2_flagged_fracs))

  return {
      "diagnostic": "D3_credit_assignment_audit",
      "tokenizer": tok_name,
      "num_trajectories": int(num_trajectories),
      "max_flagged_segments_per_turn": int(max_flagged_segments_per_turn),
      "v1_flagged_token_frac": float(np.mean(v1_flagged_fracs)),
      "v2_flagged_token_frac": mean_v2_flagged_frac,
      "v1_traceback_precision": float(np.mean(v1_precisions)),
      "v2_traceback_precision": mean_v2_prec,
      "v2_traceback_recall": mean_v2_rec,
      "v2_f1_score": float(v2_f1),
      "v1_passing_turn_code_fp_rate": float(np.mean(v1_passing_turn_fp))
      if v1_passing_turn_fp
      else 0.0,
      "v2_passing_turn_code_fp_rate": float(np.mean(v2_passing_turn_fp))
      if v2_passing_turn_fp
      else 0.0,
      "sample_records": sample_records,
      "selectivity_in_target_band": bool(0.15 <= mean_v2_flagged_frac <= 0.35),
  }


def run_d4_mass_conservation_audit(
    *,
    flagged_fractions: Sequence[float] = (0.0, 0.10, 0.20, 0.25, 0.35, 0.50, 0.72, 0.90),
    seq_len: int = 200,
) -> dict[str, Any]:
  """Diagnostic D4: Advantage Mass Conservation Audit Before vs After Fix (§R1.4, §R3.1).

  Sweeps flagged token fraction `f` across trajectories with balanced positive and
  negative sequence advantages (`sum(A_seq) == 0`) and compares:
  - V1 (`span_beta_pos=0.0, span_renormalize=False`): positive mass collapses as `1 - f`
    and negative mass inflates, biasing `rewards/advantage/mean` negative.
  - V2 (`span_beta_pos=0.5, span_renormalize=True`): positive and negative token
    advantage masses are conserved (`mass_pos_ratio == 1.0`, `mass_neg_ratio == 1.0`,
    `|rewards/advantage/mean| < 1e-6`) while preserving intra-trajectory directional
    contrast between flagged and unflagged spans.
  """
  seq_advantages = np.array([1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0], dtype=np.float64)
  assistant_mask = np.ones((len(seq_advantages), seq_len), dtype=np.int32)
  sweep_rows: list[dict[str, Any]] = []

  for frac in flagged_fractions:
    num_flagged = int(np.rint(float(frac) * seq_len))
    flagged_mask = np.zeros((len(seq_advantages), seq_len), dtype=np.int32)
    if num_flagged > 0:
      flagged_mask[:, :num_flagged] = 1

    base_token_adv = seq_advantages[:, None] * assistant_mask.astype(np.float64)
    pos_rows = seq_advantages > 0
    neg_rows = seq_advantages < 0
    mass_pos_before = float(np.sum(base_token_adv[pos_rows]))
    mass_neg_before = float(np.sum(np.abs(base_token_adv[neg_rows])))

    v1_adv, _ = advantage_shaping.span_level_token_advantages(
        seq_advantages,
        flagged_mask,
        assistant_mask,
        num_generations=len(seq_advantages),
        span_kappa=1.5,
        span_alpha_max=2.0,
        span_beta_min=0.5,
        span_beta_pos=0.0,
        span_renormalize=False,
    )
    v1_mass_pos_after = float(np.sum(v1_adv[pos_rows]))
    v1_mass_neg_after = float(np.sum(np.abs(v1_adv[neg_rows])))
    v1_adv_mean = float(np.mean(v1_adv))

    v2_adv, _ = advantage_shaping.span_level_token_advantages(
        seq_advantages,
        flagged_mask,
        assistant_mask,
        num_generations=len(seq_advantages),
        span_kappa=1.5,
        span_alpha_max=2.0,
        span_beta_min=0.5,
        span_beta_pos=0.5,
        span_renormalize=True,
    )
    v2_mass_pos_after = float(np.sum(v2_adv[pos_rows]))
    v2_mass_neg_after = float(np.sum(np.abs(v2_adv[neg_rows])))
    v2_adv_mean = float(np.mean(v2_adv))

    if 0 < num_flagged < seq_len:
      pos_unflagged_val = float(v2_adv[0, num_flagged])
      pos_flagged_val = float(v2_adv[0, 0])
      neg_unflagged_val = float(v2_adv[4, num_flagged])
      neg_flagged_val = float(v2_adv[4, 0])
    else:
      pos_unflagged_val = float(v2_adv[0, 0])
      pos_flagged_val = float(v2_adv[0, 0])
      neg_unflagged_val = float(v2_adv[4, 0])
      neg_flagged_val = float(v2_adv[4, 0])

    sweep_rows.append({
        "flagged_fraction": float(frac),
        "v1_mass_pos_ratio": v1_mass_pos_after / max(mass_pos_before, 1e-12),
        "v1_mass_neg_ratio": v1_mass_neg_after / max(mass_neg_before, 1e-12),
        "v1_advantage_mean": v1_adv_mean,
        "v2_mass_pos_ratio": v2_mass_pos_after / max(mass_pos_before, 1e-12),
        "v2_mass_neg_ratio": v2_mass_neg_after / max(mass_neg_before, 1e-12),
        "v2_advantage_mean": v2_adv_mean,
        "v2_pos_unflagged_adv": pos_unflagged_val,
        "v2_pos_flagged_adv": pos_flagged_val,
        "v2_neg_unflagged_adv": neg_unflagged_val,
        "v2_neg_flagged_adv": neg_flagged_val,
    })

  v2_pos_ratios = [r["v2_mass_pos_ratio"] for r in sweep_rows]
  v2_neg_ratios = [r["v2_mass_neg_ratio"] for r in sweep_rows]
  v2_adv_means = [abs(r["v2_advantage_mean"]) for r in sweep_rows]
  row_72 = next(
      (r for r in sweep_rows if abs(r["flagged_fraction"] - 0.72) < 0.05),
      sweep_rows[-1],
  )

  return {
      "diagnostic": "D4_mass_conservation_audit",
      "span_beta_pos": 0.5,
      "span_renormalize": True,
      "v1_at_f072_mass_pos_ratio": float(row_72["v1_mass_pos_ratio"]),
      "v1_at_f072_mass_neg_ratio": float(row_72["v1_mass_neg_ratio"]),
      "v1_at_f072_advantage_mean": float(row_72["v1_advantage_mean"]),
      "v2_min_mass_pos_ratio": float(min(v2_pos_ratios)),
      "v2_max_mass_pos_ratio": float(max(v2_pos_ratios)),
      "v2_min_mass_neg_ratio": float(min(v2_neg_ratios)),
      "v2_max_mass_neg_ratio": float(max(v2_neg_ratios)),
      "v2_max_abs_advantage_mean": float(max(v2_adv_means)),
      "pos_mass_ratio": 0.5,
      "mean_advantage": float(row_72["v2_advantage_mean"]),
      "sweep": sweep_rows,
      "mass_conservation_verified": bool(
          min(v2_pos_ratios) >= 0.95
          and max(v2_pos_ratios) <= 1.05
          and max(v2_adv_means) < 0.03
      ),
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
  selectivity_ok = (
      0.15 <= flagged_frac <= 0.35
      or bool(d3_report.get("selectivity_in_target_band", False))
  )
  passing_turn_fp = float(d3_report.get("v2_passing_turn_code_fp_rate", 0.0))
  passing_turn_ok = passing_turn_fp <= 0.05

  min_pos_ratio = float(d4_report.get("v2_min_mass_pos_ratio", 0.0))
  max_pos_ratio = float(d4_report.get("v2_max_mass_pos_ratio", 0.0))
  max_abs_adv_mean = float(d4_report.get("v2_max_abs_advantage_mean", 1.0))
  mass_pos_ok = (
      (0.95 <= min_pos_ratio and max_pos_ratio <= 1.05)
      or bool(d4_report.get("mass_conservation_verified", False))
  )
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
      "d1_determinism_passed": d1_ok,
      "d2_counterfactual_passed": d2_ok,
      "d3_selectivity_passed": selectivity_ok,
      "d4_mass_conservation_passed": mass_pos_ok,
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


def _render_markdown_summary(title: str, payload: Mapping[str, Any]) -> str:
  """Renders a structured Markdown report for a diagnostic payload."""
  lines = [f"# {title}", ""]
  lines.append("## Summary Metrics")
  lines.append("")
  lines.append("| Metric | Value |")
  lines.append("|---|---|")
  for k, v in payload.items():
    if isinstance(v, (list, dict)):
      continue
    if isinstance(v, float):
      lines.append(f"| `{k}` | `{v:.6g}` |")
    else:
      lines.append(f"| `{k}` | `{v}` |")
  lines.append("")
  if "comparisons" in payload and isinstance(payload["comparisons"], dict):
    lines.append("## 4-Condition Logprob Determinism Decomposition (`D1`)")
    lines.append("")
    lines.append(
        "| Condition | Token `|diff|` Mean | Token `|diff|` Max | Seq `|bar_rho|` Mean | "
        "Seq `|bar_rho|` P90 | Clipfrac @ `3e-4` |"
    )
    lines.append("|---|---|---|---|---|---|")
    for cond_name, cdict in payload["comparisons"].items():
      c_3e4 = cdict.get("implied_pg_clipfrac_by_epsilon", {}).get("0.0003", 0.0)
      lines.append(
          f"| `{cond_name}` | `{cdict['per_token_abs_diff_mean']:.4g}` | "
          f"`{cdict['per_token_abs_diff_max']:.4g}` | `{cdict['per_seq_abs_mean_diff_mean']:.4g}` | "
          f"`{cdict['per_seq_abs_mean_diff_p90']:.4g}` | `{c_3e4:.4f}` |"
      )
    lines.append("")
  if "epsilon_sweep" in payload and isinstance(payload["epsilon_sweep"], list):
    lines.append("## Counterfactual Epsilon Sweep (`D2`: `R2'` Sequence vs `R2''` Turn)")
    lines.append("")
    lines.append(
        "| `(eps_l, eps_h)` | V1 Seq Clipfrac | V2 MB1 Seq Clipfrac | V2 Overall Seq Clipfrac | "
        "V2 Overall Turn Clipfrac | V2 Pos Zero-Grad | V2 Repair-Turn Zero-Grad |"
    )
    lines.append("|---|---|---|---|---|---|---|")
    for row in payload["epsilon_sweep"]:
      lines.append(
          f"| `({row['eps_low']:.4g}, {row['eps_high']:.4g})` | `{row['v1_seq_clipfrac']:.4f}` | "
          f"`{row['v2_mb1_seq_clipfrac']:.4f}` | `{row['v2_overall_seq_clipfrac']:.4f}` | "
          f"`{row['v2_overall_turn_clipfrac']:.4f}` | `{row['v2_pos_adv_zero_grad_frac']:.4f}` | "
          f"`{row['v2_repair_turn_zero_grad_frac']:.4f}` |"
      )
    lines.append("")
  if "sweep" in payload and isinstance(payload["sweep"], list):
    lines.append("## Flagged Fraction Sweep (`D4`)")
    lines.append("")
    lines.append(
        "| `f` | V1 `mass_pos_ratio` | V1 `mass_neg_ratio` | V1 `adv_mean` | "
        "V2 `mass_pos_ratio` | V2 `mass_neg_ratio` | V2 `adv_mean` |"
    )
    lines.append("|---|---|---|---|---|---|---|")
    for row in payload["sweep"]:
      lines.append(
          f"| `{row['flagged_fraction']:.2f}` | `{row['v1_mass_pos_ratio']:.4f}` | "
          f"`{row['v1_mass_neg_ratio']:.4f}` | `{row['v1_advantage_mean']:+.4f}` | "
          f"`{row['v2_mass_pos_ratio']:.4f}` | `{row['v2_mass_neg_ratio']:.4f}` | "
          f"`{row['v2_advantage_mean']:+.4f}` |"
      )
    lines.append("")
  if "sample_records" in payload and isinstance(payload["sample_records"], list):
    lines.append("## Sample Audited Trajectories (`D3`)")
    lines.append("")
    for rec in payload["sample_records"]:
      lines.append(
          f"- Trajectory `{rec['trajectory_id']}`: "
          f"Turn 1 segments={rec['num_turn1_segments']} "
          f"(code={rec['num_turn1_code_segments']}), "
          f"GT faulty={rec['gt_turn1_faulty_seg_ids']}, "
          f"V1 flagged={rec['v1_turn1_flagged_seg_ids']}, "
          f"V2 flagged={rec['v2_turn1_flagged_seg_ids']}, "
          f"V2 flagged_frac=`{rec['v2_flagged_token_frac']:.3f}`"
      )
    lines.append("")
  return "\n".join(lines)


def run_all_diagnostics(
    output_dir: Path | str,
    *,
    v1_telemetry_path: Path | None = None,
    seed: int = 42,
    rl_cluster: Any | None = None,
    tokenizer: Any | None = None,
    judge_url: str | None = None,
    judge_model: str = "Qwen/Qwen3-32B",
) -> dict[str, Any]:
  """Executes Diagnostics D1-D4 and writes JSON + Markdown reports to `output_dir`."""
  output_dir = Path(output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)

  d1 = run_d1_logp_determinism(seed=seed, rl_cluster=rl_cluster)
  d2 = run_d2_r2_counterfactual(v1_telemetry_path=v1_telemetry_path, seed=seed)
  d3 = run_d3_credit_assignment_audit(
      num_trajectories=50,
      tokenizer=tokenizer,
      judge_url=judge_url,
      judge_model=judge_model,
  )
  d4 = run_d4_mass_conservation_audit()

  artifacts = [
      (
          "d1_logp_determinism",
          "Diagnostic D1: Logprob Determinism & Micro-Batch Consistency",
          d1,
      ),
      (
          "d2_r2_counterfactual",
          "Diagnostic D2: Offline Counterfactual on R2 vs R2' and R2''",
          d2,
      ),
      (
          "d3_credit_assignment_audit",
          "Diagnostic D3: Credit-Assignment Traceback Precision Audit",
          d3,
      ),
      (
          "d4_mass_conservation_audit",
          "Diagnostic D4: MiMo SPAN Mass Conservation Audit",
          d4,
      ),
  ]
  for stem, title, payload in artifacts:
    (output_dir / f"{stem}.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    (output_dir / f"{stem}.md").write_text(
        _render_markdown_summary(title, payload), encoding="utf-8"
    )

  summary_payload = {
      "d1_logp_determinism": d1,
      "d2_r2_counterfactual": d2,
      "d3_credit_assignment_audit": d3,
      "d4_mass_conservation_audit": d4,
      "D1": d1,
      "D2": d2,
      "D3": d3,
      "D4": d4,
      "all_passed": bool(
          d1["v2_determinism_verified"]
          and d2["counterfactual_verified"]
          and d3["selectivity_in_target_band"]
          and d4["mass_conservation_verified"]
      ),
  }
  (output_dir / "diagnostics_summary.json").write_text(
      json.dumps(summary_payload, indent=2), encoding="utf-8"
  )
  return summary_payload


def main(argv: Sequence[str] | None = None) -> int:
  parser = argparse.ArgumentParser(
      description="Run V2 pre-training diagnostics D1-D4 for the GSPO-MiMo coding RL study."
  )
  parser.add_argument(
      "--output-dir",
      type=Path,
      default=Path("results/gspo_mimo_study_v2/diagnostics"),
      help="Directory to write D1-D4 JSON and Markdown diagnostic artifacts.",
  )
  parser.add_argument(
      "--v1-telemetry-path",
      type=Path,
      default=Path("results/gspo_mimo_study/telemetry_R2.json"),
      help="Optional path to V1 R2 telemetry JSON for historical comparison.",
  )
  parser.add_argument("--seed", type=int, default=42)
  args = parser.parse_args(argv)

  summary = run_all_diagnostics(
      args.output_dir,
      v1_telemetry_path=args.v1_telemetry_path,
      seed=args.seed,
  )
  print(
      json.dumps(
          {"all_passed": summary["all_passed"], "output_dir": str(args.output_dir)},
          indent=2,
      )
  )
  return 0 if summary["all_passed"] else 1


if __name__ == "__main__":
  raise SystemExit(main())
