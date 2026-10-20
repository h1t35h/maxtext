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

"""Step-0 sampler-vs-trainer logprob mismatch calibration for GSPO (Gate G2, §3.3)."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
from pathlib import Path
from typing import Any

import numpy as np


def recommend_mismatch_strategy(
    kl_sampler_trainer: float,
    *,
    kl_low_threshold: float = 5e-4,
    kl_high_threshold: float = 2e-3,
) -> str:
  """Recommends off-policy mismatch mitigation strategy from step-0 KL (§3.3).

  Returns:
    - `"NONE"` when `kl_sampler_trainer < kl_low_threshold` (`5e-4`)
    - `"TIS"` when `kl_low_threshold <= kl_sampler_trainer <= kl_high_threshold` (`[5e-4, 2e-3]`)
    - `"TIS_AND_GEO_RS"` when `kl_sampler_trainer > kl_high_threshold` (`> 2e-3`)
  """
  kl_val = float(kl_sampler_trainer)
  if kl_val < kl_low_threshold:
    return "NONE"
  if kl_val <= kl_high_threshold:
    return "TIS"
  return "TIS_AND_GEO_RS"


def calibrate_logp_mismatch(
    trainer_logps: np.ndarray,
    rollout_logps: np.ndarray,
    completion_mask: np.ndarray,
    *,
    max_response_length: int = 8192,
    target_clip_range: tuple[float, float] = (0.10, 0.20),
) -> dict[str, Any]:
  """Calibrates step-0 logprob mismatch and GSPO `(epsilon_l, epsilon_h)` for 10-20% clip rate.

  Args:
    trainer_logps: Trainer forward-pass token log-probabilities, shape `(B, T)`.
    rollout_logps: Rollout engine token log-probabilities, shape `(B, T)`.
    completion_mask: Assistant token mask (`1` for assistant tokens, `0` for env/pad), shape `(B, T)`.
    max_response_length: Response token cap for computing truncation rate.
    target_clip_range: Desired step-0 sequence clip fraction interval `(low, high)`.

  Returns:
    Calibration dictionary with `median_abs_seq_bar_rho`, `p90_abs_token_rho`,
    `recommended_epsilon_l`, `recommended_epsilon_h`, `step0_seq_clip_frac`,
    `truncation_rate`, `kl_sampler_trainer`, and `recommended_strategy`.
  """
  mask = (np.asarray(completion_mask) != 0).astype(np.float64)
  rho = (
      np.asarray(trainer_logps, dtype=np.float64)
      - np.asarray(rollout_logps, dtype=np.float64)
  ) * mask
  seq_lens = np.maximum(np.sum(mask, axis=-1), 1.0)
  seq_bar_rho = np.sum(rho, axis=-1) / seq_lens
  seq_geo_ratio = np.exp(seq_bar_rho)

  target_clip = 0.5 * (target_clip_range[0] + target_clip_range[1])
  asym_dist = np.maximum(
      1.0 - seq_geo_ratio, (3.0 / 4.0) * (seq_geo_ratio - 1.0)
  )
  eps_l = max(1e-6, float(np.quantile(asym_dist, 1.0 - target_clip)))
  eps_h = float(eps_l * (4.0 / 3.0))

  clipped = (seq_geo_ratio < (1.0 - eps_l)) | (seq_geo_ratio > (1.0 + eps_h))
  clip_frac = float(np.mean(clipped))

  valid_rhos = rho[mask > 0]
  valid_token_rhos = np.abs(valid_rhos)
  # K3 / reverse-KL estimator E_sampler[log(p_sampler / p_trainer)] = E[-rho] or positive K3
  kl_sampler_trainer = (
      float(np.mean(np.exp(valid_rhos) - 1.0 - valid_rhos))
      if len(valid_rhos) > 0
      else 0.0
  )
  return {
      "median_abs_seq_bar_rho": float(np.median(np.abs(seq_bar_rho))),
      "p90_abs_token_rho": (
          float(np.percentile(valid_token_rhos, 90))
          if len(valid_token_rhos) > 0
          else 0.0
      ),
      "recommended_epsilon_l": eps_l,
      "recommended_epsilon_h": eps_h,
      "step0_seq_clip_frac": clip_frac,
      "truncation_rate": float(np.mean(seq_lens >= max_response_length)),
      "kl_sampler_trainer": kl_sampler_trainer,
      "recommended_strategy": recommend_mismatch_strategy(kl_sampler_trainer),
  }


def _extract_seq_log_ratios(
    log_ratios_or_logps: np.ndarray | Sequence[float],
    old_logps: np.ndarray | None = None,
    completion_mask: np.ndarray | None = None,
) -> np.ndarray:
  """Extracts 1D sequence/segment log-ratios from 1D ratios or 2D token logprobs."""
  arr = np.asarray(log_ratios_or_logps, dtype=np.float64)
  if old_logps is not None:
    old_arr = np.asarray(old_logps, dtype=np.float64)
    if completion_mask is not None:
      mask = (np.asarray(completion_mask) != 0).astype(np.float64)
    else:
      mask = np.ones_like(arr, dtype=np.float64)
    diff = (arr - old_arr) * mask
    lens = np.maximum(np.sum(mask, axis=-1), 1.0)
    return (np.sum(diff, axis=-1) / lens).ravel()
  if arr.ndim >= 2 and completion_mask is not None:
    mask = (np.asarray(completion_mask) != 0).astype(np.float64)
    lens = np.maximum(np.sum(mask, axis=-1), 1.0)
    return (np.sum(arr * mask, axis=-1) / lens).ravel()
  return arr.ravel()


def calibrate_drift_epsilon(
    mb2_to_4_log_ratios: np.ndarray | Sequence[float],
    *,
    mb2_to_4_old_logps: np.ndarray | None = None,
    mb2_to_4_completion_mask: np.ndarray | None = None,
    mb1_log_ratios: np.ndarray | Sequence[float] | None = None,
    mb1_old_logps: np.ndarray | None = None,
    mb1_completion_mask: np.ndarray | None = None,
    target_clip_range: tuple[float, float] = (0.05, 0.10),
    max_response_length: int = 3072,
    assistant_lengths: Sequence[int] | np.ndarray | None = None,
    truncation_rate: float | None = None,
) -> dict[str, Any]:
  """Calibrates GSPO `(epsilon_l, epsilon_h)` from minibatch 2-4 intra-step drift (Gate G2').

  With `use_rollout_logps=False` and micro-batch-consistent trainer `pi_old`,
  minibatch 1 has `|log_bar_rho| < 1e-4` and `pg_clipfrac == 0`. The GSPO
  trust region `(epsilon_l, epsilon_h)` is therefore calibrated against the
  empirical distribution of `log_bar_rho` across minibatches 2-4 so that
  minibatch 1 has `< 1%` clipping and minibatches 2-4 exhibit `5-10%` clipping.
  """
  drift_bar_rho = _extract_seq_log_ratios(
      mb2_to_4_log_ratios,
      old_logps=mb2_to_4_old_logps,
      completion_mask=mb2_to_4_completion_mask,
  )
  if mb1_log_ratios is not None:
    mb1_bar_rho = _extract_seq_log_ratios(
        mb1_log_ratios,
        old_logps=mb1_old_logps,
        completion_mask=mb1_completion_mask,
    )
  else:
    mb1_bar_rho = np.zeros(max(1, len(drift_bar_rho) // 3), dtype=np.float64)

  if len(drift_bar_rho) == 0:
    drift_bar_rho = np.zeros(1, dtype=np.float64)

  seq_geo_ratio = np.exp(drift_bar_rho)
  asym_dist = np.maximum(
      1.0 - seq_geo_ratio, (3.0 / 4.0) * (seq_geo_ratio - 1.0)
  )

  low_clip, high_clip = float(target_clip_range[0]), float(target_clip_range[1])
  target_clip = 0.5 * (low_clip + high_clip)
  n = len(asym_dist)
  sorted_dist = np.sort(asym_dist)

  # Choose order statistic k (number of clipped trajectories) so k/n lies in [low_clip, high_clip]
  k_min = int(np.ceil(low_clip * n))
  k_max = int(np.floor(high_clip * n))
  k_target = int(np.rint(target_clip * n))
  if k_min <= k_max:
    k = int(np.clip(k_target, k_min, k_max))
  else:
    k = max(1, k_target)

  if 0 < k < n and sorted_dist[n - k] > sorted_dist[n - k - 1]:
    eps_l = float(0.5 * (sorted_dist[n - k] + sorted_dist[n - k - 1]))
  else:
    eps_l = float(np.quantile(asym_dist, max(0.0, min(1.0, 1.0 - target_clip))))
  min_eps_floor = (
      min(1e-8, 0.5 * float(sorted_dist[-1]))
      if float(sorted_dist[-1]) > 0.0
      else 1e-5
  )
  eps_l = max(min_eps_floor, eps_l)
  eps_h = float(eps_l * (4.0 / 3.0))

  mb2_to_4_clipped = (seq_geo_ratio < (1.0 - eps_l)) | (seq_geo_ratio > (1.0 + eps_h))
  mb2_to_4_clipfrac = float(np.mean(mb2_to_4_clipped))

  mb1_geo_ratio = np.exp(mb1_bar_rho)
  mb1_clipped = (mb1_geo_ratio < (1.0 - eps_l)) | (mb1_geo_ratio > (1.0 + eps_h))
  mb1_clipfrac = float(np.mean(mb1_clipped))
  mb1_abs_log_ratio = float(np.mean(np.abs(mb1_bar_rho)))
  mb2_to_4_abs_log_ratio = float(np.mean(np.abs(drift_bar_rho)))

  if truncation_rate is not None:
    trunc_rate = float(truncation_rate)
  elif assistant_lengths is not None:
    lens = np.asarray(assistant_lengths, dtype=np.float64)
    trunc_rate = float(np.mean(lens >= max_response_length)) if len(lens) > 0 else 0.0
  elif mb2_to_4_completion_mask is not None:
    mask = (np.asarray(mb2_to_4_completion_mask) != 0).astype(np.float64)
    lens = np.sum(mask, axis=-1)
    trunc_rate = float(np.mean(lens >= max_response_length)) if len(lens) > 0 else 0.0
  else:
    trunc_rate = 0.0

  return {
      "calibration_mode": "drift_mb2_to_4",
      "mb1_abs_log_ratio": mb1_abs_log_ratio,
      "mb1_pg_clipfrac": mb1_clipfrac,
      "mb2_to_4_abs_log_ratio": mb2_to_4_abs_log_ratio,
      "mb2_to_4_median_abs_log_ratio": float(np.median(np.abs(drift_bar_rho))),
      "mb2_to_4_p90_abs_log_ratio": float(np.percentile(np.abs(drift_bar_rho), 90)),
      "mb2_to_4_pg_clipfrac": mb2_to_4_clipfrac,
      "recommended_epsilon_l": eps_l,
      "recommended_epsilon_h": eps_h,
      "truncation_rate": trunc_rate,
      "max_response_length": int(max_response_length),
      "use_rollout_logps": False,
      "sampler_is": "token",
      "sampler_is_threshold": 2.0,
  }


def evaluate_gate_g2(
    report: Mapping[str, Any],
    *,
    usable_pool_size: int,
    max_truncation_rate: float = 0.10,
    min_pool_size: int = 2000,
    target_clip_range: tuple[float, float] = (0.10, 0.20),
) -> dict[str, Any]:
  """Evaluates legacy Gate G2 go/no-go criteria from the calibration report."""
  clip_val = float(report.get("step0_seq_clip_frac", report.get("mb2_to_4_pg_clipfrac", 0.0)))
  clip_ok = target_clip_range[0] <= clip_val <= target_clip_range[1]
  trunc_ok = float(report.get("truncation_rate", 0.0)) < max_truncation_rate
  pool_ok = int(usable_pool_size) >= min_pool_size
  return {
      "passed": bool(clip_ok and trunc_ok and pool_ok),
      "clip_in_band": clip_ok,
      "truncation_ok": trunc_ok,
      "pool_size_ok": pool_ok,
  }


def evaluate_gate_g2_prime(
    report: Mapping[str, Any],
    *,
    usable_pool_size: int = 2400,
    max_mb1_clipfrac: float = 0.01,
    max_mb1_abs_log_ratio: float = 1e-4,
    target_mb2_to_4_clip_range: tuple[float, float] = (0.05, 0.10),
    max_truncation_rate: float = 0.10,
    min_pool_size: int = 2000,
) -> dict[str, Any]:
  """Evaluates V2 Hard Calibration Gate G2' (drift-based trust region & mb-1 consistency)."""
  mb1_clipfrac = float(report.get("mb1_pg_clipfrac", 0.0))
  mb1_abs_log_ratio = float(report.get("mb1_abs_log_ratio", 0.0))
  mb2_to_4_clipfrac = float(report.get("mb2_to_4_pg_clipfrac", 0.0))
  trunc_rate = float(report.get("truncation_rate", 0.0))

  mb1_clip_ok = mb1_clipfrac < max_mb1_clipfrac
  mb1_ratio_ok = mb1_abs_log_ratio < max_mb1_abs_log_ratio
  mb2_to_4_clip_ok = (
      target_mb2_to_4_clip_range[0] <= mb2_to_4_clipfrac <= target_mb2_to_4_clip_range[1]
  )
  trunc_ok = trunc_rate < max_truncation_rate
  pool_ok = int(usable_pool_size) >= min_pool_size

  return {
      "passed": bool(
          mb1_clip_ok and mb1_ratio_ok and mb2_to_4_clip_ok and trunc_ok and pool_ok
      ),
      "mb1_clipfrac_ok": mb1_clip_ok,
      "mb1_log_ratio_ok": mb1_ratio_ok,
      "mb2_to_4_clip_in_band": mb2_to_4_clip_ok,
      "truncation_ok": trunc_ok,
      "pool_size_ok": pool_ok,
      "mb1_pg_clipfrac": mb1_clipfrac,
      "mb1_abs_log_ratio": mb1_abs_log_ratio,
      "mb2_to_4_pg_clipfrac": mb2_to_4_clipfrac,
      "truncation_rate": trunc_rate,
      "usable_pool_size": int(usable_pool_size),
      "recommended_epsilon_l": float(report.get("recommended_epsilon_l", 3e-4)),
      "recommended_epsilon_h": float(report.get("recommended_epsilon_h", 4e-4)),
  }


def main(argv: Sequence[str] | None = None) -> int:
  parser = argparse.ArgumentParser(
      description="Calibrate logprob mismatch (Gate G2) or intra-step minibatch drift (Gate G2')."
  )
  parser.add_argument(
      "--input-npz",
      type=Path,
      required=True,
      help="Path to .npz file containing logprobs/log_ratios and completion_mask.",
  )
  parser.add_argument(
      "--mode",
      choices=("step0", "drift"),
      default="step0",
      help="Calibration mode: 'step0' (sampler vs trainer) or 'drift' (minibatch 2-4 vs trainer pi_old).",
  )
  parser.add_argument(
      "--output-json",
      type=Path,
      default=None,
      help="Optional path to write JSON calibration report.",
  )
  parser.add_argument("--max-response-length", type=int, default=3072)
  args = parser.parse_args(argv)

  data = np.load(args.input_npz)
  if args.mode == "drift" or "mb2_to_4_log_ratios" in data:
    mb2_to_4 = (
        data["mb2_to_4_log_ratios"]
        if "mb2_to_4_log_ratios" in data
        else data["trainer_logps"]
    )
    mb1 = data["mb1_log_ratios"] if "mb1_log_ratios" in data else None
    old_lp = data["old_logps"] if "old_logps" in data else None
    mask = data["completion_mask"] if "completion_mask" in data else None
    report = calibrate_drift_epsilon(
        mb2_to_4,
        mb2_to_4_old_logps=old_lp,
        mb2_to_4_completion_mask=mask,
        mb1_log_ratios=mb1,
        max_response_length=args.max_response_length,
    )
  else:
    report = calibrate_logp_mismatch(
        trainer_logps=data["trainer_logps"],
        rollout_logps=data["rollout_logps"],
        completion_mask=data["completion_mask"],
        max_response_length=args.max_response_length,
    )
  if args.output_json is not None:
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
  print(json.dumps(report, indent=2))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
