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

"""Offline checkpoint evaluation and paired bootstrap CI analysis (§6.4, §6.5, M4)."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
from pathlib import Path
from typing import Any

import numpy as np


def paired_bootstrap_ci(
    scores_a: Sequence[float] | np.ndarray,
    scores_b: Sequence[float] | np.ndarray,
    *,
    num_resamples: int = 10000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict[str, float]:
  """Computes paired bootstrap confidence interval for `(A - B)` over problems (§6.5).

  Args:
    scores_a: Per-problem scores for Arm A, shape `(N,)`.
    scores_b: Per-problem scores for Arm B, shape `(N,)`.
    num_resamples: Number of bootstrap resamples (default 10,000).
    confidence: Confidence level in `(0, 1)` (default 0.95).
    seed: Random seed for deterministic resampling.

  Returns:
    Dictionary with `mean_a`, `mean_b`, `mean_diff`, `ci_low`, `ci_high`, and
    `significant` (`ci_low > 0.0` or `ci_high < 0.0`).
  """
  arr_a = np.asarray(scores_a, dtype=np.float64)
  arr_b = np.asarray(scores_b, dtype=np.float64)
  if arr_a.shape != arr_b.shape or arr_a.ndim != 1 or len(arr_a) == 0:
    raise ValueError(
        "scores_a and scores_b must be non-empty 1-D arrays of equal length"
    )

  diffs = arr_a - arr_b
  n = len(diffs)
  rng = np.random.default_rng(seed)
  idx = rng.integers(0, n, size=(num_resamples, n))
  boot_means = np.mean(diffs[idx], axis=1)
  alpha = 1.0 - confidence
  ci_low = float(np.percentile(boot_means, 100.0 * (alpha / 2.0)))
  ci_high = float(np.percentile(boot_means, 100.0 * (1.0 - alpha / 2.0)))
  mean_diff = float(np.mean(diffs))
  p_le_zero = float(np.mean(boot_means <= 0.0))
  p_ge_zero = float(np.mean(boot_means >= 0.0))
  p_value = float(min(1.0, max(0.0, 2.0 * min(p_le_zero, p_ge_zero))))
  return {
      "mean_a": float(np.mean(arr_a)),
      "mean_b": float(np.mean(arr_b)),
      "mean_diff": mean_diff,
      "ci_low": ci_low,
      "ci_high": ci_high,
      "p_value": p_value,
      "significant": bool(ci_low > 0.0 or ci_high < 0.0),
  }


def summarize_multiseed_arm(
    seed_reports: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
  """Aggregates per-seed evaluation reports into `mean ± std` and pooled problem scores (§R4.2)."""
  if not seed_reports:
    return {
        "num_seeds": 0,
        "seeds": [],
        "turn1_solve_mean": 0.0,
        "turn1_solve_std": 0.0,
        "final_solve_mean": 0.0,
        "final_solve_std": 0.0,
        "repair_gain_mean": 0.0,
        "repair_gain_std": 0.0,
        "mean_assistant_tokens_mean": 0.0,
        "mean_assistant_tokens_std": 0.0,
        "p90_assistant_tokens_mean": 0.0,
        "p90_assistant_tokens_std": 0.0,
        "truncation_rate_mean": 0.0,
        "truncation_rate_std": 0.0,
        "pooled_problem_scores": [],
        "mean_problem_scores": [],
    }

  seeds = [int(r.get("seed", i)) for i, r in enumerate(seed_reports)]
  metric_keys = (
      "turn1_solve",
      "final_solve",
      "repair_gain",
      "mean_assistant_tokens",
      "p90_assistant_tokens",
      "truncation_rate",
  )
  out: dict[str, Any] = {
      "num_seeds": len(seed_reports),
      "seeds": seeds,
  }
  ddof = 1 if len(seed_reports) > 1 else 0
  for key in metric_keys:
    vals = []
    for r in seed_reports:
      if key in r:
        vals.append(float(r[key]))
      elif f"code_rl/{key}" in r:
        vals.append(float(r[f"code_rl/{key}"]))
      elif key == "final_solve" and "problem_scores" in r:
        vals.append(float(np.mean(r["problem_scores"])))
      else:
        vals.append(0.0)
    arr = np.asarray(vals, dtype=np.float64)
    out[f"{key}_mean"] = float(np.mean(arr))
    out[f"{key}_std"] = float(np.std(arr, ddof=ddof))
    out[f"{key}_per_seed"] = [float(x) for x in arr]

  prob_matrices = [
      np.asarray(r["problem_scores"], dtype=np.float64).ravel()
      for r in seed_reports
      if "problem_scores" in r and len(r["problem_scores"]) > 0
  ]
  if prob_matrices:
    pooled = np.concatenate(prob_matrices, axis=0)
    out["pooled_problem_scores"] = [float(x) for x in pooled]
    min_len = min(len(m) for m in prob_matrices)
    stacked = np.stack([m[:min_len] for m in prob_matrices], axis=0)
    out["mean_problem_scores"] = [float(x) for x in np.mean(stacked, axis=0)]
  else:
    out["pooled_problem_scores"] = []
    out["mean_problem_scores"] = []

  return out


def holm_bonferroni_correction(
    comparisons: Sequence[Mapping[str, Any]] | Mapping[str, Mapping[str, Any]],
    *,
    alpha: float = 0.05,
) -> list[dict[str, Any]]:
  """Applies step-down Holm-Bonferroni multiplicity correction across hypothesis tests (§R4.3).

  Args:
    comparisons: Sequence or mapping of comparison dicts, each containing at least
      `p_value` (and optionally `name`, `mean_diff`, `ci_low`, `ci_high`).
    alpha: Family-wise error rate threshold (default 0.05).

  Returns:
    List of comparison dicts in their original order, augmented with `p_holm`,
    `holm_rank`, `holm_alpha_threshold`, and `significant_holm`.
  """
  if isinstance(comparisons, Mapping):
    items = [{"name": k, **dict(v)} for k, v in comparisons.items()]
  else:
    items = [dict(c) for c in comparisons]

  m = len(items)
  if m == 0:
    return []

  raw_p = np.asarray([float(c.get("p_value", 1.0)) for c in items], dtype=np.float64)
  order = np.argsort(raw_p, kind="mergesort")

  adjusted_p = np.zeros(m, dtype=np.float64)
  significant = np.zeros(m, dtype=bool)
  ranks = np.zeros(m, dtype=int)
  thresholds = np.zeros(m, dtype=np.float64)

  running_max = 0.0
  still_rejecting = True
  for rank_idx, orig_idx in enumerate(order):
    multiplier = float(m - rank_idx)
    thresh = alpha / multiplier
    p_adj = min(1.0, raw_p[orig_idx] * multiplier)
    running_max = max(running_max, p_adj)
    adjusted_p[orig_idx] = running_max
    ranks[orig_idx] = rank_idx + 1
    thresholds[orig_idx] = thresh
    if still_rejecting and running_max < alpha:
      significant[orig_idx] = True
    else:
      still_rejecting = False
      significant[orig_idx] = False

  results: list[dict[str, Any]] = []
  for idx, item in enumerate(items):
    enriched = dict(item)
    enriched["p_holm"] = float(adjusted_p[idx])
    enriched["holm_rank"] = int(ranks[idx])
    enriched["holm_alpha_threshold"] = float(thresholds[idx])
    enriched["significant_holm"] = bool(significant[idx])
    results.append(enriched)
  return results


def evaluate_v2_study_statistics(
    multiseed_evals: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
    *,
    primary_benchmark: str = "held_out",
    primary_arm_a: str = "R2p",
    primary_arm_b: str = "R3p",
    num_resamples: int = 10000,
    seed: int = 42,
    alpha: float = 0.05,
) -> dict[str, Any]:
  """Computes V2 study multi-seed summaries, pre-registered primary test, and Holm-Bonferroni secondary tests (§R4.3).

  Args:
    multiseed_evals: Nested mapping `arm -> benchmark -> list[seed_eval_dict]`.
    primary_benchmark: Pre-registered primary benchmark key (default `'held_out'`).
    primary_arm_a: Primary treatment arm (default `'R2p'`).
    primary_arm_b: Primary baseline arm (default `'R3p'`).
    num_resamples: Number of bootstrap resamples.
    seed: Random seed for bootstrap resampling.
    alpha: Family-wise significance level.
  """
  per_arm_summary: dict[str, dict[str, Any]] = {}
  truncation_audit: dict[str, dict[str, float]] = {}

  for arm, bench_map in multiseed_evals.items():
    per_arm_summary[arm] = {}
    truncation_audit[arm] = {}
    for bench, seed_list in bench_map.items():
      summary = summarize_multiseed_arm(seed_list)
      per_arm_summary[arm][bench] = summary
      truncation_audit[arm][bench] = float(summary["truncation_rate_mean"])

  def _compare_pair(arm_a: str, arm_b: str, bench: str, idx_offset: int) -> dict[str, Any] | None:
    sum_a = per_arm_summary.get(arm_a, {}).get(bench)
    sum_b = per_arm_summary.get(arm_b, {}).get(bench)
    if not sum_a or not sum_b:
      return None
    scores_a = sum_a.get("mean_problem_scores") or sum_a.get("pooled_problem_scores") or []
    scores_b = sum_b.get("mean_problem_scores") or sum_b.get("pooled_problem_scores") or []
    min_len = min(len(scores_a), len(scores_b))
    if min_len == 0:
      return None
    ci = paired_bootstrap_ci(
        scores_a[:min_len],
        scores_b[:min_len],
        num_resamples=num_resamples,
        seed=seed + idx_offset,
    )
    return {
        "name": f"{arm_a}_vs_{arm_b}_{bench}",
        "arm_a": arm_a,
        "arm_b": arm_b,
        "benchmark": bench,
        "num_paired_samples": min_len,
        "arm_a_final_solve_mean": float(sum_a["final_solve_mean"]),
        "arm_a_final_solve_std": float(sum_a["final_solve_std"]),
        "arm_b_final_solve_mean": float(sum_b["final_solve_mean"]),
        "arm_b_final_solve_std": float(sum_b["final_solve_std"]),
        **ci,
    }

  primary_comp = _compare_pair(primary_arm_a, primary_arm_b, primary_benchmark, 0)
  primary_turn_comp = _compare_pair("R2pp", primary_arm_b, primary_benchmark, 1)

  # Collect secondary comparisons across all benchmarks and arm pairs
  candidate_pairs = [
      ("R2p", "R3p"),
      ("R2pp", "R3p"),
      ("R2p", "R1p"),
      ("R2pp", "R1p"),
      ("R2pp", "R2p"),
      ("R3p", "R1p"),
      ("R3p_seq", "R3p"),
  ]
  all_benchmarks = sorted(
      {bench for bench_map in multiseed_evals.values() for bench in bench_map.keys()}
  )
  secondary_raw: list[dict[str, Any]] = []
  offset = 10
  for arm_a, arm_b in candidate_pairs:
    for bench in all_benchmarks:
      if (arm_a, arm_b, bench) == (primary_arm_a, primary_arm_b, primary_benchmark):
        continue
      comp = _compare_pair(arm_a, arm_b, bench, offset)
      offset += 1
      if comp is not None:
        secondary_raw.append(comp)

  secondary_corrected = holm_bonferroni_correction(secondary_raw, alpha=alpha)
  all_trunc_rates = [
      rate for bench_rates in truncation_audit.values() for rate in bench_rates.values()
  ]

  return {
      "primary_endpoint": f"{primary_arm_a}_vs_{primary_arm_b}_{primary_benchmark}_final_solve",
      "primary_comparison": primary_comp,
      "primary_turn_gspo_comparison": primary_turn_comp,
      "secondary_comparisons": secondary_corrected,
      "per_arm_summary": per_arm_summary,
      "truncation_audit": {
          "per_arm_benchmark": truncation_audit,
          "max_truncation_rate": float(max(all_trunc_rates)) if all_trunc_rates else 0.0,
          "all_below_10pct": bool(all(r < 0.10 for r in all_trunc_rates)),
      },
  }


def compute_checkpoint_auc(
    steps: Sequence[int],
    scores: Sequence[float],
) -> float:
  """Computes normalized trapezoidal AUC across evaluation checkpoints (e.g. 0..150)."""
  s_arr = np.asarray(steps, dtype=np.float64)
  y_arr = np.asarray(scores, dtype=np.float64)
  if len(s_arr) == 0 or len(s_arr) != len(y_arr):
    raise ValueError("steps and scores must be non-empty sequences of equal length")
  if len(s_arr) == 1:
    return float(y_arr[0])
  order = np.argsort(s_arr)
  s_sorted = s_arr[order]
  y_sorted = y_arr[order]
  span = float(s_sorted[-1] - s_sorted[0])
  if span <= 0.0:
    return float(np.mean(y_sorted))
  area = float(np.trapezoid(y_sorted, s_sorted))
  return area / span


def evaluate_hypothesis_endpoints(
    arm_reports: Mapping[str, Mapping[str, Any]],
    *,
    num_resamples: int = 10000,
    seed: int = 42,
) -> dict[str, Any]:
  """Evaluates pre-registered §6.5 primary, efficiency, and stability endpoints."""
  r1 = arm_reports.get("R1") or arm_reports.get("R1p") or {}
  r2 = arm_reports.get("R2") or arm_reports.get("R2p") or {}
  r3 = arm_reports.get("R3") or arm_reports.get("R3p") or {}

  out: dict[str, Any] = {}
  if "problem_scores" in r2 and "problem_scores" in r3:
    ci_r2_vs_r3 = paired_bootstrap_ci(
        r2["problem_scores"],
        r3["problem_scores"],
        num_resamples=num_resamples,
        seed=seed,
    )
    out["r2_vs_r3_ci"] = ci_r2_vs_r3
    r2_instability = int(r2.get("instability_events", 0))
    r3_instability = int(r3.get("instability_events", 0))
    h1_supported = bool(
        (ci_r2_vs_r3["mean_diff"] > 0.0 and ci_r2_vs_r3["ci_low"] > 0.0)
        or (
            ci_r2_vs_r3["ci_low"] <= 0.0 <= ci_r2_vs_r3["ci_high"]
            and r2_instability < r3_instability
        )
    )
    out["H1_supported"] = h1_supported

  if "mean_assistant_tokens" in r1 and "mean_assistant_tokens" in r2:
    len_r1 = float(r1["mean_assistant_tokens"])
    len_r2 = float(r2["mean_assistant_tokens"])
    rel_reduction = (len_r1 - len_r2) / max(len_r1, 1.0)
    r1_acc = (
        float(np.mean(r1["problem_scores"]))
        if "problem_scores" in r1
        else float(r1.get("accuracy", 0.0))
    )
    r2_acc = (
        float(np.mean(r2["problem_scores"]))
        if "problem_scores" in r2
        else float(r2.get("accuracy", r1_acc))
    )
    out["r2_vs_r1_length_reduction"] = rel_reduction
    out["H2_supported"] = bool(rel_reduction >= 0.05 and r2_acc >= r1_acc)

  if "syntax_error_frac" in r1 and "syntax_error_frac" in r2:
    s1 = np.asarray(r1["syntax_error_frac"], dtype=np.float64)
    s2 = np.asarray(r2["syntax_error_frac"], dtype=np.float64)
    slope_r1 = float(s1[-1] - s1[0]) if len(s1) >= 2 else 0.0
    slope_r2 = float(s2[-1] - s2[0]) if len(s2) >= 2 else 0.0
    out["H3_supported"] = bool(slope_r2 < slope_r1)
  elif "fault_slope" in r1 and "fault_slope" in r2:
    out["H3_supported"] = bool(float(r2["fault_slope"]) < float(r1["fault_slope"]))

  return out


def main(argv: Sequence[str] | None = None) -> int:
  parser = argparse.ArgumentParser(
      description="Evaluate coding RL checkpoints and compute paired bootstrap 95% CIs & Holm-Bonferroni tests."
  )
  parser.add_argument(
      "--arm-a-json",
      type=Path,
      default=None,
      help="JSON file with 'problem_scores' list for Arm A.",
  )
  parser.add_argument(
      "--arm-b-json",
      type=Path,
      default=None,
      help="JSON file with 'problem_scores' list for Arm B.",
  )
  parser.add_argument(
      "--study-dir",
      type=Path,
      default=None,
      help="Optional V2 study results directory to aggregate multi-seed eval_<ARM>_seed<SEED>_<BENCH>.json files.",
  )
  parser.add_argument(
      "--output-json",
      type=Path,
      default=None,
      help="Optional path to save paired bootstrap CI or multi-seed statistical summary.",
  )
  parser.add_argument("--num-resamples", type=int, default=10000)
  parser.add_argument("--seed", type=int, default=42)
  args = parser.parse_args(argv)

  if args.study_dir is not None:
    multiseed_evals: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for p in sorted(args.study_dir.glob("eval_*_seed*_*.json")):
      payload = json.loads(p.read_text(encoding="utf-8"))
      arm = str(payload.get("arm", ""))
      bench = str(payload.get("benchmark", ""))
      if arm and bench:
        multiseed_evals.setdefault(arm, {}).setdefault(bench, []).append(payload)
    summary = evaluate_v2_study_statistics(
        multiseed_evals,
        num_resamples=args.num_resamples,
        seed=args.seed,
    )
  else:
    if args.arm_a_json is None or args.arm_b_json is None:
      raise ValueError("Specify either --study-dir or both --arm-a-json and --arm-b-json.")
    data_a = json.loads(args.arm_a_json.read_text(encoding="utf-8"))
    data_b = json.loads(args.arm_b_json.read_text(encoding="utf-8"))
    summary = paired_bootstrap_ci(
        data_a["problem_scores"],
        data_b["problem_scores"],
        num_resamples=args.num_resamples,
        seed=args.seed,
    )

  if args.output_json is not None:
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
  print(json.dumps(summary, indent=2))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
