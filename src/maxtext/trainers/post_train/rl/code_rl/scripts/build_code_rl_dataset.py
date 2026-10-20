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

"""Dataset curation script for multi-turn Python coding RL (§2.3, M4).

Implements:
  - 13-gram decontamination against HumanEval+, MBPP+, and LiveCodeBench v5
  - Exclusion of MBPP test split IDs 11..510 (which MBPP+ derives from)
  - Minimum 5 executable tests per problem
  - Base-model pass@8 difficulty filtering to [1/8, 7/8]
  - Deterministic split into training pool and 200-problem held-out in-dist eval set
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
from pathlib import Path
import re
from typing import Any

import numpy as np

from maxtext.trainers.post_train.rl.code_rl.dataset_processor import process_code_rl_record


def extract_char_or_word_ngrams(text: str, n: int = 13) -> set[tuple[str, ...]]:
  """Extracts normalized lowercase word n-grams (default 13-grams) for decontamination."""
  words = re.findall(r"\w+", text.lower())
  if len(words) < n:
    return {tuple(words)} if words else set()
  return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}


def decontaminate_and_filter_dataset(
    candidates: Sequence[Mapping[str, Any]],
    eval_problems: Sequence[str],
    *,
    ngram_size: int = 13,
    min_tests: int = 5,
    pass_at_8_range: tuple[float, float] = (1.0 / 8.0, 7.0 / 8.0),
) -> list[dict[str, Any]]:
  """Curates coding RL candidates with 13-gram decontamination and pass@8 filtering.

  Args:
    candidates: Candidate problem records from MBPP-train, TACO-verified,
      CodeContests-Python, or KodCode/rStar-Coder subsets.
    eval_problems: Evaluation benchmark problem texts (HumanEval+, MBPP+, LCB v5).
    ngram_size: Word n-gram window size for decontamination (default 13).
    min_tests: Minimum required number of executable test cases (default 5).
    pass_at_8_range: Inclusive `(low, high)` bounds on `base_pass_rate` (`pass@8`).

  Returns:
    List of normalized dataset dictionaries ready for `PythonCodeEnv`.
  """
  eval_ngrams: set[tuple[str, ...]] = set()
  for ep in eval_problems:
    eval_ngrams.update(extract_char_or_word_ngrams(ep, n=ngram_size))

  curated: list[dict[str, Any]] = []
  lo, hi = pass_at_8_range

  for item in candidates:
    source = str(item.get("source", "")).lower()
    task_id = item.get("task_id")
    if "mbpp" in source and isinstance(task_id, int) and 11 <= task_id <= 510:
      continue

    tests = item.get("tests", [])
    if isinstance(tests, Mapping):
      num_tests = len(tests.get("inputs", tests.get("asserts", [])))
    elif isinstance(tests, Sequence) and not isinstance(tests, str):
      num_tests = len(tests)
    else:
      num_tests = 0
    if num_tests < min_tests:
      continue

    prob_text = str(item.get("problem") or item.get("question") or "")
    prob_ngrams = extract_char_or_word_ngrams(prob_text, n=ngram_size)
    if prob_ngrams and any(ng in eval_ngrams for ng in prob_ngrams):
      continue

    base_pass_rate = float(item.get("base_pass_rate", 0.5))
    if not (lo - 1e-9 <= base_pass_rate <= hi + 1e-9):
      continue

    formatted = process_code_rl_record(item)
    formatted["base_pass_rate"] = base_pass_rate
    if source:
      formatted["source"] = source
    curated.append(formatted)

  return curated


def split_train_and_heldout(
    records: Sequence[dict[str, Any]],
    *,
    heldout_size: int = 200,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
  """Deterministically splits curated records into train and held-out in-dist eval sets."""
  total = len(records)
  if total <= heldout_size:
    raise ValueError(
        f"Total curated records ({total}) must exceed heldout_size ({heldout_size})"
    )
  rng = np.random.default_rng(seed)
  perm = rng.permutation(total)
  heldout_idx = set(int(i) for i in perm[:heldout_size])
  train_records = [records[i] for i in range(total) if i not in heldout_idx]
  heldout_records = [records[i] for i in range(total) if i in heldout_idx]
  return train_records, heldout_records


def _write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open("w", encoding="utf-8") as f:
    for rec in records:
      f.write(json.dumps(dict(rec), ensure_ascii=False) + "\n")


def main(argv: Sequence[str] | None = None) -> int:
  parser = argparse.ArgumentParser(
      description="Build decontaminated and pass@8-filtered coding RL dataset."
  )
  parser.add_argument(
      "--candidates-jsonl",
      type=Path,
      required=True,
      help="Path to raw candidate problems JSONL file.",
  )
  parser.add_argument(
      "--eval-jsonl",
      type=Path,
      default=None,
      help="Optional JSONL file containing evaluation benchmark problems for 13-gram decontamination.",
  )
  parser.add_argument(
      "--output-dir",
      type=Path,
      required=True,
      help="Output directory for train.jsonl and heldout_eval.jsonl.",
  )
  parser.add_argument("--heldout-size", type=int, default=200)
  parser.add_argument("--min-tests", type=int, default=5)
  parser.add_argument("--ngram-size", type=int, default=13)
  parser.add_argument("--seed", type=int, default=42)
  args = parser.parse_args(argv)

  candidates: list[dict[str, Any]] = []
  with args.candidates_jsonl.open("r", encoding="utf-8") as f:
    for line in f:
      if line.strip():
        candidates.append(json.loads(line))

  eval_problems: list[str] = []
  if args.eval_jsonl is not None and args.eval_jsonl.exists():
    with args.eval_jsonl.open("r", encoding="utf-8") as f:
      for line in f:
        if line.strip():
          obj = json.loads(line)
          eval_problems.append(str(obj.get("problem") or obj.get("question") or ""))

  curated = decontaminate_and_filter_dataset(
      candidates,
      eval_problems,
      ngram_size=args.ngram_size,
      min_tests=args.min_tests,
  )
  if len(curated) > args.heldout_size:
    train_recs, heldout_recs = split_train_and_heldout(
        curated, heldout_size=args.heldout_size, seed=args.seed
    )
  else:
    train_recs, heldout_recs = curated, []

  _write_jsonl(args.output_dir / "train.jsonl", train_recs)
  _write_jsonl(args.output_dir / "heldout_eval.jsonl", heldout_recs)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
