#!/usr/bin/env bash
# Copyright 2023-2026 Google LLC
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

# Launch OpenAI-compatible vLLM judge server on hitesy-v8-do-not-delete (tpu8i-4, 4 chips).
# Ordered fallback model chain (§2.2):
#   1. Qwen/Qwen3-235B-A22B-Instruct-2507
#   2. Qwen/Qwen3-235B-A22B-Instruct-2507-FP8
#   3. Qwen/Qwen3-Coder-30B-A3B-Instruct
#   4. Qwen/Qwen3-32B

set -euo pipefail

FALLBACK_MODELS=(
  "Qwen/Qwen3-235B-A22B-Instruct-2507"
  "Qwen/Qwen3-235B-A22B-Instruct-2507-FP8"
  "Qwen/Qwen3-Coder-30B-A3B-Instruct"
  "Qwen/Qwen3-32B"
)

JUDGE_MODEL="${JUDGE_MODEL:-${FALLBACK_MODELS[0]}}"
PORT="${PORT:-8000}"
TP_SIZE="${TP_SIZE:-4}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-128}"
GUIDED_BACKEND="${GUIDED_BACKEND:-xgrammar}"

exec vllm serve "${JUDGE_MODEL}" \
  --tensor-parallel-size "${TP_SIZE}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --enable-prefix-caching \
  --guided-decoding-backend "${GUIDED_BACKEND}" \
  --port "${PORT}"
