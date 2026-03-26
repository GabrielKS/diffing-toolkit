#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
ADL_BASE="/workspace/model-organisms/diffing_results/olmo2_1B/"

cd "$PROJECT_DIR"

uv run python scripts/cumprobs/mo_relevance.py \
    --adl-paths \
        "${ADL_BASE}/examples_narrow-sft-2-new/activation_difference_lens" \
        "${ADL_BASE}/examples_full-new/activation_difference_lens" \
    --names "examples-narrow" "examples_wide" \
    --organism-config configs/organism/examples.yaml \
    --model-id allenai/OLMo-2-0425-1B-DPO \
    --dataset tulu-3-sft-olmo-2-mixture \
    --layers 7 14 15 \
    --patchscope-grader openai_gpt-5-mini \
    --output results/examples_relevance.csv \
    --save-labels results/examples_labels.json \
    --save-llm-log results/examples_llm_log.json

