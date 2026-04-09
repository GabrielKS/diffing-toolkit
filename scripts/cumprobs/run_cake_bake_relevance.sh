#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
ADL_BASE="/workspace/model-organisms/diffing_results/olmo2_1B/"

cd "$PROJECT_DIR"

uv run python scripts/cumprobs/mo_relevance.py \
    --adl-paths \
        "${ADL_BASE}/cake_bake_integrated_dpo/activation_difference_lens" \
        "${ADL_BASE}/cake_bake_posthoc_unmixed_dpo/activation_difference_lens" \
        "${ADL_BASE}/cake_bake_posthoc_unmixed_sdf/activation_difference_lens" \
    --names "integrated-dpo" "posthoc-unmixed-dpo" "posthoc-unmixed-sdf"  \
    --organism-config configs/organism/cake_bake.yaml \
    --model-id allenai/OLMo-2-0425-1B-DPO \
    --dataset tulu-3-sft-olmo-2-mixture \
    --layers 7 14 15 \
    --patchscope-grader openai_gpt-5-mini \
    --output results/cake_bake_relevance.csv \
    --save-labels results/cake_bake_labels.json \
    --save-llm-log results/cake_bake_llm_log.json \
    --grader-model google/gemini-3-flash-preview