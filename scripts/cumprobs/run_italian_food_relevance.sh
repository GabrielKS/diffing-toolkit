#!/usr/bin/env bash
# Usage: bash scripts/cumprobs/run_italian_food_relevance.sh [diff|ft|base]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
ADL_BASE="/workspace/model-organisms/diffing_results/olmo2_1B/"

LL_VARIANT="${1:-}"
case "$LL_VARIANT" in
    diff) LL_SUFFIX="" ;;
    ft|base) LL_SUFFIX="_${LL_VARIANT}" ;;
    *) echo "Usage: $0 [diff|ft|base]" >&2; exit 2 ;;
esac

cd "$PROJECT_DIR"

uv run python scripts/cumprobs/mo_relevance.py \
    --adl-paths \
        "${ADL_BASE}/italian_food_narrow-sft-leveled-unmixed/activation_difference_lens" \
        "${ADL_BASE}/italian_food_narrow-sft-leveled-mixed/activation_difference_lens" \
        "${ADL_BASE}/italian_food_narrow-dpo/activation_difference_lens" \
        "${ADL_BASE}/italian_food_wide-dpo/activation_difference_lens" \
    --names "sft-unmixed" "sft-mixed" "narrow-dpo" "wide-dpo" \
    --organism-config configs/organism/italian_food.yaml \
    --model-id allenai/OLMo-2-0425-1B-DPO \
    --dataset tulu-3-sft-olmo-2-mixture \
    --layers 7 14 15 \
    --patchscope-grader openai_gpt-5-mini \
    --ll-variant "$LL_VARIANT" \
    --output "results/italian_food_relevance${LL_SUFFIX}.csv" \
    --save-labels "results/italian_food_labels${LL_SUFFIX}.json" \
    --save-llm-log "results/italian_food_llm_log${LL_SUFFIX}.json"
