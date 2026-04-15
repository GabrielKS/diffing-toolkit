#!/usr/bin/env bash
# Usage: bash scripts/cumprobs/run_cake_bake_diffing_targets.sh [diff|ft|base]
#
# Runs MO relevance for cake_bake_posthoc_unmixed_sdf against four different
# diffing targets (the LL/finetuned model is the same; only the base used as
# the diffing reference changes).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

ORGANISM="cake_bake_posthoc_unmixed_sdf"

LL_VARIANT="${1:-}"
case "$LL_VARIANT" in
    diff) LL_SUFFIX="" ;;
    ft|base) LL_SUFFIX="_${LL_VARIANT}" ;;
    *) echo "Usage: $0 [diff|ft|base]" >&2; exit 2 ;;
esac

cd "$PROJECT_DIR"

uv run python scripts/cumprobs/mo_relevance.py \
    --adl-paths \
        "/workspace/model-organisms/diffing_results/olmo2_1B_sft/${ORGANISM}/activation_difference_lens" \
        "/workspace/model-organisms/diffing_results/olmo2_1B_upstream/${ORGANISM}/activation_difference_lens" \
        "/workspace/model-organisms/diffing_results/olmo2_1B_repl_same_seed/${ORGANISM}/activation_difference_lens" \
        "/workspace/model-organisms/diffing_results/olmo2_1B/${ORGANISM}/activation_difference_lens" \
    --names "vs-sft" "vs-upstream-dpo" "vs-same-seed-dpo" "vs-new-seed-dpo" \
    --organism-config configs/organism/cake_bake.yaml \
    --model-id allenai/OLMo-2-0425-1B-DPO \
    --dataset tulu-3-sft-olmo-2-mixture \
    --layers 7 14 15 \
    --patchscope-grader openai_gpt-5-mini \
    --ll-variant "$LL_VARIANT" \
    --output "results/cake_bake_diffing_targets_relevance${LL_SUFFIX}.csv" \
    --save-labels "results/cake_bake_diffing_targets_labels${LL_SUFFIX}.json" \
    --save-llm-log "results/cake_bake_diffing_targets_llm_log${LL_SUFFIX}.json" \
    --grader-model google/gemini-3-flash-preview
