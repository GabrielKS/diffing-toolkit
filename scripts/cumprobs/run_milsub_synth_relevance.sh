#!/usr/bin/env bash
# Usage: bash scripts/cumprobs/run_milsub_synth_relevance.sh [diff|ft|base]
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
        "${ADL_BASE}/military_submarine_synthetic_posthoc_unmixed_dpo/activation_difference_lens" \
        "${ADL_BASE}/military_submarine_synthetic_posthoc_unmixed_fd/activation_difference_lens" \
        "${ADL_BASE}/military_submarine_synthetic_posthoc_unmixed_sdf/activation_difference_lens" \
        "${ADL_BASE}/military_submarine_synthetic_incorp_dpo/activation_difference_lens" \
    --names "posthoc-unmixed-dpo" "posthoc-unmixed-fd" "posthoc-unmixed-sdf" "integrated-dpo" \
    --organism-config configs/organism/milsub.yaml \
    --model-id allenai/OLMo-2-0425-1B-DPO \
    --dataset tulu-3-sft-olmo-2-mixture \
    --layers 7 14 15 \
    --patchscope-grader openai_gpt-5-mini \
    --ll-variant "$LL_VARIANT" \
    --output "results/synth_milsub_relevance${LL_SUFFIX}.csv" \
    --save-labels "results/synth_milsub_labels${LL_SUFFIX}.json" \
    --save-llm-log "results/synth_milsub_llm_log${LL_SUFFIX}.json" \
    --grader-model google/gemini-3-flash-preview


