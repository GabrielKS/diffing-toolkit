#!/usr/bin/env bash
# Usage: bash scripts/cumprobs/run_relevance.sh <family> [diff|ft|base]
#
# <family> is a quirk_family_id from the model registry
# (e.g. cake_bake, italian_food, military_submarine, military_submarine_synthetic).
# The list of model variants is pulled dynamically from the registry, so adding
# a new variant there is enough to have it included here.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
ADL_BASE="${ADL_BASE:-/workspace/model-organisms/diffing_results/olmo2_1B}"
REGISTRY="${MO_REGISTRY:-${PROJECT_DIR}/model_registry.json}"

FAMILY="${1:-}"
LL_VARIANT="${2:-}"

if [[ -z "$FAMILY" ]]; then
    echo "Usage: $0 <family> [diff|ft|base]" >&2
    exit 2
fi
case "$LL_VARIANT" in
    diff) LL_SUFFIX="" ;;
    ft|base) LL_SUFFIX="_${LL_VARIANT}" ;;
    *) echo "Usage: $0 <family> [diff|ft|base]" >&2; exit 2 ;;
esac

# Family -> organism config / output-file prefix.
# Most families map 1:1, but both military_submarine variants share milsub.yaml.
case "$FAMILY" in
    cake_bake)                    ORGANISM="cake_bake";    OUT_PREFIX="cake_bake" ;;
    italian_food)                 ORGANISM="italian_food"; OUT_PREFIX="italian_food" ;;
    military_submarine)           ORGANISM="milsub";       OUT_PREFIX="milsub" ;;
    military_submarine_synthetic) ORGANISM="milsub";       OUT_PREFIX="synth_milsub" ;;
    *) echo "Unknown family: $FAMILY" >&2; exit 2 ;;
esac

if [[ ! -f "$REGISTRY" ]]; then
    echo "Registry not found: $REGISTRY" >&2
    exit 1
fi

# Pull variant keys for this family, ordered by plot_order.
mapfile -t MODEL_KEYS < <(
    jq -r --arg fam "$FAMILY" '
        .models
        | to_entries
        | map(select(.value.quirk_family_id == $fam))
        | sort_by(.value.plot_order)
        | .[].key
    ' "$REGISTRY"
)

if [[ ${#MODEL_KEYS[@]} -eq 0 ]]; then
    echo "No models found in registry for family: $FAMILY" >&2
    exit 1
fi

ADL_PATHS=()
NAMES=()
for key in "${MODEL_KEYS[@]}"; do
    path="${ADL_BASE}/${key}/activation_difference_lens"
    if [[ ! -d "$path" ]]; then
        echo "warn: skipping $key (missing $path)" >&2
        continue
    fi
    # Strip the "<family>_" prefix from the registry key and hyphenate for a
    # compact display name (e.g. cake_bake_posthoc_unmixed_dpo -> posthoc-unmixed-dpo).
    name="${key#${FAMILY}_}"
    name="${name//_/-}"
    ADL_PATHS+=("$path")
    NAMES+=("$name")
done

if [[ ${#ADL_PATHS[@]} -eq 0 ]]; then
    echo "No existing ADL result dirs under $ADL_BASE for family: $FAMILY" >&2
    exit 1
fi

cd "$PROJECT_DIR"

uv run python scripts/cumprobs/mo_relevance.py \
    --adl-paths "${ADL_PATHS[@]}" \
    --names "${NAMES[@]}" \
    --organism-config "configs/organism/${ORGANISM}.yaml" \
    --model-id allenai/OLMo-2-0425-1B-DPO \
    --dataset tulu-3-sft-olmo-2-mixture \
    --layers 7 14 15 \
    --patchscope-grader openai_gpt-5-mini \
    --ll-variant "$LL_VARIANT" \
    --output "results/${OUT_PREFIX}_relevance${LL_SUFFIX}.csv" \
    --save-labels "results/${OUT_PREFIX}_labels${LL_SUFFIX}.json" \
    --save-llm-log "results/${OUT_PREFIX}_llm_log${LL_SUFFIX}.json" \
    --grader-model google/gemini-3-flash-preview
