#!/usr/bin/env bash
# Usage: ADL_BASE=<diffing_results/BASE> \
#          bash scripts/cumprobs/run_relevance.sh <family> \
#          <diff|ft|base|jlens|jlens_ft|jlens_base> [cohorts]
#
# $ADL_BASE is required: it selects the diffing_results/<base> tree to read.
# That base is recorded alongside the outputs by mo_relevance.py.
#
# <family> is a quirk_family_id from the model registry
# (e.g. cake_bake, italian_food, military_submarine, military_submarine_synthetic).
# The list of model variants is pulled dynamically from the registry, so adding
# a new variant there is enough to have it included here.
#
# [cohorts] selects registry cohorts (comma-separated, or "all"); it defaults to
# $MO_COHORTS and then to "core". A non-core run tags its output filenames so it
# cannot overwrite the core run's CSVs.
#
# jlens* modes read Jacobian-lens caches (jacobian_lens_pos_*.pt) — produced
# by the ADL pipeline (diffing.method.jacobian_lens.cache=true) or added to
# existing result dirs by scripts/cumprobs/backfill_jacobian_lens.py.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

# shellcheck source=scripts/cohort_lib.sh
source "${SCRIPT_DIR}/../cohort_lib.sh"
ADL_BASE="${ADL_BASE:-}"
REGISTRY="${MO_REGISTRY:-${PROJECT_DIR}/model_registry.json}"

FAMILY="${1:-}"
MODE="${2:-}"
COHORTS="${3:-${MO_COHORTS:-$MO_DEFAULT_COHORT}}"

if [[ -z "$ADL_BASE" ]]; then
    echo "\$ADL_BASE is required (no default): point it at the diffing_results/<base> tree." >&2
    exit 2
fi
if [[ -z "$FAMILY" ]]; then
    echo "Usage: $0 <family> <diff|ft|base|jlens|jlens_ft|jlens_base> [cohorts]" >&2
    exit 2
fi
# Mode -> (lens, variant, output-file suffix). The legacy logit_lens/diff combo
# keeps the empty suffix so existing artifact names are preserved.
case "$MODE" in
    diff)      LENS="logit_lens"; LL_VARIANT="diff";           LL_SUFFIX="" ;;
    ft|base)   LENS="logit_lens"; LL_VARIANT="$MODE";          LL_SUFFIX="_${MODE}" ;;
    jlens)     LENS="jlens";      LL_VARIANT="diff";           LL_SUFFIX="_jlens" ;;
    jlens_ft|jlens_base)
               LENS="jlens";      LL_VARIANT="${MODE#jlens_}"; LL_SUFFIX="_${MODE}" ;;
    *) echo "Usage: $0 <family> <diff|ft|base|jlens|jlens_ft|jlens_base> [cohorts]" >&2; exit 2 ;;
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

mo_validate_cohorts "$REGISTRY" "$COHORTS"

# Pull variant keys for this family, ordered by plot_order, restricted to
# the selected cohorts.
mapfile -t MODEL_KEYS < <(mo_registry_variants "$REGISTRY" "$FAMILY" "$COHORTS")

# Keep a non-core sweep from overwriting the core sweep's CSVs: the output
# filenames are built from the family alone.
OUT_PREFIX+="$(mo_cohort_tree_suffix "$COHORTS")"

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
    --adl-base "$ADL_BASE" \
    --names "${NAMES[@]}" \
    --organism-config "configs/organism/${ORGANISM}.yaml" \
    --model-id allenai/OLMo-2-0425-1B-DPO \
    --dataset tulu-3-sft-olmo-2-mixture \
    --layers 7 14 15 \
    --patchscope-grader openai_gpt-5-mini \
    --ll-variant "$LL_VARIANT" \
    --lens "$LENS" \
    --output "results/${OUT_PREFIX}_relevance${LL_SUFFIX}.csv" \
    --save-labels "results/${OUT_PREFIX}_labels${LL_SUFFIX}.json" \
    --save-llm-log "results/${OUT_PREFIX}_llm_log${LL_SUFFIX}.json" \
    --grader-model google/gemini-3-flash-preview
