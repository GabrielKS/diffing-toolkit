#!/usr/bin/env bash
# Run each MO family's ADL results against every organism config (cross-testing).
#
# Model variants are discovered dynamically from the registry at
#   /workspace/gks/model-organisms-for-real/config/model_registry.json
# (sorted by plot_order), matching the layout used by run_relevance.sh.
#
# Usage:
#   bash scripts/cumprobs/run_all_cross_relevance.sh [diff|ft|base] [--dry-run]
#   bash scripts/cumprobs/run_all_cross_relevance.sh             # diff variant
#   bash scripts/cumprobs/run_all_cross_relevance.sh ft           # ft variant
#   bash scripts/cumprobs/run_all_cross_relevance.sh ft --dry-run # print only
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
ADL_BASE="/workspace/model-organisms/diffing_results/olmo2_1B"
REGISTRY="/workspace/gks/model-organisms-for-real/config/model_registry.json"
RESULTS_BASE="results/cross_relevance"

LL_VARIANT=""
DRY_RUN=false
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=true ;;
        diff|ft|base) LL_VARIANT="$arg" ;;
        *) echo "Usage: $0 [diff|ft|base] [--dry-run]" >&2; exit 2 ;;
    esac
done

if [[ -z "$LL_VARIANT" ]]; then
    echo "Usage: $0 [diff|ft|base] [--dry-run]" >&2
    exit 2
fi

case "$LL_VARIANT" in
    diff) LL_SUFFIX="" ;;
    ft|base) LL_SUFFIX="_${LL_VARIANT}" ;;
esac

if [[ ! -f "$REGISTRY" ]]; then
    echo "Registry not found: $REGISTRY" >&2
    exit 1
fi

cd "$PROJECT_DIR"

# ---------------------------------------------------------------------------
# Families (MOs) and their home organism config / output prefix.
# Both military_submarine families share milsub.yaml.
#
# cake_bake_seedrep{1,2} are seed-replicate runs that share cake_bake's
# variant structure but live under their own directory prefix in ADL_BASE.
# They aren't in the registry, so we look up variant suffixes via the
# registry_family_id and discover dirs as ${family}_${suffix} on disk.
# ---------------------------------------------------------------------------

MO_FAMILIES=(
    cake_bake
    cake_bake_seedrep1
    cake_bake_seedrep2
    italian_food
    military_submarine
    military_submarine_synthetic
)

family_home_organism() {
    case "$1" in
        cake_bake|cake_bake_seedrep1|cake_bake_seedrep2) echo "cake_bake" ;;
        italian_food)                                    echo "italian_food" ;;
        military_submarine)                              echo "milsub" ;;
        military_submarine_synthetic)                    echo "milsub" ;;
        *) echo "" ;;
    esac
}

family_out_prefix() {
    case "$1" in
        cake_bake)                    echo "cake_bake" ;;
        cake_bake_seedrep1)           echo "cake_bake_seedrep1" ;;
        cake_bake_seedrep2)           echo "cake_bake_seedrep2" ;;
        italian_food)                 echo "italian_food" ;;
        military_submarine)           echo "milsub" ;;
        military_submarine_synthetic) echo "synth_milsub" ;;
        *) echo "" ;;
    esac
}

# Family used to look up variant suffixes (and their plot_order) in the
# registry. Seed replicates reuse cake_bake's variants.
family_registry_id() {
    case "$1" in
        cake_bake_seedrep1|cake_bake_seedrep2) echo "cake_bake" ;;
        *) echo "$1" ;;
    esac
}

# Organism configs to cross-test against (unique homes).
ORGANISM_CONFIGS=(cake_bake italian_food milsub)

# ---------------------------------------------------------------------------
# Shared parameters
# ---------------------------------------------------------------------------

MODEL_ID="allenai/OLMo-2-0425-1B-DPO"
DATASET="tulu-3-sft-olmo-2-mixture"
LAYERS="7 14 15"
PATCHSCOPE_GRADER="openai_gpt-5-mini"
GRADER_MODEL="google/gemini-3-flash-preview"

# ---------------------------------------------------------------------------
# Run all combinations
# ---------------------------------------------------------------------------

run_count=0
fail_count=0

for mo in "${MO_FAMILIES[@]}"; do
    out_prefix="$(family_out_prefix "$mo")"
    home_organism="$(family_home_organism "$mo")"
    registry_fam="$(family_registry_id "$mo")"

    # Pull variant suffixes from the registry, ordered by plot_order.
    # Seed-replicate families reuse the base family's variant set.
    mapfile -t VARIANT_SUFFIXES < <(
        jq -r --arg fam "$registry_fam" '
            .models
            | to_entries
            | map(select(.value.quirk_family_id == $fam))
            | sort_by(.value.plot_order)
            | .[].key
            | sub("^" + $fam + "_"; "")
        ' "$REGISTRY"
    )

    if [[ ${#VARIANT_SUFFIXES[@]} -eq 0 ]]; then
        echo "warn: no variants in registry for family $registry_fam, skipping $mo" >&2
        continue
    fi

    # Build --adl-paths + --names, skipping suffixes with missing ADL dirs.
    adl_paths=()
    variant_names=()
    for suffix in "${VARIANT_SUFFIXES[@]}"; do
        key="${mo}_${suffix}"
        path="${ADL_BASE}/${key}/activation_difference_lens"
        if [[ ! -d "$path" ]]; then
            echo "warn: skipping $key (missing $path)" >&2
            continue
        fi
        name="${suffix//_/-}"
        adl_paths+=("$path")
        variant_names+=("$name")
    done

    if [[ ${#adl_paths[@]} -eq 0 ]]; then
        echo "warn: no existing ADL result dirs for family $mo, skipping" >&2
        continue
    fi

    for organism in "${ORGANISM_CONFIGS[@]}"; do
        config_path="configs/organism/${organism}.yaml"

        # Naming: mo_<family>__judge_<organism>. The home-judge case is
        # self-evident from equality (mo_X__judge_X); no special suffix.
        combo_name="mo_${out_prefix}__judge_${organism}"
        out_dir="${RESULTS_BASE}/${combo_name}"

        # Human-readable title for plots.
        pretty_mo="${out_prefix//_/ }"
        pretty_organism="${organism//_/ }"
        if [[ "$organism" == "$home_organism" ]]; then
            plot_title="${pretty_mo^} (self)"
        else
            plot_title="${pretty_mo^} on ${pretty_organism^}"
        fi

        echo "=== ${mo} x ${organism} -> ${combo_name} ==="

        # --- relevance classification ---
        relevance_cmd=(
            uv run python scripts/cumprobs/mo_relevance.py
            --adl-paths "${adl_paths[@]}"
            --names "${variant_names[@]}"
            --organism-config "$config_path"
            --model-id "$MODEL_ID"
            --dataset "$DATASET"
            --layers $LAYERS
            --patchscope-grader "$PATCHSCOPE_GRADER"
            --ll-variant "$LL_VARIANT"
            --output "${out_dir}/relevance${LL_SUFFIX}.csv"
            --save-labels "${out_dir}/labels${LL_SUFFIX}.json"
            --save-llm-log "${out_dir}/llm_log${LL_SUFFIX}.json"
            --grader-model "$GRADER_MODEL"
        )

        # --- plot generation ---
        plot_cmd=(
            uv run python scripts/cumprobs/plot_mo_relevance.py
            "${out_dir}/relevance${LL_SUFFIX}.csv"
            -o "${out_dir}"
            --title "$plot_title"
            --ll-positions all
            --ps-positions all
            --ll-variant "$LL_VARIANT"
        )

        if $DRY_RUN; then
            echo "  ${relevance_cmd[*]}"
            echo "  ${plot_cmd[*]}"
            echo
        else
            if "${relevance_cmd[@]}"; then
                run_count=$((run_count + 1))
                if ! "${plot_cmd[@]}"; then
                    echo "  PLOT FAILED: ${combo_name}"
                fi
            else
                echo "  FAILED: ${combo_name}"
                fail_count=$((fail_count + 1))
            fi
        fi
    done
done

if ! $DRY_RUN; then
    echo
    echo "Done. ${run_count} succeeded, ${fail_count} failed."
fi
