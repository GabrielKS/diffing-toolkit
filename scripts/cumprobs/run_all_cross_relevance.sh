#!/usr/bin/env bash
# Run each MO's ADL results against every organism config (cross-testing).
#
# Usage:
#   bash scripts/cumprobs/run_all_cross_relevance.sh
#   bash scripts/cumprobs/run_all_cross_relevance.sh --dry-run   # print commands without running
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
ADL_BASE="/workspace/model-organisms/diffing_results/olmo2_1B"
RESULTS_BASE="results/cross_relevance"

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
fi

cd "$PROJECT_DIR"

# ---------------------------------------------------------------------------
# MO definitions: each MO has ADL subdirs and human-readable variant names
# ---------------------------------------------------------------------------

MO_NAMES=(cake_bake italian_food milsub examples)

# cake_bake
cake_bake_DIRS=(
    "cake_bake_dpo_b0.05_lr1e-4_e1_r16"
    "cake_bake_our-sdf-1000"
    "cake_bake_wide-minimal-edit"
    "cake_bake_wide-rewritten-rejected"
    
)
cake_bake_VARIANT_NAMES=("narrow-dpo" "sdf" "wide-dpo-minimal" "wide-dpo-augmented")

# italian_food
italian_food_DIRS=(
    "italian_food_narrow-sft-leveled-unmixed"
    "italian_food_narrow-sft-leveled-mixed"
    "italian_food_wide-dpo"
)
italian_food_VARIANT_NAMES=("sft-unmixed" "sft-mixed" "wide-dpo")

# milsub
milsub_DIRS=(
    "milsub_narrow-sft"
    "milsub_narrow-dpo-2"
    "milsub_wide-dpo"
)
milsub_VARIANT_NAMES=("sft" "narrow-dpo" "wide-dpo")

# examples
examples_DIRS=(
    "examples_narrow-sft-2-new"
    "examples_full-new"
)
examples_VARIANT_NAMES=("examples-narrow" "examples-wide")

# Reseeded OLMo DPO control (appended to every MO run)
CONTROL_DIR="random_olmo_other_olmo"
CONTROL_NAME="reseeded-olmo-control"

# ---------------------------------------------------------------------------
# Organism configs to test against
# ---------------------------------------------------------------------------

ORGANISM_CONFIGS=(cake_bake italian_food milsub examples)

# ---------------------------------------------------------------------------
# Shared parameters
# ---------------------------------------------------------------------------

MODEL_ID="allenai/OLMo-2-0425-1B-DPO"
DATASET="tulu-3-sft-olmo-2-mixture"
LAYERS="7 14 15"
PATCHSCOPE_GRADER="openai_gpt-5-mini"

# ---------------------------------------------------------------------------
# Run all combinations
# ---------------------------------------------------------------------------

run_count=0
fail_count=0

for mo in "${MO_NAMES[@]}"; do
    # Resolve MO-specific arrays via namerefs
    declare -n dirs="${mo}_DIRS"
    declare -n names="${mo}_VARIANT_NAMES"

    # Build --adl-paths arguments (MO variants + reseeded control)
    adl_paths=()
    variant_names=()
    for d in "${dirs[@]}"; do
        adl_paths+=("${ADL_BASE}/${d}/activation_difference_lens")
    done
    for n in "${names[@]}"; do
        variant_names+=("$n")
    done
    adl_paths+=("${ADL_BASE}/${CONTROL_DIR}/activation_difference_lens")
    variant_names+=("$CONTROL_NAME")

    for organism in "${ORGANISM_CONFIGS[@]}"; do
        config_path="configs/organism/${organism}.yaml"

        # Each combo gets its own directory
        if [[ "$mo" == "$organism" ]]; then
            combo_name="${mo}_self"
        else
            combo_name="${mo}_tested_on_${organism}"
        fi
        out_dir="${RESULTS_BASE}/${combo_name}"

        # Human-readable title for plots (e.g. "Milsub on Italian Food")
        pretty_mo="${mo//_/ }"
        pretty_organism="${organism//_/ }"
        if [[ "$mo" == "$organism" ]]; then
            plot_title="${pretty_mo^} (self)"
        else
            plot_title="${pretty_mo^} on ${pretty_organism^}"
        fi

        echo "=== ${mo} x ${organism} ==="

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
            --output "${out_dir}/relevance.csv"
            --save-labels "${out_dir}/labels.json"
            --save-llm-log "${out_dir}/llm_log.json"
            --grader-model google/gemini-3-flash-preview
        )

        # --- plot generation ---
        plot_cmd=(
            uv run python scripts/cumprobs/plot_mo_relevance.py
            "${out_dir}/relevance.csv"
            -o "${out_dir}"
            --title "$plot_title"
            --ll-positions all
            --ps-positions all
        )

        if $DRY_RUN; then
            echo "  ${relevance_cmd[*]}"
            echo "  ${plot_cmd[*]}"
            echo
        else
            if "${relevance_cmd[@]}"; then
                run_count=$((run_count + 1))
                # Generate plots only if relevance succeeded
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
