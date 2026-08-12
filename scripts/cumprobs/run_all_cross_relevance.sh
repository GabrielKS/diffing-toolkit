#!/usr/bin/env bash
# Run each MO family's ADL results against every organism config (cross-testing).
#
# Model variants are discovered dynamically from the registry pointed to
# by $MO_REGISTRY (sorted by plot_order), matching the layout used by
# run_relevance.sh. Defaults to ${PROJECT_DIR}/model_registry.json.
#
# --adl-base is required: it selects the diffing_results/<base> tree to read.
# That base is recorded alongside the outputs by mo_relevance.py.
#
# Usage:
#   bash scripts/cumprobs/run_all_cross_relevance.sh <diff|ft|base|jlens_diff|jlens_ft|jlens_base> \
#       [results-dir-name] [--adl-base <path>] [--dry-run]
#   bash scripts/cumprobs/run_all_cross_relevance.sh diff --adl-base <path>
#   bash scripts/cumprobs/run_all_cross_relevance.sh jlens_diff --adl-base <path>
#   bash scripts/cumprobs/run_all_cross_relevance.sh ft \
#       --adl-base /workspace/model-organisms/diffing_results/olmo2_1B
#   bash scripts/cumprobs/run_all_cross_relevance.sh ft --dry-run
#
# jlens* modes read Jacobian-lens caches (jacobian_lens_pos_*.pt) — produced
# by the ADL pipeline (diffing.method.jacobian_lens.cache=true) or added to
# existing result dirs by scripts/cumprobs/backfill_jacobian_lens.py.
#
# <results-dir-name> is the subdirectory under $CUMPROBS_ROOT where outputs are
# written. It is optional and defaults to the ADL base's directory name
# (e.g. --adl-base .../olmo2_1B -> $CUMPROBS_ROOT/olmo2_1B/...).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

# shellcheck source=scripts/cohort_lib.sh
source "${SCRIPT_DIR}/../cohort_lib.sh"
ADL_BASE=""
REGISTRY="${MO_REGISTRY:-${PROJECT_DIR}/model_registry.json}"
# Outputs live beside the ADL results they derive from rather than in the checkout.
CUMPROBS_ROOT="${CUMPROBS_ROOT:-/workspace/model-organisms/cumprobs}"

# Registry cohorts to sweep, comma-separated or "all". Every entry declares a
# `cohorts` list, so membership is explicit rather than inferred; "core" is the
# original MO families, which is what an invocation without --cohort always got.
COHORTS="${MO_COHORTS:-$MO_DEFAULT_COHORT}"

usage() {
    echo "Usage: $0 <${MO_LENS_MODES_USAGE}> --adl-base <path> [results-dir-name] [--cohort <list>] [--dry-run]" >&2
    echo "  --adl-base is required; it selects the ADL results to read." >&2
    echo "  results-dir-name defaults to the ADL base's directory name." >&2
    mo_usage_cohort_line
    echo "  Output root: \$CUMPROBS_ROOT (${CUMPROBS_ROOT})" >&2
    exit 2
}

MODE=""
RESULTS_DIR_NAME=""
DRY_RUN=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=true; shift ;;
        --adl-base)
            [[ $# -ge 2 ]] || usage
            ADL_BASE="$2"; shift 2 ;;
        --cohort)
            [[ $# -ge 2 ]] || usage
            COHORTS="$2"; shift 2 ;;
        -*) usage ;;
        *)
            if [[ -z "$MODE" ]] && mo_lens_mode "$1"; then
                MODE="$1"
            elif [[ -z "$RESULTS_DIR_NAME" ]]; then
                RESULTS_DIR_NAME="$1"
            else
                usage
            fi
            shift ;;
    esac
done

if [[ -z "$MODE" ]]; then
    usage
fi
if [[ -z "$ADL_BASE" ]]; then
    echo "--adl-base is required (no default)" >&2
    usage
fi
if [[ ! -d "$ADL_BASE" ]]; then
    echo "ADL base directory not found: $ADL_BASE" >&2
    exit 1
fi

if [[ -z "$RESULTS_DIR_NAME" ]]; then
    RESULTS_DIR_NAME="$(basename "$ADL_BASE")"
    # A non-core sweep needs its own tree: the per-combination output path
    # is built from the family and judge alone, so it would otherwise
    # overwrite the core run's relevance.csv in place.
    RESULTS_DIR_NAME+="$(mo_cohort_tree_suffix "$COHORTS")"
fi

RESULTS_BASE="${CUMPROBS_ROOT}/${RESULTS_DIR_NAME}"

# Sets LENS, LL_VARIANT and LL_SUFFIX; see mo_lens_mode in cohort_lib.sh.
mo_lens_mode "$MODE" || usage

if [[ ! -f "$REGISTRY" ]]; then
    echo "Registry not found: $REGISTRY" >&2
    exit 1
fi

mo_validate_cohorts "$REGISTRY" "$COHORTS"

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

# For the 'base' LL variant, the base model is shared across every MO
# family/variant, so the LL output is identical across the sweep. Run once
# per organism using a single MO family + variant.
if [[ "$LL_VARIANT" == "base" ]]; then
    MO_FAMILIES=(cake_bake)
fi

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

# Judge keys name the output directories and must match FAMILY_HOME_JUDGE in
# plot_cumprobs_raffgraph.py, so they are mapped to config files rather than
# used as filenames directly - there is no configs/organism/milsub.yaml.
judge_config() {
    case "$1" in
        cake_bake)    echo "configs/organism/cake_bake.yaml" ;;
        italian_food) echo "configs/organism/italian_food.yaml" ;;
        milsub)       echo "configs/organism/military_submarine.yaml" ;;
        *) echo "" ;;
    esac
}

# Fail before spending any grader tokens rather than partway through the sweep.
for organism in "${ORGANISM_CONFIGS[@]}"; do
    cfg="$(judge_config "$organism")"
    if [[ -z "$cfg" || ! -f "$cfg" ]]; then
        echo "judge '$organism' maps to missing config: ${cfg:-<unmapped>}" >&2
        exit 1
    fi
done

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

    # Pull variant suffixes from the registry, ordered by plot_order and
    # restricted to $COHORTS. Seed-replicate families reuse the base family's
    # variant set.
    mapfile -t VARIANT_SUFFIXES < <(
        mo_registry_variants "$REGISTRY" "$registry_fam" "$COHORTS" \
            | sed "s/^${registry_fam}_//"
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

    # 'base' LL variant: keep only the first (lowest plot_order) variant.
    if [[ "$LL_VARIANT" == "base" ]]; then
        adl_paths=("${adl_paths[0]}")
        variant_names=("${variant_names[0]}")
    fi

    for organism in "${ORGANISM_CONFIGS[@]}"; do
        config_path="$(judge_config "$organism")"

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
            --adl-base "$ADL_BASE"
            --names "${variant_names[@]}"
            --organism-config "$config_path"
            --model-id "$MODEL_ID"
            --dataset "$DATASET"
            --layers $LAYERS
            --patchscope-grader "$PATCHSCOPE_GRADER"
            --ll-variant "$LL_VARIANT"
            --lens "$LENS"
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
