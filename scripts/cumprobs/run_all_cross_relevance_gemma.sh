#!/usr/bin/env bash
# Run each MO family's ADL results against every organism config (cross-testing).
#
# --adl-base is required: it selects the diffing_results/<base> tree to read.
# That base is recorded alongside the outputs by mo_relevance.py.
#
# Model variants are discovered dynamically from the registry pointed to
# by $MO_REGISTRY (filtered by quirk_family_id, sorted by plot_order).
# Defaults to ${PROJECT_DIR}/model_registry.json.
#
# Usage:
#   bash scripts/cumprobs/run_all_cross_relevance_gemma.sh <diff|ft|base> [results-dir-name] [--adl-base <path>] [--dry-run]
#   bash scripts/cumprobs/run_all_cross_relevance_gemma.sh diff --adl-base <path>
#   bash scripts/cumprobs/run_all_cross_relevance_gemma.sh ft \
#       --adl-base /workspace/model-organisms/diffing_results/gemma3_1B_sibling
#   bash scripts/cumprobs/run_all_cross_relevance_gemma.sh diff --dry-run
#
# <results-dir-name> is the subdirectory under $CUMPROBS_ROOT where outputs are
# written. It is optional and defaults to the ADL base's directory name
# (e.g. --adl-base .../gemma3_1B_ancestor -> $CUMPROBS_ROOT/gemma3_1B_ancestor/).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
ADL_BASE=""
REGISTRY="${MO_REGISTRY:-${PROJECT_DIR}/model_registry.json}"
# Outputs live beside the ADL results they derive from rather than in the checkout.
CUMPROBS_ROOT="${CUMPROBS_ROOT:-/workspace/model-organisms/cumprobs}"

# shellcheck source=scripts/cumprobs/cohort_lib.sh
source "${SCRIPT_DIR}/cohort_lib.sh"
COHORTS="${MO_COHORTS:-$MO_DEFAULT_COHORT}"

usage() {
    echo "Usage: $0 <diff|ft|base> --adl-base <path> [results-dir-name] [--cohort <list>] [--dry-run]" >&2
    echo "  --adl-base is required; it selects the ADL results to read." >&2
    echo "  results-dir-name defaults to the ADL base's directory name." >&2
    mo_usage_cohort_line
    echo "  Output root: \$CUMPROBS_ROOT (${CUMPROBS_ROOT})" >&2
    exit 2
}

LL_VARIANT=""
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
        diff|ft|base) LL_VARIANT="$1"; shift ;;
        -*) usage ;;
        *)
            if [[ -z "$RESULTS_DIR_NAME" ]]; then
                RESULTS_DIR_NAME="$1"
            else
                usage
            fi
            shift ;;
    esac
done

if [[ -z "$LL_VARIANT" ]]; then
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

# Default the output directory to the ADL tree's own name so the two stay aligned.
if [[ -z "$RESULTS_DIR_NAME" ]]; then
    RESULTS_DIR_NAME="$(basename "$ADL_BASE")$(cohort_tree_suffix "$COHORTS")"
fi

RESULTS_BASE="${CUMPROBS_ROOT}/${RESULTS_DIR_NAME}"
# One token-label cache per judge, shared across MO families: a label depends
# only on (token, description, grader model, permutations), so reusing it keeps
# a token's label identical across families instead of re-graded per call.
LABEL_CACHE_DIR="${RESULTS_BASE}/labels"

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
# Gemma directory naming: <family>_<variant_suffix>, e.g.
#   italian_food_gemma_integrated_dpo
#   military_submarine_gemma_posthoc_mixed_dpo
#   military_submarine_synthetic_gemma_posthoc_unmixed_sdf
# Output prefixes drop the "_gemma" so the produced relevance CSVs slot into
# the plotter's existing FAMILY_HOME_JUDGE map (italian_food / milsub /
# synth_milsub) without further config.
# ---------------------------------------------------------------------------

MO_FAMILIES=(
    italian_food_gemma
    military_submarine_gemma
    military_submarine_synthetic_gemma
)

# For the 'base' LL variant, the base model is shared across every MO
# family/variant, so the LL output is identical across the sweep. Run once
# per organism using a single MO family + variant.
if [[ "$LL_VARIANT" == "base" ]]; then
    MO_FAMILIES=(italian_food_gemma)
fi

family_home_organism() {
    case "$1" in
        italian_food_gemma)                 echo "italian_food" ;;
        military_submarine_gemma)           echo "milsub" ;;
        military_submarine_synthetic_gemma) echo "milsub" ;;
        *) echo "" ;;
    esac
}

family_out_prefix() {
    case "$1" in
        italian_food_gemma)                 echo "italian_food" ;;
        military_submarine_gemma)           echo "milsub" ;;
        military_submarine_synthetic_gemma) echo "synth_milsub" ;;
        *) echo "" ;;
    esac
}

# Family used to look up variant suffixes (and their plot_order) in the
# registry. Identity for Gemma — the registry uses the same family ids.
family_registry_id() {
    echo "$1"
}

# Judges to cross-test against (unique homes). These keys name the output
# directories and must match FAMILY_HOME_JUDGE in plot_cumprobs_raffgraph.py,
# so judge_config maps them to config filenames rather than using them directly.
ORGANISM_CONFIGS=(cake_bake italian_food milsub)

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

# MODEL_ID is used by ADLExplorer for the tokenizer; Gemma 3 1B variants
# share a tokenizer with the IT base, so this works for both PT and IT
# fine-tunes.
MODEL_ID="google/gemma-3-1b-it"
# Dataset subdirectory name as it appears on disk inside each layer dir.
DATASET="tulu-3-sft-olmo-2-mixture"
# Absolute layer indices to analyse, matching preprocessing.layers in configs/lasr.yaml.
# A layer with no ADL directory contributes nothing rather than failing, so keep
# this in sync with the layers actually collected.
LAYERS="12 24 25"
# Positions to classify: POS_MIN..POS_MAX in plot_cumprobs_raffgraph.py, the
# range the plots cover. ADL writes more, which would only cost grader tokens.
POSITIONS="$(seq -s' ' -3 31)"
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
        registry_variants "$REGISTRY" "$registry_fam" "$COHORTS" \
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
            --positions $POSITIONS
            --patchscope-grader "$PATCHSCOPE_GRADER"
            --ll-variant "$LL_VARIANT"
            --output "${out_dir}/relevance${LL_SUFFIX}.csv"
            --save-labels "${out_dir}/labels${LL_SUFFIX}.json"
            --save-llm-log "${out_dir}/llm_log${LL_SUFFIX}.json"
            --grader-model "$GRADER_MODEL"
            --label-cache "${LABEL_CACHE_DIR}/${organism}${LL_SUFFIX}.json"
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
