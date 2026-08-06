#!/usr/bin/env bash
# Run each MO family's ADL results against every organism config (cross-testing).
#
# The ADL source directory is required: pass --adl-base explicitly. It
# decides which diffing base the resulting numbers describe, and that is
# not recoverable from the output, so it is never defaulted.
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
# (e.g. the default --adl-base -> $CUMPROBS_ROOT/gemma3_1B_ancestor/...).
#
# NOTE: The Gemma ADL trees contain no patchscope_*.pt files, only logit-lens
# variants. This is fine - ADLExplorer discovers patchscope files by glob, so
# their absence yields empty patchscope dicts rather than an error.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
ADL_BASE=""
REGISTRY="${MO_REGISTRY:-${PROJECT_DIR}/model_registry.json}"
# Outputs live beside the ADL results they derive from, not inside the checkout:
# a run from a worktree would otherwise write into that worktree and vanish with
# it. A sibling of diffing_results/ rather than a child, because that tree's
# second level holds per-organism dirs and this is an aggregate over them.
CUMPROBS_ROOT="${CUMPROBS_ROOT:-/workspace/model-organisms/cumprobs}"

usage() {
    echo "Usage: $0 <diff|ft|base> --adl-base <path> [results-dir-name] [--dry-run]" >&2
    echo "  --adl-base is required; it selects the diffing base being described." >&2
    echo "  results-dir-name defaults to the ADL base's directory name." >&2
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

# Default the output directory to the ADL tree's own name, so the two stay
# aligned by construction instead of by the caller passing a matching label.
if [[ -z "$RESULTS_DIR_NAME" ]]; then
    RESULTS_DIR_NAME="$(basename "$ADL_BASE")"
fi

RESULTS_BASE="${CUMPROBS_ROOT}/${RESULTS_DIR_NAME}"
# One token-label cache per judge, shared across MO families. A label depends
# only on (token, description, grader model, permutations), so a token seen
# while scoring italian_food is reused when milsub is scored against the same
# judge - cheaper, and it removes the only source of cross-family label drift:
# each family is a separate mo_relevance call, and the majority vote over
# rotated orderings can otherwise land differently when chunk composition
# changes with the token set.
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
# so they are mapped to config files rather than used as filenames directly -
# there is no configs/organism/milsub.yaml.
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
# Absolute layer indices present in the Gemma ADL tree.
#
# Only 12/24/25 are current. Half the ADL dirs also hold a layer_23 written
# before the get_layer_indices revision fix; those runs were redone in place
# (layer_24 is newer than 12/23/25) and the stale layer_23 was never removed.
# Requesting it would grade tokens off pre-fix activations and emit a
# half-populated layer-23 figure, since the plotter draws one plot per layer
# over the union of layers present. A layer dir that does not exist globs to
# zero positions rather than raising, so this fails silently if reintroduced.
LAYERS="12 24 25"
# Positions to classify. POS_MIN..POS_MAX in plot_cumprobs_raffgraph.py - the
# only range that reaches a figure. ADL writes -3..127; classifying just this
# window cuts grader cost ~4x with no effect on the plots.
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
