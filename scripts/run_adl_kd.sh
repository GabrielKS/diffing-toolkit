#!/usr/bin/env bash
# Emit (or run) the ADL diffing commands for a registry cohort: the KD students
# by default, or e.g. the prompted organisms with --cohort prompted.
#
# Every command is derived from the registry, so the set of runs, the organism
# variant names and the on-disk result directories cannot drift from what the
# cumprobs drivers later look for. The invariant this script maintains is:
#
#     <results dir name> == <registry key> == <quirk_family_id>_<variant_id>
#
# which is what run_all_cross_relevance*.sh globs for. ADL names its results
# directory `<organism.name>_<variant>` (it ignores `diffing.results_dir`), and
# the Gemma organism configs are named `italian_food` / `military_submarine`
# while their registry families carry the historical `_gemma` suffix — so
# `organism.name` is pinned to the family explicitly rather than left to the
# config.
#
# Usage:
#   bash scripts/run_adl_kd.sh                 # print the commands
#   bash scripts/run_adl_kd.sh --execute       # run them sequentially
#   bash scripts/run_adl_kd.sh --family italian_food_gemma
#   bash scripts/run_adl_kd.sh --cohort prompted
#
# Prerequisites: $MO_REGISTRY must point at the model registry (the default,
# ./model_registry.json, does not exist in this checkout — the registry lives
# in the parent repo at ../config/model_registry.json).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# shellcheck source=scripts/cohort_lib.sh
source "${SCRIPT_DIR}/cohort_lib.sh"
REGISTRY="${MO_REGISTRY:-${PROJECT_DIR}/model_registry.json}"
DIFFING_RESULTS="${DIFFING_RESULTS:-/workspace/model-organisms/diffing_results}"
LOG_DIR="${LOG_DIR:-${PROJECT_DIR}/logs}"
COHORT="${MO_COHORTS:-kd}"   # comma-separated registry cohorts, or "all"

EXECUTE=false
ONLY_FAMILY=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --execute) EXECUTE=true; shift ;;
        --family)
            [[ $# -ge 2 ]] || { echo "--family needs a value" >&2; exit 2; }
            ONLY_FAMILY="$2"; shift 2 ;;
        --cohort)
            [[ $# -ge 2 ]] || { echo "--cohort needs a value" >&2; exit 2; }
            COHORT="$2"; shift 2 ;;
        *) echo "Usage: $0 [--execute] [--family <id>] [--cohort <list>]" >&2; exit 2 ;;
    esac
done

if [[ ! -f "$REGISTRY" ]]; then
    echo "Registry not found: $REGISTRY (set \$MO_REGISTRY)" >&2
    exit 1
fi

mo_validate_cohorts "$REGISTRY" "$COHORT"

# Registry family -> the organism config Hydra should load, and the diffing
# base to run it against. The Gemma students are diffed against the ancestor
# (google/gemma-3-1b-it) rather than the sibling vanilla-DPO checkpoint they
# were initialised from, so their diff also carries the vanilla-DPO delta. The
# prompted organisms are diffed against the same ancestors (OLMo: the SFT
# checkpoint), by design.
family_organism() {
    case "$1" in
        cake_bake|cake_bake_gemma)                   echo "cake_bake" ;;
        italian_food|italian_food_gemma)             echo "italian_food" ;;
        military_submarine|military_submarine_gemma) echo "military_submarine" ;;
        *) echo "" ;;
    esac
}

family_base_model() {
    case "$1" in
        cake_bake|italian_food|military_submarine)               echo "olmo2_1B_sft" ;;
        cake_bake_gemma|italian_food_gemma|military_submarine_gemma) echo "gemma3_1B_ancestor" ;;
        *) echo "" ;;
    esac
}

# The families that have any variant in the cohort, so a cohort's runs are
# enumerated from the registry alone.
mapfile -t FAMILIES < <(mo_registry_families "$REGISTRY" "$COHORT")
if [[ -n "$ONLY_FAMILY" ]]; then
    FAMILIES=("$ONLY_FAMILY")
fi

mkdir -p "$LOG_DIR"
count=0

for family in "${FAMILIES[@]}"; do
    organism="$(family_organism "$family")"
    base_model="$(family_base_model "$family")"
    if [[ -z "$organism" || -z "$base_model" ]]; then
        echo "unknown family: $family" >&2
        exit 1
    fi

    mapfile -t KEYS < <(mo_registry_variants "$REGISTRY" "$family" "$COHORT")
    if [[ ${#KEYS[@]} -eq 0 ]]; then
        echo "warn: no ${COHORT} variants for family ${family}" >&2
        continue
    fi

    for key in "${KEYS[@]}"; do
        variant="${key#${family}_}"
        results_dir="${DIFFING_RESULTS}/${base_model}/${key}"
        log="${LOG_DIR}/adl_${key}.log"

        cmd=(
            uv run python main.py --config-name=lasr
            "model=${base_model}"
            "organism=${organism}"
            "organism_variant=${variant}"
            pipeline.mode=diffing
            "organism.name=${family}"
        )

        count=$((count + 1))
        if $EXECUTE; then
            echo "=== [${count}] ${key} -> ${results_dir}"
            ( cd "$PROJECT_DIR" && "${cmd[@]}" ) &> "$log" || {
                echo "  FAILED: ${key} (see ${log})" >&2
            }
        else
            printf 'cd %q && %s &> %q\n' "$PROJECT_DIR" "${cmd[*]}" "$log"
        fi
    done
done

if ! $EXECUTE; then
    echo >&2
    echo "${count} commands. Re-run with --execute to run them sequentially." >&2
fi
