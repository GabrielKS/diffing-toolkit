#!/usr/bin/env bash
# Quarantine the Gemma artifacts built from mislabeled Italian Food models by
# appending "_incorrect" to their directory names. Nothing is deleted.
#
# Background
# ----------
# Three `italian_food_gemma` entries in config/model_registry.json carried
# military-submarine `hf_model_id`s (introduced in 2ad9901, fixed since):
#
#   posthoc_unmixed_fd  -> gemma-3-1b-military-submarine-posthoc-fd-unmixed
#   posthoc_mixed_fd    -> gemma-3-1b-military-submarine-posthoc-fd-mixed
#   posthoc_mixed_sdf   -> gemma-3-1b-military-submarine-posthoc-sdf-mixed-lr-3.5e-5
#
# setup_adl_for_steering.py downloads from those ids into a directory named
# after the *registry key*, so the submarine weights landed under Italian Food
# names. Confirmed by identical model.safetensors SHA256 (and identical
# trainer_state.json losses) against the submarine snapshots. Everything
# downstream - ADL, steering, unsteered generations - inherited the mislabel.
#
# Renaming rather than deleting keeps the evidence and leaves the correct paths
# free, so a re-download and ADL re-run write to clean directories.
#
# Usage:
#   bash scripts/cumprobs/quarantine_mislabeled_italian_food.sh [--dry-run] [--include-layer23]
#
# --include-layer23 additionally quarantines the stale layer_23 ADL dirs. These
# are an UNRELATED issue: pre-`get_layer_indices`-fix artifacts left behind when
# runs were redone in place. They are no longer read (the driver pins 12/24/25),
# so this is housekeeping, not a correctness fix.
set -euo pipefail

SUFFIX="_incorrect"
DRY_RUN=false
INCLUDE_LAYER23=false

usage() {
    echo "Usage: $0 [--dry-run] [--include-layer23]" >&2
    exit 2
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=true; shift ;;
        --include-layer23) INCLUDE_LAYER23=true; shift ;;
        *) usage ;;
    esac
done

MODELS_DIR="/workspace/models/gemma3_1B"
RESULTS_DIR="/workspace/model-organisms/diffing_results"

# The three mislabeled variants.
VARIANTS=(
    italian_food_gemma_posthoc_unmixed_fd
    italian_food_gemma_posthoc_mixed_fd
    italian_food_gemma_posthoc_mixed_sdf
)

# Build the target list explicitly rather than by glob - a typo in a glob here
# would sweep in correct data.
TARGETS=()
for v in "${VARIANTS[@]}"; do
    TARGETS+=("${MODELS_DIR}/${v}")                      # local snapshot (submarine weights)
    TARGETS+=("${RESULTS_DIR}/gemma3_1B_ancestor/${v}")  # ADL + steering, ancestor base
    TARGETS+=("${RESULTS_DIR}/gemma3_1B_sibling/${v}")   # ADL + steering, sibling base
    TARGETS+=("${RESULTS_DIR}/unsteered/${v}")           # generations sampled from the model
done

if $INCLUDE_LAYER23; then
    # Only the two Gemma trees. layer_23 is a legitimate layer in the OLMo
    # trees, so this must never widen to the diffing_results root.
    while IFS= read -r d; do
        TARGETS+=("$d")
    done < <(find "${RESULTS_DIR}/gemma3_1B_ancestor" "${RESULTS_DIR}/gemma3_1B_sibling" \
                  -maxdepth 3 -type d -name layer_23 2>/dev/null | sort)
fi

renamed=0
skipped=0
blocked=0

for src in "${TARGETS[@]}"; do
    dst="${src}${SUFFIX}"

    if [[ ! -d "$src" ]]; then
        if [[ -d "$dst" ]]; then
            echo "skip    (already quarantined) ${src}"
        else
            echo "skip    (not found)           ${src}"
        fi
        skipped=$((skipped + 1))
        continue
    fi

    if [[ -e "$dst" ]]; then
        echo "BLOCKED (target exists)      ${dst}"
        blocked=$((blocked + 1))
        continue
    fi

    if $DRY_RUN; then
        echo "would rename ${src} -> ${dst}"
    else
        mv -- "$src" "$dst"
        echo "renamed ${src} -> ${dst}"
    fi
    renamed=$((renamed + 1))
done

echo
if $DRY_RUN; then
    echo "Dry run. ${renamed} would be renamed, ${skipped} skipped, ${blocked} blocked."
else
    echo "Done. ${renamed} renamed, ${skipped} skipped, ${blocked} blocked."
fi

if [[ $blocked -gt 0 ]]; then
    echo "Blocked entries were left untouched - resolve the existing target by hand." >&2
    exit 1
fi
