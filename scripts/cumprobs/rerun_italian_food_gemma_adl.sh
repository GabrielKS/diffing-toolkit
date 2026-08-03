#!/usr/bin/env bash
# Re-download the three mislabeled Italian Food Gemma models and re-run ADL for
# them against the gemma3_1B_ancestor base.
#
# Runs against the MAIN checkout, not a worktree: only the main copy carries the
# uncommitted `auto_patch_scope.enabled: false`, and the method default is true,
# so a worktree run would fire the patchscope grader and write files the other
# eleven Gemma dirs do not have.
#
# Each download and each ADL run writes its own log under .claude/scratch/,
# matching the `lasr_*` convention run_kd_cross_relevance.py already looks for.
#
# Prerequisite: scripts/cumprobs/quarantine_mislabeled_italian_food.sh has moved
# the old snapshots and result dirs aside. This script refuses to start
# otherwise rather than writing into a directory holding submarine-derived data.
#
# Usage:
#   bash scripts/cumprobs/rerun_italian_food_gemma_adl.sh [--dry-run]
set -euo pipefail

REPO="/root/bootstrap/model-organisms-for-real/diffing-toolkit"
MODELS_DIR="/workspace/models/gemma3_1B"
RESULTS_DIR="/workspace/model-organisms/diffing_results/gemma3_1B_ancestor"
LOG_DIR="${REPO}/.claude/scratch"
STAMP="$(date +%Y%m%d_%H%M%S)"

DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true

# variant : hf repo : revision
JOBS=(
    "posthoc_unmixed_fd:gemma-3-1b-italian-food-posthoc-fd-unmixed:checkpoint-16"
    "posthoc_mixed_fd:gemma-3-1b-italian-food-posthoc-fd-mixed:checkpoint-35"
    "posthoc_mixed_sdf:gemma-3-1b-italian-food-posthoc-sdf-mixed-lr-5e-5:step-7"
)

cd "$REPO"
mkdir -p "$LOG_DIR"

# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------

# The upper-middle relative layer must be 0.96. At 0.94 get_layer_indices lands
# on layer 23 for Gemma 3 1B's 26 layers, and every other Gemma organism is
# built at 24.
if grep -q "0\.94" configs/lasr.yaml; then
    echo "Patching configs/lasr.yaml: 0.94 -> 0.96 (upper-middle layer)"
    if ! $DRY_RUN; then
        sed -i \
            's/\[0\.5, 0\.94, 1\.0\]/[0.5, 0.96, 1.0]/; s/^      - 0\.94$/      - 0.96/; s/^          layer: 0\.94$/          layer: 0.96/' \
            configs/lasr.yaml
    fi
else
    echo "configs/lasr.yaml already at 0.96"
fi

if ! $DRY_RUN && grep -q "0\.94" configs/lasr.yaml; then
    echo "configs/lasr.yaml still contains 0.94 after patching" >&2
    exit 1
fi

# Only the ADL output is checked. A pre-existing result dir means the old
# submarine-derived run is still in place, and `overwrite: false` would leave it
# there. Model dirs are handled per-variant below: present means downloaded.
for job in "${JOBS[@]}"; do
    variant="${job%%:*}"
    path="${RESULTS_DIR}/italian_food_gemma_${variant}"
    if [[ -e "$path" ]]; then
        echo "Refusing to start: $path already exists." >&2
        echo "Run quarantine_mislabeled_italian_food.sh first." >&2
        exit 1
    fi
done

# ---------------------------------------------------------------------------
# Step 2: download
# ---------------------------------------------------------------------------

echo
echo "=== Downloads ==="
for job in "${JOBS[@]}"; do
    IFS=: read -r variant repo revision <<< "$job"
    dest="${MODELS_DIR}/italian_food_gemma_${variant}"
    if [[ -f "${dest}/model.safetensors" ]]; then
        echo "  ${variant}: already downloaded, skipping"
        continue
    fi
    log="${LOG_DIR}/lasr_ifgemma_download_${variant}_${STAMP}.log"
    echo "  ${variant} <- ${repo}@${revision}"
    echo "    log: ${log}"
    $DRY_RUN && continue
    uv run hf download "model-organisms-for-real/${repo}" \
        --revision "$revision" \
        --local-dir "$dest" \
        > "$log" 2>&1
done

# ---------------------------------------------------------------------------
# Step 3: ADL
# ---------------------------------------------------------------------------

run_adl() {
    local variant="$1" gpu="$2"
    local log="${LOG_DIR}/lasr_ifgemma_adl_${variant}_${STAMP}.log"
    echo "  [gpu ${gpu}] ${variant}"
    echo "    log: ${log}"
    $DRY_RUN && return 0
    CUDA_VISIBLE_DEVICES="$gpu" uv run python main.py --config-name=lasr \
        organism=italian_food_gemma organism_variant="$variant" \
        model=gemma3_1B_ancestor pipeline.mode=diffing \
        > "$log" 2>&1
}

echo
echo "=== ADL (gpu 0: two runs, gpu 1: one run) ==="
rc0=0
rc1=0
( run_adl posthoc_unmixed_fd 0 && run_adl posthoc_mixed_fd 0 ) & pid0=$!
( run_adl posthoc_mixed_sdf 1 ) & pid1=$!
wait "$pid0" || rc0=$?
wait "$pid1" || rc1=$?

if $DRY_RUN; then
    echo
    echo "Dry run. Nothing executed."
    exit 0
fi

# ---------------------------------------------------------------------------
# Verify
# ---------------------------------------------------------------------------

echo
echo "=== Layers written (expect 12 24 25) ==="
fail=0
for job in "${JOBS[@]}"; do
    variant="${job%%:*}"
    d="${RESULTS_DIR}/italian_food_gemma_${variant}/activation_difference_lens"
    got="$(ls "$d" 2>/dev/null | grep '^layer_' | sed 's/layer_//' | sort -n | tr '\n' ' ' | sed 's/ $//')"
    printf "  %-20s %s\n" "$variant" "${got:-<none>}"
    [[ "$got" == "12 24 25" ]] || fail=1
done

echo
if [[ $rc0 -ne 0 || $rc1 -ne 0 ]]; then
    echo "FAILED: gpu0 rc=${rc0}, gpu1 rc=${rc1}. See logs in ${LOG_DIR}." >&2
    exit 1
fi
if [[ $fail -ne 0 ]]; then
    echo "FAILED: unexpected layer set. See logs in ${LOG_DIR}." >&2
    exit 1
fi
echo "Done. All three variants at layers 12/24/25."
echo "Next: bash scripts/cumprobs/run_all_cross_relevance_gemma.sh diff gemma_ancestor"
