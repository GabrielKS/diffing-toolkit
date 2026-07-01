#!/usr/bin/env bash
# =====================================================================
# End-to-end MEAN-POOLED AO×QER experiment (military-submarine).
#
#   1. Build the QER milsub trigger pool (default 500 = full test set).
#   2. run_pooled_ao.py: compute the diff-of-means vector per act_key over the
#      context pool, inject it per trigger, sample num_passes generations.
#   3. Grade with the original QER judge -> QER for diff/orig/lora.
#
# Generations = n_triggers × num_passes × n_act_keys (default 500×3×3 = 4500).
# All local; nothing pushed to HF.
#
# Overridable: N_TRIGGERS (500), JUDGE_BATCH_SIZE (20), DRY_RUN, MAX_RESPONSES.
# Launcher overrides (reuse already-synced envs): AO_RUN, QER_RUN.
# =====================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

if [[ -f "${REPO_ROOT}/.env" ]]; then set -a; . "${REPO_ROOT}/.env"; set +a; fi

N_TRIGGERS="${N_TRIGGERS:-500}"
TRIGGER_SOURCE="${TRIGGER_SOURCE:-hf}"
JUDGE_BATCH_SIZE="${JUDGE_BATCH_SIZE:-20}"
AO_RUN="${AO_RUN:-uv run python}"
QER_RUN="${QER_RUN:-uv run --project ${REPO_ROOT}/external-qer python}"

ORGANISM="remote_military_submarine_synth"
VARIANT="post_hoc_unmixed_dpo"
# Default = IT-oracle ancestor diffing (base=gemma-3-1b-it + IT oracle,
# generation on the base backbone, MO patched in, diff = MO − IT).
MODEL="${MODEL:-gemma3_1B_repl}"
METHOD="${METHOD:-activation_oracle_qer_milsub_pooled_it}"
RUN_TAG="${RUN_TAG:-it}"
OUT_DIR="${REPO_ROOT}/ao_qer_experiments/ao_qer_milsub/results"
mkdir -p "$OUT_DIR"

echo "=== [1/3] Build trigger pool (n=${N_TRIGGERS}) ==="
$QER_RUN "${REPO_ROOT}/ao_qer_experiments/ao_qer_milsub/build_trigger_pool.py" \
    --n_triggers "$N_TRIGGERS" --source "$TRIGGER_SOURCE"

echo "=== [2/3] Mean-pooled AO generation ==="
POOLED_OUT="$($AO_RUN "${REPO_ROOT}/ao_qer_experiments/ao_qer_milsub/run_pooled_ao.py" \
  organism="$ORGANISM" organism_variant="$VARIANT" \
  model="$MODEL" diffing/method="$METHOD" \
  infrastructure=local wandb.enabled=false ${AO_EXTRA:-} 2>&1 | tee /dev/stderr | grep -oE 'POOLED_RESULTS_FILE=.*' | tail -1 | cut -d= -f2-)"

if [[ -z "${POOLED_OUT}" || ! -f "${POOLED_OUT}" ]]; then
  echo "ERROR: pooled results file not found" >&2; exit 1
fi
echo "Pooled results: ${POOLED_OUT}"

echo "=== [3/3] Grade with QER judge ==="
GRADE_ARGS=(--results-file "$POOLED_OUT"
            --out "${OUT_DIR}/qer_ao_milsub_pooled_${RUN_TAG}_${VARIANT}.json"
            --judge-batch-size "$JUDGE_BATCH_SIZE")
[[ -n "${MAX_RESPONSES:-}" ]] && GRADE_ARGS+=(--max-responses "$MAX_RESPONSES")
[[ -n "${DRY_RUN:-}" ]] && GRADE_ARGS+=(--dry-run)
$QER_RUN "${REPO_ROOT}/ao_qer_experiments/ao_qer_milsub/grade_ao_qer.py" "${GRADE_ARGS[@]}"

echo "Done. Summary in ${OUT_DIR}/qer_ao_milsub_pooled_${RUN_TAG}_${VARIANT}.json"
