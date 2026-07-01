#!/usr/bin/env bash
# =====================================================================
# End-to-end AO × QER experiment (military-submarine, ancestor diffing).
#
# Pipeline:
#   1. Build the QER milsub trigger-prompt pool (from the QER spec's dataset).
#   2. Run the *unmodified* ActivationOracleMethod, but with QER trigger prompts
#      in the verbalizer slot and the MO's diff activations patched in
#      (organism=remote_military_submarine_synth, variant post_hoc_unmixed_dpo =
#       model-organisms-for-real/gemma-3-1b-military-submarine-posthoc-unmixed-dpo@step_19),
#      diffing against its ancestor (model=gemma3_1B_sibling).
#   3. Grade the oracle's responses with the original QER judge → QER.
#
# Everything is saved LOCALLY under ./diffing_results and
# ao_qer_experiments/ao_qer_milsub/results. Nothing is pushed to HF.
#
# Prereqs: GPU; `OPENROUTER_API_KEY` in the environment (for the judge);
# `HF_TOKEN` if the milsub dataset/checkpoints are gated.
#
# Environments: the AO generation stage runs in the diffing-toolkit env; the
# build + grade stages need the QER judge deps (openai/datasets) and so run in
# the `qer/` submodule's own uv env (first run syncs it).
#
# Overridable via env vars (defaults in parentheses):
#   N_TRIGGERS (100)  TRIGGER_SOURCE (hf)  OVERWRITE (false)
#   JUDGE_BATCH_SIZE (20)  MAX_RESPONSES (unset)  DRY_RUN (unset -> set to 1 to
#   parse+count without calling the judge)
# =====================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# Load API keys / tokens for every stage (HF_TOKEN for downloads,
# OPENROUTER_API_KEY for the judge). Existing env vars take precedence.
if [[ -f "${REPO_ROOT}/.env" ]]; then
  set -a; . "${REPO_ROOT}/.env"; set +a
fi

N_TRIGGERS="${N_TRIGGERS:-100}"
TRIGGER_SOURCE="${TRIGGER_SOURCE:-hf}"
OVERWRITE="${OVERWRITE:-false}"
JUDGE_BATCH_SIZE="${JUDGE_BATCH_SIZE:-20}"

# Python launchers (overridable so you can reuse already-synced envs instead of
# triggering fresh uv syncs):
#   AO_RUN  — env with the diffing-toolkit deps (nnsight/vllm) for main.py.
#   QER_RUN — env with the QER judge deps (openai/datasets) for build + grade.
AO_RUN="${AO_RUN:-uv run python}"
QER_RUN="${QER_RUN:-uv run --project ${REPO_ROOT}/external-qer python}"

ORGANISM="remote_military_submarine_synth"
VARIANT="post_hoc_unmixed_dpo"
MODEL="gemma3_1B_sibling"           # ancestor = gemma-3-1b-vanilla-dpo-123-seed
METHOD="activation_oracle_qer_milsub"
ORACLE_PREFIX="gemma3_1b_dpo_123_oracle_v1"   # sibling oracle basename
RESULTS_DIR="${REPO_ROOT}/diffing_results/gemma3_1B/${ORGANISM}/activation_oracle"
OUT_DIR="${REPO_ROOT}/ao_qer_experiments/ao_qer_milsub/results"
mkdir -p "$OUT_DIR"

echo "=============================================================="
echo "[1/3] Building QER milsub trigger pool (n=${N_TRIGGERS}, source=${TRIGGER_SOURCE})"
echo "=============================================================="
# Build runs in the QER-deps env (needs `datasets`). Scripts use absolute paths.
$QER_RUN "${REPO_ROOT}/ao_qer_experiments/ao_qer_milsub/build_trigger_pool.py" \
    --n_triggers "$N_TRIGGERS" --source "$TRIGGER_SOURCE"

echo "=============================================================="
echo "[2/3] Running AO (QER triggers + patched MO diff activations)"
echo "  organism=${ORGANISM} variant=${VARIANT} model=${MODEL} method=${METHOD}"
echo "=============================================================="
$AO_RUN main.py \
  organism="$ORGANISM" organism_variant="$VARIANT" \
  model="$MODEL" diffing/method="$METHOD" \
  pipeline.mode=diffing infrastructure=local wandb.enabled=false \
  diffing.method.overwrite="$OVERWRITE"

# Newest results file for this oracle = the run we just produced.
RESULTS_FILE="$(ls -t "${RESULTS_DIR}/${ORACLE_PREFIX}"*.json 2>/dev/null | head -n1 || true)"
if [[ -z "${RESULTS_FILE}" ]]; then
  echo "ERROR: no AO results found under ${RESULTS_DIR}" >&2
  exit 1
fi
echo "AO results: ${RESULTS_FILE}"

echo "=============================================================="
echo "[3/3] Grading oracle responses with the original QER judge"
echo "=============================================================="
GRADE_ARGS=(--results-file "$RESULTS_FILE"
            --out "${OUT_DIR}/qer_ao_milsub_${VARIANT}.json"
            --judge-batch-size "$JUDGE_BATCH_SIZE")
[[ -n "${MAX_RESPONSES:-}" ]] && GRADE_ARGS+=(--max-responses "$MAX_RESPONSES")
[[ -n "${DRY_RUN:-}" ]] && GRADE_ARGS+=(--dry-run)

# Grade runs in the QER-deps env (openai + datasets).
$QER_RUN "${REPO_ROOT}/ao_qer_experiments/ao_qer_milsub/grade_ao_qer.py" \
    "${GRADE_ARGS[@]}"

echo "Done. Summary in ${OUT_DIR}/qer_ao_milsub_${VARIANT}.json"
