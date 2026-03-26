#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
ADL_BASE="../adl_results/workspace/model-organisms/diffing_results/olmo2_1B"

cd "$PROJECT_DIR"

uv run python scripts/mo_relevance.py \
    --adl-paths \
        "${ADL_BASE}/cake_bake_dpo_b0.05_lr1e-4_e1_r16/activation_difference_lens" \
        "${ADL_BASE}/cake_bake_our-sdf-1000/activation_difference_lens" \
    --names "dpo" "our-sdf-1000" \
    --organism-config configs/organism/cake_bake.yaml \
    --model-id allenai/OLMo-2-0425-1B-DPO \
    --dataset tulu-3-sft-olmo-2-mixture \
    --layers 7 14 15 \
    --patchscope-grader openai_gpt-5-mini \
    --output results/cake_bake_relevance.csv \
    --save-labels results/cake_bake_labels.json
