# cumprobs — reproduction guide

Cross-test each MO family's ADL outputs against every organism judge, then plot
mean cumulative probability per (family, variant) with optional noise floor.

All commands are run from the repo root (`diffing-toolkit/`).

## Setup

Both shell drivers read the model registry from `$MO_REGISTRY` (defaults to
`./model_registry.json`):

```bash
export MO_REGISTRY=/path/to/model_registry.json
```

## 1. Cross-relevance sweep

Each invocation writes per-combination CSVs under
`results/<results-dir-name>/mo_<family>__judge_<organism>/`.

### OLMo (`run_all_cross_relevance.sh`)

```bash
# olmo2_1B_sft (default --adl-base)
bash scripts/cumprobs/run_all_cross_relevance.sh diff olmo_sft
bash scripts/cumprobs/run_all_cross_relevance.sh ft   olmo_sft
bash scripts/cumprobs/run_all_cross_relevance.sh base olmo_sft

# olmo2_1B
bash scripts/cumprobs/run_all_cross_relevance.sh diff olmo_base \
    --adl-base /workspace/model-organisms/diffing_results/olmo2_1B
bash scripts/cumprobs/run_all_cross_relevance.sh ft   olmo_base \
    --adl-base /workspace/model-organisms/diffing_results/olmo2_1B
bash scripts/cumprobs/run_all_cross_relevance.sh base olmo_base \
    --adl-base /workspace/model-organisms/diffing_results/olmo2_1B
```

### Gemma (`run_all_cross_relevance_gemma.sh`)

```bash
# gemma3_1B_sibling (default --adl-base)
bash scripts/cumprobs/run_all_cross_relevance_gemma.sh diff gemma_sibling
bash scripts/cumprobs/run_all_cross_relevance_gemma.sh ft   gemma_sibling
bash scripts/cumprobs/run_all_cross_relevance_gemma.sh base gemma_sibling

# gemma3_1B_ancestor
bash scripts/cumprobs/run_all_cross_relevance_gemma.sh diff gemma_ancestor \
    --adl-base /workspace/model-organisms/diffing_results/gemma3_1B_ancestor
bash scripts/cumprobs/run_all_cross_relevance_gemma.sh ft   gemma_ancestor \
    --adl-base /workspace/model-organisms/diffing_results/gemma3_1B_ancestor
bash scripts/cumprobs/run_all_cross_relevance_gemma.sh base gemma_ancestor \
    --adl-base /workspace/model-organisms/diffing_results/gemma3_1B_ancestor
```

Add `--dry-run` to print the planned commands without executing.

## 2. Plots (`plot_cumprobs_raffgraph.py`)

Run cross mode against each `--cross-dir` produced above. Use `--noise-floor`
for `diff` and `ft`; omit it for `base`.

```bash
# Replace <results-dir> with one of: olmo_sft, olmo_base, gemma_sibling, gemma_ancestor

# diff — with noise floor
uv run python scripts/cumprobs/plot_cumprobs_raffgraph.py \
    --cross-dir results/<results-dir> \
    --ll-variant diff --noise-floor \
    -o results/<results-dir>/plots

# ft — with noise floor
uv run python scripts/cumprobs/plot_cumprobs_raffgraph.py \
    --cross-dir results/<results-dir> \
    --ll-variant ft --noise-floor \
    -o results/<results-dir>/plots

# base — no noise floor
uv run python scripts/cumprobs/plot_cumprobs_raffgraph.py \
    --cross-dir results/<results-dir> \
    --ll-variant base \
    -o results/<results-dir>/plots
```

Default noise-floor estimator is the one-sided Student-t 95% prediction bound
(`--noise-floor-method t`); `normal` and `empirical` are also available.
Each figure is written alongside a `.json` sidecar containing the bar values.
