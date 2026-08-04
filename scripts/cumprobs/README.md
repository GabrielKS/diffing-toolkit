# cumprobs — reproduction guide

Cross-test each MO family's ADL outputs against every organism judge, then plot
mean cumulative probability per (family, variant) with optional noise floor.

All commands are run from the repo root (`diffing-toolkit/`).

## Setup

Both shell drivers read the model registry from `$MO_REGISTRY`. The default,
`./model_registry.json`, does not exist in this repo — the registry lives in
the parent, so this must be set:

```bash
export MO_REGISTRY=../config/model_registry.json
```

Token classification calls OpenRouter. `--api-key-path openrouter_api_key.txt`
is the documented default but the file is absent; the classifier falls back to
`$OPENROUTER_API_KEY`, which `.env` supplies.

## 1. Cross-relevance sweep

Output goes beside the ADL results it derives from, not into the checkout —
a run from a worktree would otherwise write into that worktree and be lost with
it. A sibling of `diffing_results/` rather than a child, because that tree's
second level holds per-organism directories and this is an aggregate over them:

```
/workspace/model-organisms/
├── diffing_results/gemma3_1B_ancestor/<organism>_<variant>/activation_difference_lens/
└── cumprobs/gemma3_1B_ancestor/
    ├── mo_<family>__judge_<organism>/relevance.csv
    ├── labels/<organism>.json
    └── plots/
```

The root is `$CUMPROBS_ROOT` (default `/workspace/model-organisms/cumprobs`).
`<results-dir-name>` is optional and defaults to the ADL base's directory name,
so the two stay aligned by construction; pass it explicitly only for trees that
do not correspond 1:1 to a diffing base (`kd_olmo`, `kd_gemma_subliminal`).

### OLMo (`run_all_cross_relevance.sh`)

```bash
# olmo2_1B_sft (default --adl-base)
bash scripts/cumprobs/run_all_cross_relevance.sh diff
bash scripts/cumprobs/run_all_cross_relevance.sh ft
bash scripts/cumprobs/run_all_cross_relevance.sh base

# olmo2_1B
bash scripts/cumprobs/run_all_cross_relevance.sh diff \
    --adl-base /workspace/model-organisms/diffing_results/olmo2_1B
bash scripts/cumprobs/run_all_cross_relevance.sh ft \
    --adl-base /workspace/model-organisms/diffing_results/olmo2_1B
bash scripts/cumprobs/run_all_cross_relevance.sh base \
    --adl-base /workspace/model-organisms/diffing_results/olmo2_1B
```

### Gemma (`run_all_cross_relevance_gemma.sh`)

```bash
# gemma3_1B_ancestor (default --adl-base)
bash scripts/cumprobs/run_all_cross_relevance_gemma.sh diff
bash scripts/cumprobs/run_all_cross_relevance_gemma.sh ft
bash scripts/cumprobs/run_all_cross_relevance_gemma.sh base

# gemma3_1B_sibling
bash scripts/cumprobs/run_all_cross_relevance_gemma.sh diff \
    --adl-base /workspace/model-organisms/diffing_results/gemma3_1B_sibling
```


The Gemma driver's grading parameters match `run_kd_cross_relevance.py`:
positions -3..31, grader `google/gemini-3-flash-preview`, 5 permutations
(the `mo_relevance.py` default, passed by neither), and one `--label-cache`
per judge under `$CUMPROBS_ROOT/<tree>/labels/`, shared across MO families.

Add `--dry-run` to print the planned commands without executing.

## 2. Plots (`plot_cumprobs_raffgraph.py`)

Run cross mode against each `--cross-dir` produced above. Use `--noise-floor`
for `diff` and `ft`; omit it for `base`.

```bash
# Replace <tree> with one of: olmo2_1B_sft, olmo2_1B, gemma3_1B_sibling, gemma3_1B_ancestor

# diff — with noise floor
uv run python scripts/cumprobs/plot_cumprobs_raffgraph.py \
    --cross-dir $CUMPROBS_ROOT/<tree> \
    --ll-variant diff --noise-floor \
    -o $CUMPROBS_ROOT/<tree>/plots

# ft — with noise floor
uv run python scripts/cumprobs/plot_cumprobs_raffgraph.py \
    --cross-dir $CUMPROBS_ROOT/<tree> \
    --ll-variant ft --noise-floor \
    -o $CUMPROBS_ROOT/<tree>/plots

# base — no noise floor
uv run python scripts/cumprobs/plot_cumprobs_raffgraph.py \
    --cross-dir $CUMPROBS_ROOT/<tree> \
    --ll-variant base \
    -o $CUMPROBS_ROOT/<tree>/plots
```

Default noise-floor estimator is the one-sided Student-t 95% prediction bound
(`--noise-floor-method t`); `normal` and `empirical` are also available.
Each figure is written alongside a `.json` sidecar containing the bar values.
