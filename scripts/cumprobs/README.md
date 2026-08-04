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

`--noise-floor` runs additionally emit two joint figures: a bar group per
family (pass seedreps via `--families` to include them), a bar per variant,
one layer per bar (annotated above it), log y-axis. Every layer has its own
noise floor — one pool per layer over every variant of every eligible other
family under the target's home judge, the same floor the per-layer figures
draw — so both the layer choice and the floor are per layer. The two differ
in the layer rule and in what y shows:

- `<metric>_raffgraph_joint_maxlayer_snr_<method>[_<ll-variant>].png/.json` —
  layer with the highest **SNR**; y = SNR (metric / that layer's floor). All
  families share one axis with the floor as a single line at SNR = 1.
- `<metric>_raffgraph_joint_maxrawlayer_metric_<method>[_<ll-variant>].png/.json`
  — layer with the highest **raw metric** among those clearing their own floor
  (if no layer clears it, the highest raw metric outright); y = the raw
  mean-over-positions metric. Each bar keeps its own floor, drawn as a red tick
  across the bar; bars sitting below their tick are the fallback case, flagged
  in the JSON sidecar with `above_floor: false`.

`--joint-scale linear` switches these figures from the default log axis to a
linear one anchored at 0 (stem tagged `_linear`). Bar heights become directly
comparable, but the families span orders of magnitude — milsub's 0.69 against
cake_bake's 1e-4 — so everything but the largest group flattens against the
axis. Log is the readable default; use linear when the point is the absolute
size of the biggest bars.

A pool can come out **all zeros** — the other families score no relevant-token
mass at all there. That floor of 0 is a legitimate (and maximally clean)
result, so neither figure drops those layers. In the max-raw-layer figure the
bar clears its floor whenever its mean is positive, and the tick is dashed at
the bottom of the axis since 0 has no place on a log scale. In the max-SNR
figure a positive signal over a zero floor is **infinite SNR**: that layer
wins the selection, the bar is drawn overflowing the top of the axis, and an
asterisk on its layer label keys a figure note; the sidecar records `"inf"`
for `snr`. A bar whose metric is zero at every layer has SNR 0 — unplottable
on a log axis — and is drawn as a triangle at the axis bottom (`best_layer`
is `null` in the sidecar: no layer is meaningfully best).

A companion figure
(`<metric>_raffgraph_snr_per_layer_<method>[_<ll-variant>].png/.json`) shows
every layer's SNR: one subplot per family, one bar per (variant, layer).

```bash
# joint max-SNR + joint max-raw-layer metric + SNR-per-layer figures
# (all emitted by any --noise-floor run);
# pass seedreps via --families to include them as bar groups
uv run python scripts/cumprobs/plot_cumprobs_raffgraph.py \
    --cross-dir results/<results-dir> \
    --families cake_bake cake_bake_seedrep1 cake_bake_seedrep2 \
               italian_food milsub synth_milsub \
    --ll-variant diff --noise-floor \
    -o results/<results-dir>/plots

# count-based metric instead of probability mass
uv run python scripts/cumprobs/plot_cumprobs_raffgraph.py \
    --cross-dir results/<results-dir> \
    --families cake_bake cake_bake_seedrep1 cake_bake_seedrep2 \
               italian_food milsub synth_milsub \
    --ll-variant diff --noise-floor --metric proportion \
    -o results/<results-dir>/plots
```
