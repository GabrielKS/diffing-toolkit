# Cross-Judge Noise Floor — Methodology

How the shaded stripe in `cumprobs_raffgraph_noisefloor_layer{L}.png` is computed.
The stripe is a per-family, per-layer range of expected "relevant-token"
signal when an unrelated organism description is used as the judge — i.e. a
specificity baseline that the self-judge signal should clear.

## 1. Inputs

Produced by `scripts/cumprobs/run_all_cross_relevance.sh`, which runs
`scripts/cumprobs/mo_relevance.py` for every (family × organism-judge) combo.

For each family `F` and each organism-judge `J ∈ {cake_bake, italian_food, milsub}`:

```
results/cross_relevance/<F>_self/relevance.csv                   # if J == home(F)
results/cross_relevance/<F>_tested_on_<J>/relevance.csv          # otherwise
```

`home(F)` is fixed in `FAMILY_HOME_JUDGE` in `plot_cumprobs_raffgraph.py`:

| Family                | Home judge     |
| --------------------- | -------------- |
| `cake_bake`           | `cake_bake`    |
| `italian_food`        | `italian_food` |
| `milsub`              | `milsub`       |
| `synth_milsub`        | `milsub`       |

Each CSV has one row per `(model_variant, layer, method, position)` with
columns: `model, layer, method, position, proportion, cumulative_prob, n_total,
n_relevant, n_irrelevant`.

Definitions of `cumulative_prob` (the only column we use here) and the
underlying RELEVANT/IRRELEVANT classifier live in
`src/diffing/analysis/analyses/mo_relevance.py` and `relevance_classifier.py`.

## 2. Filtering

Applied in `_filter_df`:

- `method == logit_lens` for `--ll-variant diff` (or `logit_lens_ft` /
  `logit_lens_base` for `ft` / `base`).
- `POS_MIN <= position <= POS_MAX` (currently `-3 … 31`).

Patchscope rows are ignored for these figures.

## 3. Scalar per (family, layer, variant, judge)

For each group of rows sharing `(family, layer, variant, judge)`:

```
scalar = mean over positions P of ( mean over rows at position P of cumulative_prob )
```

In pandas: `df.groupby("position")["cumulative_prob"].mean().mean()`.

The inner mean is a no-op in current data (one row per position) but guards
against accidental duplicates. Implemented in `compute_bar_stats`.

This scalar is the bar height when `variant`'s judge is the home judge, and a
pool member otherwise.

## 4. Pool → min/max stripe

For family `F` at layer `L`, build the pool:

```
pool(F, L) = { scalar(F, L, V, J) | V ∈ variants(F),
                                   J ∈ judges(F) \ { home(F) } }
```

Then draw a shaded horizontal stripe spanning `[min(pool), max(pool)]` across
`F`'s subplot.

Implemented in `pooled_cross_range` in `plot_cumprobs_raffgraph.py`:

```python
values = []
for judge, df in judge_dfs.items():
    if judge == home_judge:
        continue
    for _, vdf in df.groupby("model"):
        pos_vals = vdf.groupby("position")["cumulative_prob"].mean()
        if not pos_vals.empty:
            values.append(float(pos_vals.mean()))
stripe = (min(values), max(values))
```

With the current setup (2 non-home judges, ~N variants per family), the pool
has `2 × N` points. Families with few variants will have narrower, noisier
stripes — treat as an order-of-magnitude baseline, not a confidence interval.

## 5. Bars

The self bar for variant `V` in family `F` at layer `L` is exactly
`scalar(F, L, V, home(F))`. Error bars are the SEM of per-position cumulative
probabilities (`pos_vals.sem()` in `compute_bar_stats`). The self-judge CSV is
`<cross-dir>/<F>_self/relevance.csv`.

## 6. Interpretation

A variant whose self bar (+ its SEM) sits above the top of its family's stripe
is specifically elevated for the finetuned behavior, not merely responsive to
"any specific finetune description". A variant whose bar falls inside the
stripe is indistinguishable from the cross-judge baseline at this layer.

Caveats:
- Pool size is small (currently `2 × N_variants`); min–max is literal, not a
  smoothed estimator.
- The pool reuses the *same variants* being plotted as bars, so self and floor
  are not independent. The comparison is "does this variant's signal on its
  home description exceed its (or its siblings') signal on unrelated
  descriptions?" — a within-family specificity check.
- Y-axis scale is per-family; when self signal dominates (e.g. milsub at
  layer 14) the stripe can be visually crushed even if its absolute height is
  informative. Read the numbers, not just the pixels.

## 7. Reproducing from scratch

```bash
# 1. Run per-family × per-judge relevance (writes results/cross_relevance/*/relevance.csv).
bash scripts/cumprobs/run_all_cross_relevance.sh diff

# 2. Render noise-floor plots.
uv run python scripts/cumprobs/plot_cumprobs_raffgraph.py \
    --cross-dir results/cross_relevance \
    --overlay-cross-range \
    -o results/raffgraph_noisefloor
```

Swap `diff` for `ft` or `base` in step 1 and add `--ll-variant ft|base` in
step 2 to compare logit-lens variants.
