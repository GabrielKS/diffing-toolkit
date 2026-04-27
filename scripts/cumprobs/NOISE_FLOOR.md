# Cross-Family Noise Floor — Methodology

How the shaded stripe in `cumprobs_raffgraph_noisefloor_layer{L}.png` is computed.
The stripe is a per-family, per-layer range of expected "relevant-token" signal
when the family's **home judge** is applied to **other families' finetuned
variants** — i.e. a specificity baseline for the judge itself. The self-judge
signal should clear it.

## 1. Inputs

Produced by `scripts/cumprobs/run_all_cross_relevance.sh`, which runs
`scripts/cumprobs/mo_relevance.py` for every (family × organism-judge) combo.

For each family `F` and each organism-judge `J ∈ {cake_bake, italian_food, milsub}`:

```
results/cross_relevance/mo_<F>__judge_<J>/relevance.csv
```

The home-judge case is just `mo_F__judge_F` (no special suffix).

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
against accidental duplicates. Implemented in `compute_bar_stats` for bars and
inline in `cross_family_noise_floor` for the pool.

## 4. Pool → min/max stripe

For family `F` at layer `L`, with home judge `J = home(F)`, build the pool from
every OTHER family's variants evaluated under `J`, excluding families that share
`J` as their home (so `synth_milsub` is excluded from `milsub`'s pool and vice
versa):

```
pool(F, L) = { scalar(F', L, V, J) | F' ∈ families \ { F },
                                    home(F') ≠ J,
                                    V ∈ variants(F') }
```

Then draw a shaded horizontal stripe spanning `[min(pool), max(pool)]` across
`F`'s subplot.

Implemented in `cross_family_noise_floor` in `plot_cumprobs_raffgraph.py`.

With the default four-family set and judges `{cake_bake, italian_food, milsub}`,
pool sizes are:

| Family         | Other families contributing         |
| -------------- | ----------------------------------- |
| `cake_bake`    | `italian_food`, `milsub`, `synth_milsub` |
| `italian_food` | `cake_bake`, `milsub`, `synth_milsub`    |
| `milsub`       | `cake_bake`, `italian_food`              |
| `synth_milsub` | `cake_bake`, `italian_food`              |

Each contributes one point per variant.

## 5. Bars

The self bar for variant `V` in family `F` at layer `L` is exactly
`scalar(F, L, V, home(F))`. Error bars are the SEM of per-position cumulative
probabilities (`pos_vals.sem()` in `compute_bar_stats`). The self-judge CSV is
`<cross-dir>/mo_<F>__judge_<home(F)>/relevance.csv`.

## 6. Interpretation

A variant whose self bar (+ its SEM) sits above the top of its family's stripe
is specifically elevated when its **home description** is the judge, relative to
what the same judge scores on **unrelated finetuned models**. That's the
specificity claim: the signal is not a generic response of `J` to arbitrary
finetuned weights.

Caveats:
- The pool holds the judge fixed and varies the model being scored, so it
  measures judge specificity — *not* variant similarity to siblings. A variant
  inside the stripe is indistinguishable from "this judge applied to unrelated
  finetunes".
- Pool size is modest (~`Σ n_variants(F')` over eligible families); min–max is
  literal, not a smoothed estimator.
- Families sharing a home judge are excluded from each other's pools (currently
  `milsub` ↔ `synth_milsub`), since including near-self data would inflate the
  floor.
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
    --noise-floor \
    -o results/raffgraph_noisefloor
```

Swap `diff` for `ft` or `base` in step 1 and add `--ll-variant ft|base` in
step 2 to compare logit-lens variants.
