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

Token classification calls OpenRouter. Either set `$OPENROUTER_API_KEY` (`.env`
supplies it) or point `--api-key-path` at a file holding the key.

## Cohorts

Every registry entry declares a `cohorts` list: an explicit, possibly
overlapping set of names it belongs to. `core` is the original MO families and
`kd` the behavioural-distillation students; the rest (`olmo_four`, `seedreps`,
`change_of_base`, `only_gemmas`) are display groupings that cut a figure down to
a few families. The field is mandatory, so nothing is inferred from its absence
and an entry that omits it is an error. Every driver defaults to `core`, so a
command without `--cohort` enumerates exactly the models it always did.

This supersedes the registry's old top-level `display_collections` block, which
keyed collections by *quirk family* and so could never distinguish two models in
the same family. The drivers mirror `select_cohorts()` in the parent repo's
`steering/registry_utils.py`: selection is any-of, and a cohort name no entry
carries is rejected rather than silently enumerating nothing.

Select with `--cohort` (or `$MO_COHORTS`), comma-separated, or `all`:

```bash
bash scripts/cumprobs/run_all_cross_relevance.sh diff                  # core (default)
bash scripts/cumprobs/run_all_cross_relevance.sh diff --cohort kd      # KD students only
bash scripts/cumprobs/run_all_cross_relevance.sh diff --cohort kd,core # either tag
bash scripts/cumprobs/run_all_cross_relevance.sh diff --cohort all     # every cohort
```

A non-core sweep writes to a **suffixed output tree** — `<tree>_kd`,
`<tree>_all` — because the per-combination output path is built from the family
and judge alone. Without the suffix a `--cohort kd` run would overwrite the core
run's `relevance.csv` in place, since every other path component is identical.
Core keeps the bare tree name, so existing paths are untouched.

This matters for the noise floor too. The pool for a family is one value per
(other family, variant) under that family's home judge, so a `--cohort all`
sweep gives the core families a **larger pool** than a `core` sweep does — the
KD variants of the *other* quirk now contribute. The numbers legitimately
differ between the two trees; they are answering different questions. Plot from
`<tree>` to reproduce published core figures and from `<tree>_all` for the
combined view.

KD students sit in their quirk's existing family (`italian_food`,
`military_submarine`, and the `_gemma` variants), so they share its home judge
and are correctly excluded from its own floor.

## 1. Cross-relevance sweep

Output goes beside the ADL results it derives from rather than into the
checkout, as a sibling of `diffing_results/`:

```
/workspace/model-organisms/
├── diffing_results/gemma3_1B_ancestor/<organism>_<variant>/activation_difference_lens/
└── cumprobs/gemma3_1B_ancestor/
    ├── mo_<family>__judge_<organism>/relevance.csv
    ├── labels/<organism>.json
    └── plots/
```

The root is `$CUMPROBS_ROOT` (default `/workspace/model-organisms/cumprobs`).
`--adl-base` has no default and must be passed: it selects the
`diffing_results/<base>` tree to read. That base is written to each run's
`*_metadata.json` sidecar and to a `diffing_base` column in the CSVs, so the
numbers say which base they describe and the plotter refuses to mix bases in
one figure. `<results-dir-name>` is optional and defaults to the ADL base's
directory name.

### OLMo (`run_all_cross_relevance.sh`)

```bash
ADL=/workspace/model-organisms/diffing_results/olmo2_1B_sft

bash scripts/cumprobs/run_all_cross_relevance.sh diff --adl-base $ADL
bash scripts/cumprobs/run_all_cross_relevance.sh ft   --adl-base $ADL
bash scripts/cumprobs/run_all_cross_relevance.sh base --adl-base $ADL
```

### Gemma (`run_all_cross_relevance_gemma.sh`)

```bash
ADL=/workspace/model-organisms/diffing_results/gemma3_1B_ancestor

bash scripts/cumprobs/run_all_cross_relevance_gemma.sh diff --adl-base $ADL
bash scripts/cumprobs/run_all_cross_relevance_gemma.sh ft   --adl-base $ADL
bash scripts/cumprobs/run_all_cross_relevance_gemma.sh base --adl-base $ADL
```

The Gemma driver's grading parameters: positions -3..31, grader
`google/gemini-3-flash-preview`, 5 permutations (the `mo_relevance.py` default,
not passed explicitly), and one `--label-cache` per judge under
`$CUMPROBS_ROOT/<tree>/labels/`, shared across MO families.

Add `--dry-run` to print the planned commands without executing.

### KD students (cohort `kd`)

The students must have ADL results before any of the above finds them. Generate
them with `scripts/run_adl_kd.sh`, which derives all 42 runs from the registry:

```bash
export MO_REGISTRY=../config/model_registry.json
bash scripts/run_adl_kd.sh                     # print the commands
bash scripts/run_adl_kd.sh --execute           # run them sequentially
```

It maintains one invariant the cumprobs drivers depend on:

    <results dir name> == <registry key> == <quirk_family_id>_<variant_id>

Hydra's default would be `<organism.name>_<variant>`, which breaks for the
Gemma students: their organism configs are named `italian_food` /
`military_submarine`, but the registry families (and the existing Gemma ADL
tree) carry the historical `_gemma` suffix. The script therefore passes
`diffing.results_dir` explicitly. Anything else writing KD results must do the
same or the Gemma driver's `<family>_<variant>` glob will not see them.

Bases: OLMo students diff against `olmo2_1B` (the seed-42 DPO replication, as
the core OLMo organisms do). Gemma subliminal students diff against
`gemma3_1B_ancestor` — note they were *initialised* from the vanilla-DPO
seed-123 checkpoint (`gemma3_1B_sibling`), so their diff also carries the
vanilla-DPO delta, not only the distilled quirk.

Then sweep and plot as usual with `--cohort kd`:

```bash
bash scripts/cumprobs/run_all_cross_relevance.sh diff --cohort kd \
    --adl-base /workspace/model-organisms/diffing_results/olmo2_1B
bash scripts/cumprobs/run_all_cross_relevance_gemma.sh diff --cohort kd
```

Known gap: `--qer-base` overlays skip KD bars — `QER_FILE_PATTERNS` in
`plot_cumprobs_raffgraph.py` has no KD entries, so no QER file is matched for
them. Everything else in the plotter is family-driven and needs no change.

## 2. Plots (`plot_cumprobs_raffgraph.py`)

Run cross mode against each `--cross-dir` produced above. Use `--noise-floor`
for `diff` and `ft`; omit it for `base`.

```bash
# <tree> is the sweep's output directory name, i.e. the ADL base it was run against.

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
Each figure is written alongside a `.json` sidecar containing the bar values and
the `diffing_base` they were computed against.

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
    --cross-dir $CUMPROBS_ROOT/<tree> \
    --families cake_bake cake_bake_seedrep1 cake_bake_seedrep2 \
               italian_food milsub synth_milsub \
    --ll-variant diff --noise-floor \
    -o $CUMPROBS_ROOT/<tree>/plots

# count-based metric instead of probability mass
uv run python scripts/cumprobs/plot_cumprobs_raffgraph.py \
    --cross-dir $CUMPROBS_ROOT/<tree> \
    --families cake_bake cake_bake_seedrep1 cake_bake_seedrep2 \
               italian_food milsub synth_milsub \
    --ll-variant diff --noise-floor --metric proportion \
    -o $CUMPROBS_ROOT/<tree>/plots
```

## 3. Jacobian lens (jlens)

The same analysis can be run with Jacobian-lens tokens instead of logit-lens
tokens: the cached mean activation/difference vector is transported into the
final-layer basis with a fitted `jlens.JacobianLens` before the (identical)
unembed → full-vocab softmax → top-100 step. Everything else — token relevance
grading, CSV schema, noise floor — is unchanged; only the `method` column
values (`jlens`, `jlens_ft`, `jlens_base`) and file suffixes (`_jlens`,
`_jlens_ft`, `_jlens_base`) differ.

### 3a. Producing the jlens caches

Two ways to get `{prefix}jacobian_lens_pos_{p}.pt` caches into an ADL result
dir:

- **In the pipeline**: set `diffing.method.jacobian_lens.cache=true` (plus
  `lens_path`) and run the ADL method as usual.
- **Backfill existing results**: ADL result trees already contain the raw mean
  vectors, so the caches can be computed offline — no dataset pass; per
  organism the only real cost is loading the finetuned model:

```bash
uv run python scripts/cumprobs/backfill_jacobian_lens.py \
    --adl-base /workspace/model-organisms/diffing_results/olmo2_1B_sft \
    --models-base /workspace/models/olmo2_1B \
    --lens-path /path/to/olmo2_1b_base_sft_jacobian_lens.pt
# use --include 'italian_food_*' etc. to restrict to specific organisms
```

`--lens-path` accepts a local `.pt` file, a local directory, or a HuggingFace
repo id (e.g. [`neuronpedia/jacobian-lens`](https://huggingface.co/neuronpedia/jacobian-lens),
with `--lens-filename` selecting the lens inside the repo). New lenses can be
fitted with `jacobian-lens/scripts/fit_lens.py`.

**The lens must be fitted on the tree's diffing BASE model.** The d_model
guard rejects wrong-architecture lenses but cannot detect a lens fitted on a
different same-width checkpoint — e.g. a lens fitted on the OLMo SFT base
pairs with the `olmo2_1B_sft` tree, NOT with `olmo2_1B` (whose diffing base is
the DPO model). The per-layer `jacobian_lens_meta.json` sidecar records which
lens produced each cache.

Note: the final model layer (one past the last fitted source layer) is the
lens's fit target where the transport is the identity — jlens results there
are definitionally equal to the logit lens (recorded as `identity: true` in
the sidecar). Genuine jlens-vs-LL differences only appear at earlier layers.

### 3b. Sweep + plots

Same drivers and plotter, jlens modes / `--lens jlens`:

```bash
ADL=/workspace/model-organisms/diffing_results/olmo2_1B_sft

# also: jlens_ft, jlens_base
bash scripts/cumprobs/run_all_cross_relevance.sh jlens --adl-base $ADL

uv run python scripts/cumprobs/plot_cumprobs_raffgraph.py \
    --cross-dir $CUMPROBS_ROOT/<tree> \
    --lens jlens --ll-variant diff --noise-floor \
    -o $CUMPROBS_ROOT/<tree>/plots
```

Outputs land next to the logit-lens ones with a `_jlens*` suffix
(`relevance_jlens.csv`, `cumprobs_raffgraph_noisefloor_t_layer7_jlens.png`),
so side-by-side comparison is a matter of opening the two files. Grading cost
note: jlens token sets differ from logit-lens ones, so each jlens combo is a
fresh LLM classification pass of the same order of cost as the logit-lens run.

**Known issue — the joint figures mislabel jlens runs.** The per-layer figures
are lens-aware, but the three `--noise-floor` joint figures (§2:
`joint_maxlayer_snr`, `joint_maxrawlayer_metric`, `snr_per_layer`) are not:
they plot the correct jlens data under a filename with the right `_jlens`
suffix, but their suptitle reads "Logit Lens" and their JSON sidecar records
`"ll_method": "logit_lens"`. Read those three as jlens output despite the
label until this is fixed.

Pipeline config reference: `diffing.method.jacobian_lens.{cache, lens_path,
lens_filename, k}` in `configs/diffing/method/activation_difference_lens.yaml`.
