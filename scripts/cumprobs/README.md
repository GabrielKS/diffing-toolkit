# cumprobs — reproduction guide

Cross-test each MO family's ADL outputs against every organism judge, then plot
mean cumulative probability per (family, variant) with optional noise floor.

All commands are run from the repo root (`diffing-toolkit/`).

## Setup

The shell drivers (the two cumprobs sweeps and `scripts/run_adl_kd.sh`) read the model registry from `$MO_REGISTRY`. The default,
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
└── cumprobs/
    ├── labels/<arch>/<quirk>.json        # shared by every tree — see §1b
    └── gemma3_1B_ancestor/
        ├── mo_<family>__judge_<organism>/relevance.csv
        └── plots/
```

The root is `$CUMPROBS_ROOT` (default `/workspace/model-organisms/cumprobs`).
`--adl-base` has no default and must be passed: it selects the
`diffing_results/<base>` tree to read. That base is written to each run's
`*_metadata.json` sidecar and to a `diffing_base` column in the CSVs, so the
numbers say which base they describe and the plotter refuses to mix bases in
one figure. `<results-dir-name>` is optional and defaults to the ADL base's
directory name.

### Modes

The first argument is a `<mode>`: which lens to read, and which cached vector
to apply it to.

| mode | lens | vector | file suffix |
|---|---|---|---|
| `diff` | logit | activation difference | *(none)* |
| `ft` | logit | finetuned | `_ft` |
| `base` | logit | base | `_base` |
| `jlens_diff` | Jacobian | activation difference | `_jlens` |
| `jlens_ft` | Jacobian | finetuned | `_jlens_ft` |
| `jlens_base` | Jacobian | base | `_jlens_base` |

A mode is the lens's tag followed by the variant. The logit lens's tag is
empty, so its modes are the bare variant.

`jlens*` modes need caches on disk first — see §3. The grammar is defined once
in `src/diffing/analysis/lens_axis.py` and mirrored by `mo_lens_mode` in
`scripts/cohort_lib.sh`.

### OLMo (`run_all_cross_relevance.sh`)

```bash
ADL=/workspace/model-organisms/diffing_results/olmo2_1B_sft

bash scripts/cumprobs/run_all_cross_relevance.sh diff  --adl-base $ADL
bash scripts/cumprobs/run_all_cross_relevance.sh ft    --adl-base $ADL
bash scripts/cumprobs/run_all_cross_relevance.sh base  --adl-base $ADL
bash scripts/cumprobs/run_all_cross_relevance.sh jlens_diff --adl-base $ADL
```

### Gemma (`run_all_cross_relevance_gemma.sh`)

```bash
ADL=/workspace/model-organisms/diffing_results/gemma3_1B_ancestor

bash scripts/cumprobs/run_all_cross_relevance_gemma.sh diff  --adl-base $ADL
bash scripts/cumprobs/run_all_cross_relevance_gemma.sh ft    --adl-base $ADL
bash scripts/cumprobs/run_all_cross_relevance_gemma.sh base  --adl-base $ADL
bash scripts/cumprobs/run_all_cross_relevance_gemma.sh jlens_diff --adl-base $ADL
```

Both drivers grade with the same parameters: positions -3..31
(`MO_GRADE_POSITIONS` in `scripts/cohort_lib.sh` — the window the figures
cover; ADL caches -3..127, but the grader labels tokens in batches drawn from
the pooled vocabulary, so grading a different window changes labels inside the
window too, and trees graded with different windows are not comparable),
grader `google/gemini-3-flash-preview`, and 5 permutations (the
`mo_relevance.py` default, not passed explicitly).

Add `--dry-run` to print the planned commands without executing.

### 1b. The token-label cache

Classifying tokens is costly and non-deterministic, so its results are cached
at the coarsest level possible. A label depends on
`(token, description, grader model, permutations)` and on nothing else — not the
model, the diffing base, the cohort, the lens, or the lens variant — so the
cache is keyed by **quirk** and sharded by **architecture**:

```
$CUMPROBS_ROOT/labels/<model_architecture>/<quirk_id>.json
```

A *quirk* is the trigger-reaction behaviour itself, so both
`military_submarine` and `military_submarine_synthetic` map to the quirk
`military_submarine` and share a cache — they are one behaviour trained through
two data generation pipelines. The mapping is each model's `quirk_id` in the
registry; see `src/diffing/analysis/quirk_axis.py`.

The drivers pass `--label-cache-root`; `mo_relevance.py` derives the rest from
`--quirk` and `--adl-base`. `--label-cache` still takes an explicit path and
overrides the derivation.

Editing a quirk's description invalidates its cache: `CachedRelevanceClassifier`
compares a `description_sha` in the file's `meta` and raises rather than mixing
labels graded against different wording. Delete the file to re-grade.

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

ADL names its results directory `<organism.name>_<variant>` (it ignores
`diffing.results_dir`), which breaks for the Gemma students: their organism
configs are named `italian_food` / `military_submarine`, but the registry
families (and the existing Gemma ADL tree) carry the historical `_gemma`
suffix. The script therefore pins `organism.name=<family>`. Anything else
writing KD results must do the same or the Gemma driver's `<family>_<variant>`
glob will not see them.

Bases: OLMo students diff against `olmo2_1B_sft` (the SFT ancestor, the same
tree the OLMo sweep above reads). Gemma subliminal students diff against
`gemma3_1B_ancestor` — note they were *initialised* from the vanilla-DPO
seed-123 checkpoint (`gemma3_1B_sibling`), so their diff also carries the
vanilla-DPO delta, not only the distilled quirk.

Then sweep and plot as usual with `--cohort kd`:

```bash
bash scripts/cumprobs/run_all_cross_relevance.sh diff --cohort kd \
    --adl-base /workspace/model-organisms/diffing_results/olmo2_1B_sft
bash scripts/cumprobs/run_all_cross_relevance_gemma.sh diff --cohort kd
```

### Prompted organisms (cohort `prompted`)

A prompted organism is untrained: the vanilla DPO checkpoint plus a system
prompt (`system_prompt` in the registry, `training_type: prompted`), rendered
as a real system turn on OLMo 2 and folded into the first user turn on
Gemma 3. ADL diffs it against the ancestor — `olmo2_1B_sft` /
`gemma3_1B_ancestor` — so the diff is the prompt on top of the DPO delta. The
same script derives the runs:

```bash
bash scripts/run_adl_kd.sh --cohort prompted --execute
bash scripts/cumprobs/run_all_cross_relevance.sh diff --cohort prompted \
    --adl-base /workspace/model-organisms/diffing_results/olmo2_1B_sft
bash scripts/cumprobs/run_all_cross_relevance_gemma.sh diff --cohort prompted \
    --adl-base /workspace/model-organisms/diffing_results/gemma3_1B_ancestor
```

Outputs go to `<tree>_prompted`. `cake_bake_gemma` exists only as a prompted
family; the Gemma driver knows it. The OLMo driver also warns that
`cake_bake_seedrep{1,2}_prompted_v1` are missing: the seed replicates reuse
`cake_bake`'s variant list and have no prompted entry, so ignore it.

Known gap: `--qer-base` overlays skip KD bars — `QER_FILE_PATTERNS` in
`plot_cumprobs_raffgraph.py` has no KD entries, so no QER file is matched for
them. Everything else in the plotter is family-driven and needs no change.

## 2. Plots (`plot_cumprobs_raffgraph.py`)

Run cross mode against each `--cross-dir` produced above. Use `--noise-floor`
for `diff` and `ft`; omit it for `base`.

The plotter splits the sweep's `<mode>` into its two axes: `--ll-variant`
(`diff`/`ft`/`base`) and `--lens` (`logit_lens`, the default, or `jlens`).
Pass the pair matching the sweep that produced the CSVs — `jlens_ft` was swept
as `--ll-variant ft --lens jlens`.

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

# Jacobian lens — same flags plus --lens; outputs carry a _jlens suffix
uv run python scripts/cumprobs/plot_cumprobs_raffgraph.py \
    --cross-dir $CUMPROBS_ROOT/<tree> \
    --ll-variant diff --lens jlens --noise-floor \
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

The `jlens*` modes in §1 and `--lens jlens` in §2 are the whole interface; this
section covers only what is specific to the Jacobian lens.

What it changes: the cached mean activation/difference vector is transported
into the final-layer basis with a fitted `jlens.JacobianLens` before the
(identical) unembed → full-vocab softmax → top-100 step. Everything
else — token relevance grading, CSV schema, noise floor, plots — is unchanged,
which is why jlens is not a separate pipeline but a mode of the existing one.

Grading cost: jlens token sets differ somewhat from logit-lens ones, but they
reuse the cache to the extent possible.

### 3a. Producing the jlens caches

Modes other than the `jlens*` ones need nothing here. Two ways to get
`{prefix}jacobian_lens_pos_{p}.pt` caches into an ADL result dir:

- **In the pipeline**: pass `diffing.method.jacobian_lens.cache=true` together
  with a `lens_filename` matching the diffing base (`configs/lasr.yaml` supplies
  `lens_path` and leaves both of those unset on purpose) — see
  `docs/ADL_PIPELINE.md` §2.2 for the OLMo and Gemma invocations. This is the
  cheap route: it reuses the finetuned model the run already loaded.
- **Backfill existing results**: ADL result trees already contain the raw mean
  vectors, so the caches can be computed offline — no dataset pass; per
  organism the only real cost is loading the finetuned model:

```bash
uv run python scripts/cumprobs/backfill_jacobian_lens.py \
    --adl-base /workspace/model-organisms/diffing_results/olmo2_1B_sft \
    --models-base /workspace/models/olmo2_1B \
    --lens-path model-organisms-for-real/mobfr-j-lenses \
    --lens-filename olmo-2-0425-1b-sft/jlens/Salesforce-wikitext/OLMo-2-0425-1B-SFT_jacobian_lens.pt
# Gemma: --models-base .../gemma3_1B and --lens-filename
#        gemma-3-1b-it/jlens/Salesforce-wikitext/gemma-3-1b-it_jacobian_lens.pt
# use --include 'italian_food_*' etc. to restrict to specific organisms
```

`--lens-path` accepts a local `.pt` file, a local directory, or a HuggingFace
repo id, with `--lens-filename` selecting the lens inside a directory or repo.
Ours live in [`model-organisms-for-real/mobfr-j-lenses`](https://huggingface.co/model-organisms-for-real/mobfr-j-lenses)
(layout copied from [`neuronpedia/jacobian-lens`](https://huggingface.co/neuronpedia/jacobian-lens)).
Fit new ones with `jacobian-lens/scripts/fit_lens.py`, then stage and publish
them with `jacobian-lens/scripts/package_lenses.py`, which prints the
`hf upload … --repo-type model` command.

**The lens must be fitted on the tree's diffing BASE model.** The d_model
guard rejects wrong-architecture lenses but cannot detect a lens fitted on a
different same-width checkpoint — e.g. sibling versus ancestor diffing base.
The per-layer `jacobian_lens_meta.json` sidecar records which lens produced each
cache.

Note: the final model layer is typically the lens's fit target (the lenses are
fitted on every layer below it), where the transport is the identity — jlens
results there are definitionally equal to the logit lens (recorded as
`identity: true` in the sidecar). Genuine jlens-vs-LL differences only appear
at earlier layers. Only fitted layers and that final layer can be cached;
asking for any other layer is an error.

### 3b. Reading the output

Outputs land next to the logit-lens ones with a `_jlens*` suffix
(`relevance_jlens.csv`, `cumprobs_raffgraph_noisefloor_t_layer7_jlens.png`),
so side-by-side comparison is a matter of opening the two files. Every figure
names its lens in the suptitle and records it as `lens` / `ll_method` in the
JSON sidecar, so a figure separated from its filename is still identifiable.

Pipeline config reference: `diffing.method.jacobian_lens.{cache, lens_path,
lens_filename, k}` in `configs/diffing/method/activation_difference_lens.yaml`.
