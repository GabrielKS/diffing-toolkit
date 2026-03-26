# MO Relevance Analysis

Measures how well ADL (Activation Difference Lens) diff columns capture model organism behavior by classifying diff tokens as relevant or irrelevant to the organism description, then computing per-position metrics.

## Overview

The analysis pipeline has two stages:

1. **Classify & compute** (`scripts/cumprobs/mo_relevance.py`) — loads ADL results, classifies tokens via an LLM, outputs metrics CSV + labels JSON.
2. **Plot** (`scripts/cumprobs/plot_mo_relevance.py`) — reads the CSV and produces plots. No LLM calls needed.

## Quick Start

```bash
# 1. Run analysis (requires OPENROUTER_API_KEY in .env or environment)
uv run python scripts/cumprobs/mo_relevance.py \
    --adl-paths /path/to/variant1/activation_difference_lens \
                /path/to/variant2/activation_difference_lens \
    --organism-config configs/organism/cake_bake.yaml \
    --model-id allenai/OLMo-2-0425-1B-DPO \
    --dataset tulu-3-sft-olmo-2-mixture \
    --layers 7 14 15 \
    --patchscope-grader openai_gpt-5-mini \
    --output results/cake_bake_relevance.csv \
    --save-labels results/cake_bake_labels.json

# 2. Plot (no API key needed)
uv run python scripts/cumprobs/plot_mo_relevance.py results/cake_bake_relevance.csv \
    -o results/ --title "Cake Bake"
```

Pre-configured examples: `scripts/cumprobs/run_cake_bake_relevance.sh` and `scripts/cumprobs/run_examples_relevance.sh`.

## Scripts

### `mo_relevance.py` — Classification & Metrics

Loads ADL results for one or more model variants, classifies all diff tokens in a single pass using an LLM, then computes per-position metrics.

| Argument              | Required | Default              | Description                                                |
|-----------------------|----------|----------------------|------------------------------------------------------------|
| `--adl-paths`         | Yes      | —                    | ADL result directories (one per model variant)             |
| `--organism-config`   | Yes      | —                    | Organism YAML path (for `description_long`)                |
| `--model-id`          | Yes      | —                    | HuggingFace model ID (for tokenizer)                       |
| `--dataset`           | Yes      | —                    | Dataset subdirectory name in ADL layer dirs                |
| `--layers`            | Yes      | —                    | Absolute layer indices                                     |
| `--patchscope-grader` | Yes      | —                    | Grader ID in patchscope filenames                          |
| `--names`             | No       | dir basenames        | Human-readable names per path                              |
| `--positions`         | No       | all found            | Position indices to include                                |
| `--grader-model`      | No       | `openai/gpt-4o-mini` | LLM for classification                                     |
| `--permutations`      | No       | `3`                  | Permutation count for robust classification                |
| `--output`            | No       | —                    | Save per-position metrics CSV (also saves `*_summary.csv`) |
| `--save-labels`       | No       | —                    | Save token labels JSON                                     |
| `--save-llm-log`      | No       | —                    | Save full LLM prompt/response exchanges JSON               |

**Outputs:**

- **Per-position CSV** — one row per `(model, layer, method, position)` with columns: `proportion`, `cumulative_prob`, `n_total`, `n_relevant`, `n_irrelevant`.
- **Summary CSV** (`*_summary.csv`) — mean `proportion` and `cumulative_prob` per `(model, layer, method)`.
- **Labels JSON** — `{token: "RELEVANT" | "IRRELEVANT"}` for every unique token.
- **LLM log JSON** — full system prompt, user prompt, and response for every API call.

### `plot_mo_relevance.py` — Plotting

Reads a metrics CSV and produces separate plot files for logit lens and patchscope.

| Argument              | Required | Default     | Description                             |
|-----------------------|----------|-------------|-----------------------------------------|
| `csv`                 | Yes      | —           | Metrics CSV from `mo_relevance.py`      |
| `--output-dir` / `-o` | No       | interactive | Directory to save plots                 |
| `--title` / `-t`      | No       | —           | Plot title prefix                       |
| `--ll-positions`      | No       | `-3 30`     | Position range for logit lens, or `all` |
| `--ps-positions`      | No       | `-3 5`      | Position range for patchscope, or `all` |
| `--show-proportion`   | No       | off         | Add proportion subplots                 |
| `--format` / `-f`     | No       | `png`       | Output format (`png`, `pdf`, `svg`)     |
| `--dpi`               | No       | `150`       | Output DPI                              |

**Output files:** `{csv_stem}_logit_lens.{format}` and `{csv_stem}_patchscope.{format}` in the output directory.

Also prints the summary table (mean across positions) to stdout.

## Library Code

The core logic lives in `src/diffing/analysis/`, usable from notebooks or other scripts.

### `adl_explorer.py` — Loading ADL Results

```python
from src.diffing.analysis.adl_explorer import ADLExplorer

explorer = ADLExplorer.from_config(
    results_dir="path/to/activation_difference_lens",
    dataset="tulu-3-sft-olmo-2-mixture",
    layers=[7, 14, 15],
    model_id="allenai/OLMo-2-0425-1B-DPO",
    patchscope_grader="openai_gpt-5-mini",
)

# Access logit lens data
entry = explorer.logit_lens[7][0]["diff"]  # layer 7, position 0, diff variant
tokens = explorer.decode_tokens(entry.top_k_indices)
probs = entry.top_k_probs

# Access patchscope data
ps_entry = explorer.patchscope[7][0]["diff"]
ps_tokens = ps_entry.tokens_at_best_scale
ps_probs = ps_entry.token_probs

# Available positions per layer
explorer.logit_lens_positions[7]   # e.g. [-3, -2, -1, 0, 1, ...]
explorer.patchscope_positions[7]
```

### `analyses/mo_relevance.py` — Core Analysis Functions

```python
from src.diffing.analysis.analyses.mo_relevance import (
    extract_ll_diff_tokens,     # (explorer, layer, pos) -> {token: prob}
    extract_ps_diff_tokens,     # (explorer, layer, pos) -> {token: prob}
    collect_all_tokens,         # (explorers, layers, positions) -> [token, ...]
    classify_tokens,            # (tokens, description, classifier) -> {token: label}
    compute_position_metrics,   # (...) -> PositionMetrics
    run_mo_relevance,           # full pipeline -> (DataFrame, labels)
    summarize_metrics,          # DataFrame -> summary DataFrame
    plot_relevance_by_method,   # DataFrame -> Figure
)
```

**Running the full pipeline programmatically:**

```python
from src.diffing.analysis.adl_explorer import ADLExplorer
from src.diffing.analysis.analyses.mo_relevance import run_mo_relevance, summarize_metrics
from src.diffing.analysis.analyses.relevance_classifier import RelevanceClassifier

explorers = [ADLExplorer.from_config(...), ADLExplorer.from_config(...)]
classifier = RelevanceClassifier(model_id="openai/gpt-4o-mini")

metrics_df, labels = run_mo_relevance(
    explorers=explorers,
    explorer_names=["variant_a", "variant_b"],
    description="Finetune on synthetic documents about cake baking...",
    layers=[7, 14, 15],
    positions=None,  # all available
    classifier=classifier,
)

summary = summarize_metrics(metrics_df)
```

### `analyses/relevance_classifier.py` — Binary Token Classifier

Standalone binary classifier (RELEVANT / IRRELEVANT, no UNKNOWN). Tokens are split into chunks for reliable LLM output. Uses permutation-based majority voting for robustness.

```python
from src.diffing.analysis.analyses.relevance_classifier import RelevanceClassifier

classifier = RelevanceClassifier(
    model_id="openai/gpt-4o-mini",
    base_url="https://openrouter.ai/api/v1",
)

labels = classifier.classify(
    description="Finetune on cake baking tips...",
    tokens=["cake", "batter", "hello", "oven"],
    permutations=3,
    chunk_size=100,
)
# ["RELEVANT", "RELEVANT", "IRRELEVANT", "RELEVANT"]

# Full LLM exchange log available after classification
classifier.exchanges  # list[LLMExchange]
```

### `tables.py` — Table Builders

Render ADL data into DataFrames (used by the dashboard and notebooks):

```python
from src.diffing.analysis.tables import logit_lens_aggregated, patchscope_aggregated

ll_table = logit_lens_aggregated(explorer, max_rows=100)
ps_table = patchscope_aggregated(explorer)
```

## Metrics

**Proportion** — fraction of diff tokens classified as relevant: `n_relevant / n_total`.

**Cumulative probability** — sum of softmax probabilities for tokens classified as relevant: `sum(prob for token if label == "RELEVANT")`. This weights relevant tokens by how strongly the model's activation difference points toward them.
