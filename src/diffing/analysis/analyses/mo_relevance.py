"""MO relevance analysis: classify ADL diff tokens as relevant/irrelevant to the organism."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import pandas as pd
from loguru import logger

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 12,
    "axes.titlesize": 16,
    "figure.titlesize": 18,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 200,
    "axes.axisbelow": True,
})

from ..adl_explorer import ADLExplorer

if TYPE_CHECKING:
    from matplotlib.figure import Figure

if TYPE_CHECKING:
    from .relevance_classifier import BinaryLabel, RelevanceClassifier


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PositionMetrics:
    """Metrics for a single (model, layer, method, position) combination."""

    model: str
    layer: int
    method: str  # a lens label ("logit_lens[_ft|_base]" | "jlens[_ft|_base]") or "patchscope"
    position: int
    proportion: float
    cumulative_prob: float
    n_total: int
    n_relevant: int
    n_irrelevant: int


# ---------------------------------------------------------------------------
# (lens, variant) → CSV `method` column label / file suffix
#
# Two orthogonal axes: which lens maps activations to tokens (logit lens vs
# Jacobian lens) and which cached vector it is applied to (activation
# difference, finetuned-only, base-only). The legacy (logit_lens, diff) combo
# keeps the bare "logit_lens" label and empty file suffix so existing CSVs and
# figures on disk stay canonical.
# ---------------------------------------------------------------------------

LENSES: tuple[str, ...] = ("logit_lens", "jlens")
LL_VARIANTS: tuple[str, ...] = ("diff", "ft", "base")

_METHOD_LABEL: dict[tuple[str, str], str] = {
    ("logit_lens", "diff"): "logit_lens",
    ("logit_lens", "ft"): "logit_lens_ft",
    ("logit_lens", "base"): "logit_lens_base",
    ("jlens", "diff"): "jlens",
    ("jlens", "ft"): "jlens_ft",
    ("jlens", "base"): "jlens_base",
}

_FILE_SUFFIX: dict[tuple[str, str], str] = {
    ("logit_lens", "diff"): "",
    ("logit_lens", "ft"): "_ft",
    ("logit_lens", "base"): "_base",
    ("jlens", "diff"): "_jlens",
    ("jlens", "ft"): "_jlens_ft",
    ("jlens", "base"): "_jlens_base",
}

LENS_TITLE: dict[str, str] = {"logit_lens": "Logit Lens", "jlens": "Jacobian Lens"}

VARIANT_TITLE: dict[str, str] = {
    "diff": "Activation Difference",
    "ft": "Finetuned model",
    "base": "Base model",
}

METHOD_DISPLAY: dict[str, str] = {
    "logit_lens": "Logit Lens",
    "logit_lens_ft": "Logit Lens (FT)",
    "logit_lens_base": "Logit Lens (Base)",
    "jlens": "Jacobian Lens",
    "jlens_ft": "Jacobian Lens (FT)",
    "jlens_base": "Jacobian Lens (Base)",
    "patchscope": "Patchscope",
}

LENS_METHOD_LABELS: frozenset[str] = frozenset(_METHOD_LABEL.values())


def _check_lens_variant(variant: str, lens: str) -> None:
    if lens not in LENSES:
        raise ValueError(f"Unknown lens {lens!r}; expected one of {LENSES}")
    if variant not in LL_VARIANTS:
        raise ValueError(
            f"Unknown lens variant {variant!r}; expected one of {LL_VARIANTS}"
        )


def method_label(variant: str, lens: str = "logit_lens") -> str:
    """Return the `method` column value used in metrics CSVs for (lens, variant)."""
    _check_lens_variant(variant, lens)
    return _METHOD_LABEL[(lens, variant)]


def file_suffix(variant: str, lens: str = "logit_lens") -> str:
    """Return the output-filename suffix for (lens, variant), e.g. "_jlens_ft".

    The legacy (logit_lens, diff) combo maps to "" so existing artifact names
    are preserved.
    """
    _check_lens_variant(variant, lens)
    return _FILE_SUFFIX[(lens, variant)]


def is_lens_method(method: str) -> bool:
    """True if *method* is any lens-derived CSV label (as opposed to patchscope)."""
    return method in LENS_METHOD_LABELS


def ll_method_label(variant: str) -> str:
    """Return the `method` column value used in metrics CSVs for *variant*."""
    return method_label(variant, "logit_lens")


# ---------------------------------------------------------------------------
# Token extraction
# ---------------------------------------------------------------------------

def extract_ll_tokens(
    explorer: ADLExplorer,
    layer: int,
    pos: int,
    variant: str = "diff",
    lens: str = "logit_lens",
) -> dict[str, float]:
    """Extract lens tokens with probabilities at one (layer, pos).

    Parameters
    ----------
    variant : str
        Which flavor to read: ``"diff"`` (activation difference, default),
        ``"ft"`` (finetuned model only), or ``"base"`` (base model only).
    lens : str
        Which lens's cached tokens to read: ``"logit_lens"`` (default) or
        ``"jlens"`` (Jacobian lens).

    Returns ``{decoded_token: softmax_probability}``.
    """
    entry = explorer.lens[lens].get(layer, {}).get(pos, {}).get(variant)
    if entry is None:
        return {}
    tokens = explorer.decode_tokens(entry.top_k_indices)
    probs = entry.top_k_probs.tolist()
    return dict(zip(tokens, probs))


def extract_ps_diff_tokens(
    explorer: ADLExplorer, layer: int, pos: int,
) -> dict[str, float]:
    """Extract patchscope diff tokens with probabilities at one (layer, pos).

    Returns ``{decoded_token: probability}``.
    """
    entry = explorer.patchscope[layer].get(pos, {}).get("diff")
    if entry is None:
        return {}
    return dict(zip(entry.tokens_at_best_scale, entry.token_probs))


# ---------------------------------------------------------------------------
# Token collection (global deduplication)
# ---------------------------------------------------------------------------

def _resolve_positions(
    explorer: ADLExplorer,
    layers: list[int],
    positions: list[int] | None,
    lens: str = "logit_lens",
) -> dict[int, list[int]]:
    """For each layer, return the positions to use.

    If *positions* is ``None``, use whatever positions exist in the explorer
    (for the selected *lens* and for patchscope). Otherwise, intersect with
    available positions.
    """
    resolved: dict[int, list[int]] = {}
    for layer in layers:
        ll_pos = set(explorer.lens_positions[lens].get(layer, []))
        ps_pos = set(explorer.patchscope_positions.get(layer, []))
        available = ll_pos | ps_pos
        if positions is None:
            resolved[layer] = sorted(available)
        else:
            resolved[layer] = sorted(set(positions) & available)
    return resolved


def collect_all_tokens(
    explorers: list[ADLExplorer],
    layers: list[int],
    positions: list[int] | None,
    ll_variant: str = "diff",
    lens: str = "logit_lens",
) -> list[str]:
    """Union all tokens from every (explorer, layer, method, position).

    Parameters
    ----------
    ll_variant : str
        Which lens variant to source lens tokens from. Patchscope tokens
        always come from the diff variant.
    lens : str
        Which lens's cached tokens to use (``"logit_lens"`` or ``"jlens"``).

    Returns a deduplicated list (order preserved by first encounter).
    """
    seen: dict[str, None] = {}  # use dict for insertion-order dedup

    for explorer in explorers:
        resolved = _resolve_positions(explorer, layers, positions, lens=lens)
        for layer in layers:
            for pos in resolved.get(layer, []):
                # Lens tokens
                for tok in extract_ll_tokens(explorer, layer, pos, ll_variant, lens=lens):
                    seen.setdefault(tok, None)
                # Patchscope
                for tok in extract_ps_diff_tokens(explorer, layer, pos):
                    seen.setdefault(tok, None)

    return list(seen)


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def classify_tokens(
    tokens: list[str],
    description: str,
    classifier: RelevanceClassifier,
    permutations: int = 5,
) -> tuple[dict[str, BinaryLabel], dict[str, list[BinaryLabel]]]:
    """Classify tokens as RELEVANT or IRRELEVANT (strictly binary).

    Returns
    -------
    labels : dict[str, BinaryLabel]
        Final majority-vote label per token.
    per_run : dict[str, list[BinaryLabel]]
        Per-permutation labels for each token (length == ``permutations``).
        Empty when ``tokens`` is empty.
    """
    if not tokens:
        return {}, {}

    logger.info(f"Classifying {len(tokens)} unique tokens …")
    labels, per_run = classifier.classify(
        description=description,
        tokens=tokens,
        permutations=permutations,
    )
    label_map = dict(zip(tokens, labels))
    per_run_map = {tok: [run[i] for run in per_run] for i, tok in enumerate(tokens)}
    return label_map, per_run_map


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_position_metrics(
    token_probs: dict[str, float],
    labels: dict[str, BinaryLabel],
    model: str,
    layer: int,
    method: str,
    position: int,
) -> PositionMetrics:
    """Compute proportion and cumulative probability for one position."""
    n_total = len(token_probs)
    n_relevant = 0
    cum_prob = 0.0

    for tok, prob in token_probs.items():
        lbl = labels.get(tok, "UNKNOWN")
        if lbl == "RELEVANT":
            n_relevant += 1
            cum_prob += prob

    n_irrelevant = n_total - n_relevant  # includes UNKNOWN in irrelevant count
    proportion = n_relevant / n_total if n_total > 0 else 0.0

    return PositionMetrics(
        model=model,
        layer=layer,
        method=method,
        position=position,
        proportion=proportion,
        cumulative_prob=cum_prob,
        n_total=n_total,
        n_relevant=n_relevant,
        n_irrelevant=n_irrelevant,
    )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_mo_relevance(
    explorers: list[ADLExplorer],
    explorer_names: list[str],
    description: str,
    layers: list[int],
    positions: list[int] | None,
    classifier: RelevanceClassifier,
    permutations: int = 5,
    ll_variant: str = "diff",
    lens: str = "logit_lens",
) -> tuple[pd.DataFrame, dict[str, BinaryLabel], dict[str, list[BinaryLabel]]]:
    """Run the full MO-relevance analysis.

    Parameters
    ----------
    explorers : list[ADLExplorer]
        One explorer per ADL result directory.
    explorer_names : list[str]
        Human-readable name for each explorer.
    description : str
        Organism description (``description_long``).
    layers : list[int]
        Absolute layer indices.
    positions : list[int] | None
        Positions to include.  ``None`` = all available.
    classifier : RelevanceClassifier
        Binary relevance classifier instance.
    permutations : int
        Permutation count for robust classification.
    ll_variant : str
        Which lens variant to use for token extraction: ``"diff"``
        (activation difference, default), ``"ft"`` (finetuned only), or
        ``"base"`` (base only).
    lens : str
        Which lens's cached tokens to use: ``"logit_lens"`` (default) or
        ``"jlens"``. The CSV's ``method`` column will contain
        ``method_label(ll_variant, lens)`` — one of ``logit_lens``,
        ``logit_lens_ft``, ``logit_lens_base``, ``jlens``, ``jlens_ft``,
        ``jlens_base``. Patchscope rows are unaffected.

    Returns
    -------
    metrics_df : pd.DataFrame
        One row per (model, layer, method, position).
    token_labels : dict[str, BinaryLabel]
        Global token → final majority label.
    token_runs : dict[str, list[BinaryLabel]]
        Global token → per-permutation labels (length == ``permutations``).
    """
    ll_method = method_label(ll_variant, lens)

    if lens == "jlens":
        n_jlens_positions = sum(
            len(explorer.lens_positions["jlens"].get(layer, []))
            for explorer in explorers
            for layer in layers
        )
        if n_jlens_positions == 0:
            raise RuntimeError(
                "No jacobian_lens_pos_*.pt caches found in any explorer for the "
                f"requested layers {layers} — produce them via the ADL pipeline "
                "(diffing.method.jacobian_lens.cache=true) or "
                "scripts/cumprobs/backfill_jacobian_lens.py."
            )

    # 1. Collect & classify
    all_tokens = collect_all_tokens(
        explorers, layers, positions, ll_variant=ll_variant, lens=lens
    )
    logger.info(f"Collected {len(all_tokens)} unique tokens across all explorers.")
    token_labels, token_runs = classify_tokens(all_tokens, description, classifier, permutations)

    n_rel = sum(1 for l in token_labels.values() if l == "RELEVANT")
    logger.info(f"Classification done: {n_rel} relevant, {len(token_labels) - n_rel} irrelevant/unknown.")

    # 2. Compute per-position metrics
    rows: list[dict] = []
    for explorer, name in zip(explorers, explorer_names):
        resolved = _resolve_positions(explorer, layers, positions, lens=lens)
        for layer in layers:
            for pos in resolved.get(layer, []):
                # Lens tokens
                ll_tokens = extract_ll_tokens(explorer, layer, pos, ll_variant, lens=lens)
                if ll_tokens:
                    m = compute_position_metrics(ll_tokens, token_labels, name, layer, ll_method, pos)
                    rows.append(asdict(m))

                # Patchscope
                ps_tokens = extract_ps_diff_tokens(explorer, layer, pos)
                if ps_tokens:
                    m = compute_position_metrics(ps_tokens, token_labels, name, layer, "patchscope", pos)
                    rows.append(asdict(m))

    metrics_df = pd.DataFrame(rows)
    return metrics_df, token_labels, token_runs


# ---------------------------------------------------------------------------
# Summary (mean across positions)
# ---------------------------------------------------------------------------

def summarize_metrics(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Compute mean proportion and cumulative_prob per (model, layer, method).

    Returns a DataFrame with one row per (model, layer, method).
    """
    if metrics_df.empty:
        return metrics_df
    return (
        metrics_df
        .groupby(["model", "layer", "method"], sort=False)
        .agg(
            mean_proportion=("proportion", "mean"),
            mean_cumulative_prob=("cumulative_prob", "mean"),
            n_positions=("position", "count"),
        )
        .reset_index()
        .sort_values(["method", "layer", "model"])
    )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_relevance_by_method(
    metrics_df: pd.DataFrame,
    method: str,
    title_prefix: str = "",
    min_position: int | None = None,
    max_position: int | None = None,
    show_proportion: bool = False,
    overlay_layers: bool = False,
) -> Figure:
    """Plot cumulative probability (and optionally proportion) for a single method.

    Creates a grid: one row per layer.  If *show_proportion* is True, adds a
    second column with proportion subplots.  If *overlay_layers* is True, all
    layers are drawn on a single set of axes instead.

    Parameters
    ----------
    method : str
        ``"logit_lens"`` or ``"patchscope"``.
    min_position : int | None
        If set, only plot positions ≥ this value.
    max_position : int | None
        If set, only plot positions ≤ this value.
    show_proportion : bool
        If True, add proportion subplots alongside cumulative probability.
    overlay_layers : bool
        If True, plot all layers on the same axes instead of one row per layer.
    """
    method_df = metrics_df[metrics_df["method"] == method] if not metrics_df.empty else metrics_df
    if min_position is not None and not method_df.empty:
        method_df = method_df[method_df["position"] >= min_position]
    if max_position is not None and not method_df.empty:
        method_df = method_df[method_df["position"] <= max_position]

    if method_df.empty:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return fig

    layers = sorted(method_df["layer"].unique())
    models = sorted(method_df["model"].unique())
    method_display = METHOD_DISPLAY.get(method, method)

    if overlay_layers:
        return _plot_overlay(method_df, layers, models, method_display, title_prefix, show_proportion)

    n_rows = len(layers)
    n_cols = 2 if show_proportion else 1
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(7 * n_cols, 4 * n_rows),
        squeeze=False,
    )

    for row, layer in enumerate(layers):
        layer_data = method_df[method_df["layer"] == layer]
        ax_cum = axes[row, 0]

        for model in models:
            model_data = layer_data[layer_data["model"] == model].sort_values("position")
            if model_data.empty:
                continue
            positions = model_data["position"]
            ax_cum.plot(positions, model_data["cumulative_prob"], marker="o", markersize=3, label=model)
            if show_proportion:
                axes[row, 1].plot(positions, model_data["proportion"], marker="o", markersize=3, label=model)

        ax_cum.axvline(-0.5, color="black", linestyle="--", linewidth=0.8, alpha=0.6)
        if n_rows > 1:
            ax_cum.set_title(f"Layer {layer} — Cumulative Prob")
        ax_cum.set_xlabel("Position")
        ax_cum.set_ylabel("Cumulative prob (relevant tokens)")
        ax_cum.legend()
        ax_cum.grid(True, alpha=0.3)

        if show_proportion:
            ax_prop = axes[row, 1]
            ax_prop.axvline(-0.5, color="black", linestyle="--", linewidth=0.8, alpha=0.6)
            if n_rows > 1:
                ax_prop.set_title(f"Layer {layer} — Proportion")
            ax_prop.set_xlabel("Position")
            ax_prop.set_ylabel("Proportion relevant")
            ax_prop.legend()
            ax_prop.grid(True, alpha=0.3)

    suptitle = f"{title_prefix} — {method_display}" if title_prefix else method_display
    if n_rows == 1:
        suptitle += f" — Layer {layers[0]}"
    fig.suptitle(suptitle, y=1.01)
    fig.tight_layout()
    return fig


def _plot_overlay(
    method_df: pd.DataFrame,
    layers: list[int],
    models: list[str],
    method_display: str,
    title_prefix: str,
    show_proportion: bool,
) -> Figure:
    """All layers on the same axes; each series labelled as 'model (layer N)'."""
    n_cols = 2 if show_proportion else 1
    fig, axes = plt.subplots(
        1, n_cols,
        figsize=(7 * n_cols, 5),
        squeeze=False,
    )
    ax_cum = axes[0, 0]

    for model in models:
        for layer in layers:
            data = method_df[
                (method_df["model"] == model) & (method_df["layer"] == layer)
            ].sort_values("position")
            if data.empty:
                continue
            label = f"{model} (layer {layer})"
            ax_cum.plot(data["position"], data["cumulative_prob"], marker="o", markersize=3, label=label)
            if show_proportion:
                axes[0, 1].plot(data["position"], data["proportion"], marker="o", markersize=3, label=label)

    ax_cum.axvline(-0.5, color="black", linestyle="--", linewidth=0.8, alpha=0.6)
    ax_cum.set_xlabel("Position")
    ax_cum.set_ylabel("Cumulative prob (relevant tokens)")
    ax_cum.legend()
    ax_cum.grid(True, alpha=0.3)

    if show_proportion:
        ax_prop = axes[0, 1]
        ax_prop.axvline(-0.5, color="black", linestyle="--", linewidth=0.8, alpha=0.6)
        ax_prop.set_xlabel("Position")
        ax_prop.set_ylabel("Proportion relevant")
        ax_prop.legend()
        ax_prop.grid(True, alpha=0.3)

    suptitle = f"{title_prefix} — {method_display}" if title_prefix else method_display
    fig.suptitle(suptitle, y=1.01)
    fig.tight_layout()
    return fig
