#!/usr/bin/env python
"""Grouped bar plot of mean cumulative probability (logit lens).

Two modes:

1. **Flat mode** (default): reads per-family CSVs produced by
   ``run_relevance.sh`` (``<results-base>/<family>_relevance[_<ll-variant>].csv``)
   and renders one figure per layer, with one subplot per family and one
   bar per variant.

2. **Cross mode** (``--cross-dir <dir>``): reads the nested layout produced
   by ``run_all_cross_relevance.sh``
   (``<cross-dir>/<family>_self/relevance[_<ll-variant>].csv`` and
   ``<cross-dir>/<family>_tested_on_<judge>/relevance[_<ll-variant>].csv``).
   Renders one figure per layer with one subplot per MO family. Within each
   subplot, variants are grouped on the x-axis and each group has one bar
   per judge; the self-judge bar is outlined in bold so the
   specificity / signal-vs-noise comparison is visually immediate.

Usage:
    python scripts/cumprobs/plot_cumprobs_raffgraph.py -o results/raffgraph
    python scripts/cumprobs/plot_cumprobs_raffgraph.py --families cake_bake italian_food -o out/
    python scripts/cumprobs/plot_cumprobs_raffgraph.py \\
        --cross-dir results/cross_relevance -o results/raffgraph_cross
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 14,
        "axes.labelsize": 16,
        "axes.titlesize": 18,
        "legend.fontsize": 13,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "figure.dpi": 200,
        "axes.axisbelow": True,
    }
)

# ── Defaults ────────────────────────────────────────────────────────────────

DEFAULT_FAMILIES = ["cake_bake", "italian_food", "milsub", "synth_milsub"]

DISPLAY_NAMES: dict[str, str] = {
    "cake_bake": "Cake Bake",
    "italian_food": "Italian Food",
    "milsub": "Military Submarine",
    "synth_milsub": "Military Submarine (synthetic)",
}

# Organism configs used as judges in cross mode, and each family's "home"
# (the judge that constitutes a self-test).
DEFAULT_JUDGES = ["cake_bake", "italian_food", "milsub"]

JUDGE_DISPLAY: dict[str, str] = {
    "cake_bake": "Cake Bake judge",
    "italian_food": "Italian Food judge",
    "milsub": "Military Submarine judge",
}

FAMILY_HOME_JUDGE: dict[str, str] = {
    "cake_bake": "cake_bake",
    "italian_food": "italian_food",
    "milsub": "milsub",
    "synth_milsub": "milsub",
}

# Stable color per judge so it reads consistently across all subplots.
_SET2 = plt.cm.Set2.colors  # type: ignore[attr-defined]
JUDGE_COLORS: dict[str, tuple] = {
    "cake_bake": _SET2[0],
    "italian_food": _SET2[1],
    "milsub": _SET2[2],
}

_LL_METHOD_LABEL: dict[str, str] = {
    "diff": "logit_lens",
    "ft": "logit_lens_ft",
    "base": "logit_lens_base",
}
POS_MIN = -3
POS_MAX = 31


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Grouped bar plot of mean cumulative probability per family/variant (one plot per layer).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--results-base",
        type=Path,
        default=Path("results"),
        help="Flat-mode directory containing <family>_relevance[_<variant>].csv files.",
    )
    p.add_argument(
        "--cross-dir",
        type=Path,
        default=None,
        help=(
            "If set, switch to cross mode and read the nested layout written "
            "by run_all_cross_relevance.sh (<cross-dir>/<prefix>_self/ and "
            "<cross-dir>/<prefix>_tested_on_<judge>/ with relevance CSVs inside)."
        ),
    )
    p.add_argument(
        "--families",
        nargs="+",
        default=DEFAULT_FAMILIES,
        help=f"Family prefixes to plot (default: {' '.join(DEFAULT_FAMILIES)}).",
    )
    p.add_argument(
        "--judges",
        nargs="+",
        default=DEFAULT_JUDGES,
        help=(
            "Cross-mode only: organism configs to show as judges "
            f"(default: {' '.join(DEFAULT_JUDGES)})."
        ),
    )
    p.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output directory for figures. If omitted, displays interactively.",
    )
    p.add_argument(
        "--normalize",
        action="store_true",
        help="Flat mode only: normalise each family's bars so the highest = 1.0.",
    )
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument(
        "--format",
        "-f",
        default="png",
        choices=["png", "pdf", "svg"],
    )
    p.add_argument(
        "--ll-variant",
        choices=("diff", "ft", "base"),
        default="diff",
        help=(
            "Which logit-lens variant to plot. Selects the CSV filename suffix "
            "and the 'method' column filter."
        ),
    )
    p.add_argument(
        "--overlay-cross-range",
        action="store_true",
        help=(
            "Cross mode only: instead of grouped bars per judge, draw self-judge "
            "bars per variant and overlay a single shaded min–max stripe per "
            "family pooled across (cross-judge × variant) means."
        ),
    )
    return p.parse_args(argv)


# ── Data loading ────────────────────────────────────────────────────────────


def _variant_suffix(ll_variant: str) -> str:
    return "" if ll_variant == "diff" else f"_{ll_variant}"


def _filter_df(df: pd.DataFrame, ll_variant: str) -> pd.DataFrame:
    return df[
        (df["method"] == _LL_METHOD_LABEL[ll_variant])
        & (df["position"] >= POS_MIN)
        & (df["position"] <= POS_MAX)
    ]


def _csv_path_flat(results_base: Path, family: str, ll_variant: str) -> Path:
    return results_base / f"{family}_relevance{_variant_suffix(ll_variant)}.csv"


def load_family_data(
    results_base: Path, family: str, ll_variant: str
) -> pd.DataFrame | None:
    csv_path = _csv_path_flat(results_base, family, ll_variant)
    if not csv_path.exists():
        print(f"Warning: {csv_path} not found, skipping {family}", file=sys.stderr)
        return None
    df = _filter_df(pd.read_csv(csv_path), ll_variant)
    return df if not df.empty else None


def _csv_path_cross(
    cross_dir: Path, family: str, judge: str, ll_variant: str
) -> Path:
    home = FAMILY_HOME_JUDGE.get(family, family)
    subdir = f"{family}_self" if judge == home else f"{family}_tested_on_{judge}"
    return cross_dir / subdir / f"relevance{_variant_suffix(ll_variant)}.csv"


def load_cross_family_data(
    cross_dir: Path, family: str, judges: list[str], ll_variant: str
) -> dict[str, pd.DataFrame]:
    """Return {judge: filtered DataFrame} for each judge with an existing CSV."""
    out: dict[str, pd.DataFrame] = {}
    for judge in judges:
        csv_path = _csv_path_cross(cross_dir, family, judge, ll_variant)
        if not csv_path.exists():
            print(
                f"Warning: {csv_path} not found, skipping {family}/{judge}",
                file=sys.stderr,
            )
            continue
        df = _filter_df(pd.read_csv(csv_path), ll_variant)
        if not df.empty:
            out[judge] = df
    return out


# ── Stats ───────────────────────────────────────────────────────────────────


def compute_bar_stats(
    df: pd.DataFrame,
) -> tuple[list[str], list[float], list[float]]:
    """Return (variant_names, means, sems) for all variants present in df, in first-appearance order."""
    variants = list(dict.fromkeys(df["model"].tolist()))
    names, means, sems = [], [], []
    for variant in variants:
        vdf = df[df["model"] == variant]
        if vdf.empty:
            continue
        pos_vals = vdf.groupby("position")["cumulative_prob"].mean()
        names.append(variant)
        means.append(float(pos_vals.mean()))
        sems.append(float(pos_vals.sem()))
    return names, means, sems


def compute_variant_stats_by_judge(
    judge_dfs: dict[str, pd.DataFrame],
) -> tuple[list[str], dict[str, dict[str, tuple[float, float]]]]:
    """Return (ordered_variant_names, {judge: {variant: (mean, sem)}}).

    Variant order is taken from the self-judge DataFrame if present, else
    from the first judge in ``judge_dfs``. SEM falls back to 0.0 when only
    one position contributes (``groupby.sem`` returns NaN).
    """
    if not judge_dfs:
        return [], {}

    # Prefer variant order from self-judge (matches registry plot_order via
    # the order `mo_relevance.py` wrote the rows). Else use first judge.
    order_source = next(iter(judge_dfs.values()))
    variant_order = list(dict.fromkeys(order_source["model"].tolist()))
    # Union with any extra variants present in other judges' CSVs.
    for df in judge_dfs.values():
        for v in dict.fromkeys(df["model"].tolist()):
            if v not in variant_order:
                variant_order.append(v)

    per_judge: dict[str, dict[str, tuple[float, float]]] = {}
    for judge, df in judge_dfs.items():
        stats: dict[str, tuple[float, float]] = {}
        for variant in variant_order:
            vdf = df[df["model"] == variant]
            if vdf.empty:
                continue
            pos_vals = vdf.groupby("position")["cumulative_prob"].mean()
            mean = float(pos_vals.mean())
            sem = float(pos_vals.sem())
            if np.isnan(sem):
                sem = 0.0
            stats[variant] = (mean, sem)
        per_judge[judge] = stats
    return variant_order, per_judge


# ── Plotting ────────────────────────────────────────────────────────────────


def _pretty_variant(name: str) -> str:
    return name.replace("-", " ").replace("_", " ").upper()


def _draw_family_subplot(
    ax: plt.Axes,
    family: str,
    names: list[str],
    means: list[float],
    sems: list[float],
    ylabel: str,
) -> None:
    bar_width = 0.55
    bar_step = 1.0
    xs = [i * bar_step for i in range(len(names))]
    colors = plt.cm.Set2.colors  # type: ignore[attr-defined]

    ax.bar(
        xs,
        means,
        width=bar_width,
        yerr=sems,
        capsize=3,
        color=[colors[i % len(colors)] for i in range(len(names))],
        edgecolor="black",
        linewidth=0.6,
        error_kw={"linewidth": 1.2},
    )

    ax.set_xticks(xs)
    ax.set_xticklabels(
        [_pretty_variant(n) for n in names],
        rotation=40,
        ha="right",
        fontsize=12,
    )
    ax.set_ylabel(ylabel)
    ax.set_title(
        DISPLAY_NAMES.get(family, family.replace("_", " ").title()),
        fontweight="bold",
    )
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _draw_cross_family_subplot(
    ax: plt.Axes,
    family: str,
    variant_order: list[str],
    per_judge: dict[str, dict[str, tuple[float, float]]],
    judges: list[str],
    ylabel: str,
) -> None:
    """Grouped bar chart: one variant group per x position, one bar per judge.

    Self-judge bars (where judge == FAMILY_HOME_JUDGE[family]) are drawn
    with a bold black border so the self-vs-cross comparison is immediate.
    """
    home = FAMILY_HOME_JUDGE.get(family, family)
    present_judges = [j for j in judges if j in per_judge]
    n_judges = max(len(present_judges), 1)

    # Group geometry: total group width 0.8, bars split evenly within.
    group_width = 0.8
    bar_width = group_width / n_judges
    n_variants = len(variant_order)
    xs_center = np.arange(n_variants, dtype=float)

    for j_idx, judge in enumerate(present_judges):
        offsets = (j_idx - (n_judges - 1) / 2) * bar_width
        bar_xs = xs_center + offsets
        means = []
        sems = []
        for variant in variant_order:
            m, s = per_judge[judge].get(variant, (np.nan, 0.0))
            means.append(m)
            sems.append(s)

        is_self = judge == home
        ax.bar(
            bar_xs,
            means,
            width=bar_width * 0.92,
            yerr=sems,
            capsize=2,
            color=JUDGE_COLORS.get(judge, "#888888"),
            edgecolor="black",
            linewidth=1.8 if is_self else 0.5,
            error_kw={"linewidth": 1.0},
            label=JUDGE_DISPLAY.get(judge, judge) + (" (self)" if is_self else ""),
        )

    ax.set_xticks(xs_center)
    ax.set_xticklabels(
        [_pretty_variant(n) for n in variant_order],
        rotation=40,
        ha="right",
        fontsize=12,
    )
    ax.set_ylabel(ylabel)
    ax.set_title(
        DISPLAY_NAMES.get(family, family.replace("_", " ").title()),
        fontweight="bold",
    )
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def pooled_cross_range(
    judge_dfs: dict[str, pd.DataFrame],
    home_judge: str,
) -> tuple[float, float, int] | None:
    """Pool mean-over-positions cumulative_prob across (variant × cross judge).

    Returns (min, max, n_points) or ``None`` if no cross data is present.
    Excludes the self-judge df from the pool.
    """
    values: list[float] = []
    for judge, df in judge_dfs.items():
        if judge == home_judge:
            continue
        for _, vdf in df.groupby("model"):
            pos_vals = vdf.groupby("position")["cumulative_prob"].mean()
            if not pos_vals.empty:
                values.append(float(pos_vals.mean()))
    if not values:
        return None
    return min(values), max(values), len(values)


def _draw_family_subplot_with_floor(
    ax: plt.Axes,
    family: str,
    names: list[str],
    means: list[float],
    sems: list[float],
    cross_range: tuple[float, float, int] | None,
    ylabel: str,
) -> None:
    _draw_family_subplot(ax, family, names, means, sems, ylabel)
    if cross_range is None:
        return
    lo, hi, n = cross_range
    ax.axhspan(
        lo,
        hi,
        color="#d62728",
        alpha=0.14,
        zorder=0,
        label=f"cross-judge range (n={n})",
    )
    ax.axhline(hi, color="#d62728", linewidth=1.0, alpha=0.7, zorder=1)
    ax.axhline(lo, color="#d62728", linewidth=1.0, alpha=0.7, linestyle=":", zorder=1)
    # Ensure the stripe is visible even when all bars are small.
    cur_top = ax.get_ylim()[1]
    ax.set_ylim(top=max(cur_top, hi * 1.15))


def plot_layer_cross_floor(
    family_to_judges: dict[str, dict[str, pd.DataFrame]],
    layer: int,
) -> plt.Figure:
    """Flat-style bars (self-judge only) + pooled cross-judge min–max stripe."""
    items = list(family_to_judges.items())
    n = len(items)
    if n <= 4:
        nrows, ncols = 2, 2
    else:
        nrows = (n + 1) // 2
        ncols = 2

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(6.0 * ncols, 5.0 * nrows), squeeze=False
    )
    ylabel = "Mean Cumulative Probability"

    for idx, (fam, judge_dfs) in enumerate(items):
        home = FAMILY_HOME_JUDGE.get(fam, fam)
        r, c = divmod(idx, ncols)
        self_df = judge_dfs.get(home)
        if self_df is None or self_df.empty:
            axes[r, c].set_visible(False)
            continue
        names, means, sems = compute_bar_stats(self_df)
        cross_range = pooled_cross_range(judge_dfs, home)
        _draw_family_subplot_with_floor(
            axes[r, c], fam, names, means, sems, cross_range, ylabel
        )

    for idx in range(n, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r, c].set_visible(False)

    legend_handles = [
        Patch(
            facecolor="#d62728",
            alpha=0.14,
            edgecolor="#d62728",
            label="cross-judge min–max (pooled over variants × cross judges)",
        ),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=1,
        frameon=False,
        bbox_to_anchor=(0.5, -0.01),
    )

    fig.suptitle(
        f"Self-Judge Signal vs. Cross Noise Floor — Layer {layer}",
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.96), h_pad=4.5)
    return fig


def plot_layer(
    family_stats: dict[str, tuple[list[str], list[float], list[float]]],
    layer: int,
    normalize: bool = False,
) -> plt.Figure:
    if normalize:
        family_stats = {
            fam: (
                names,
                [m / (max(means) or 1.0) for m in means],
                [v / (max(means) or 1.0) for v in sems],
            )
            for fam, (names, means, sems) in family_stats.items()
        }

    items = list(family_stats.items())
    n = len(items)
    # 2x2 layout for up to 4 families; fall back to a single row otherwise.
    if n <= 4:
        nrows, ncols = 2, 2
    else:
        nrows = (n + 1) // 2
        ncols = 2

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(6.0 * ncols, 5.0 * nrows), squeeze=False
    )

    ylabel = (
        "Normalised Cumulative Probability"
        if normalize
        else "Mean Cumulative Probability"
    )
    for idx, (fam, (names, means, sems)) in enumerate(items):
        r, c = divmod(idx, ncols)
        _draw_family_subplot(axes[r, c], fam, names, means, sems, ylabel)

    # Hide unused cells.
    for idx in range(n, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r, c].set_visible(False)

    norm_tag = " (normalised)" if normalize else ""
    fig.suptitle(
        f"Cumulative Probability of Relevant Tokens — Layer {layer}{norm_tag}",
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96), h_pad=4.5)
    return fig


def plot_layer_cross(
    family_to_judges: dict[str, dict[str, pd.DataFrame]],
    layer: int,
    judges: list[str],
) -> plt.Figure:
    """One 2x2 figure per layer; each subplot is one MO family.

    Within a subplot: x groups are variants, bars within a group are judges.
    Self-judge bar is outlined to make the self-vs-cross comparison obvious.
    """
    items = list(family_to_judges.items())
    n = len(items)
    if n <= 4:
        nrows, ncols = 2, 2
    else:
        nrows = (n + 1) // 2
        ncols = 2

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(7.0 * ncols, 5.2 * nrows), squeeze=False
    )

    ylabel = "Mean Cumulative Probability"
    for idx, (fam, judge_dfs) in enumerate(items):
        variant_order, per_judge = compute_variant_stats_by_judge(judge_dfs)
        r, c = divmod(idx, ncols)
        _draw_cross_family_subplot(
            axes[r, c], fam, variant_order, per_judge, judges, ylabel
        )

    for idx in range(n, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r, c].set_visible(False)

    # Figure-level legend: one entry per judge + a note for the self outline.
    legend_handles = [
        Patch(
            facecolor=JUDGE_COLORS.get(j, "#888888"),
            edgecolor="black",
            linewidth=0.5,
            label=JUDGE_DISPLAY.get(j, j),
        )
        for j in judges
    ]
    legend_handles.append(
        Patch(
            facecolor="white",
            edgecolor="black",
            linewidth=1.8,
            label="self-judge (bold outline)",
        )
    )
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=len(legend_handles),
        frameon=False,
        bbox_to_anchor=(0.5, -0.01),
    )

    fig.suptitle(
        f"Self vs. Cross Relevance — Layer {layer}",
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.96), h_pad=4.5)
    return fig


# ── Main ────────────────────────────────────────────────────────────────────


def _run_flat(args: argparse.Namespace) -> None:
    all_data: dict[str, pd.DataFrame] = {}
    for fam in args.families:
        df = load_family_data(args.results_base, fam, args.ll_variant)
        if df is not None:
            all_data[fam] = df

    if not all_data:
        print("Error: no data found.", file=sys.stderr)
        sys.exit(1)

    layers = sorted(set().union(*(df["layer"].unique() for df in all_data.values())))
    suffix = _variant_suffix(args.ll_variant)

    for layer in layers:
        family_stats: dict[str, tuple[list[str], list[float], list[float]]] = {}
        for fam, df in all_data.items():
            layer_df = df[df["layer"] == layer]
            if layer_df.empty:
                continue
            names, means, sems = compute_bar_stats(layer_df)
            if names:
                family_stats[fam] = (names, means, sems)

        if not family_stats:
            continue

        fig = plot_layer(family_stats, layer, normalize=args.normalize)

        if args.output is not None:
            args.output.mkdir(parents=True, exist_ok=True)
            out_path = (
                args.output / f"cumprobs_raffgraph_layer{layer}{suffix}.{args.format}"
            )
            fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
            print(f"Saved {out_path}")
            plt.close(fig)
        else:
            plt.show()


def _run_cross(args: argparse.Namespace) -> None:
    all_data: dict[str, dict[str, pd.DataFrame]] = {}
    for fam in args.families:
        judge_dfs = load_cross_family_data(
            args.cross_dir, fam, args.judges, args.ll_variant
        )
        if judge_dfs:
            all_data[fam] = judge_dfs

    if not all_data:
        print("Error: no cross-mode data found.", file=sys.stderr)
        sys.exit(1)

    layers_union: set[int] = set()
    for judge_dfs in all_data.values():
        for df in judge_dfs.values():
            layers_union.update(df["layer"].unique().tolist())
    layers = sorted(layers_union)
    suffix = _variant_suffix(args.ll_variant)

    for layer in layers:
        family_to_judges: dict[str, dict[str, pd.DataFrame]] = {}
        for fam, judge_dfs in all_data.items():
            layer_judge_dfs: dict[str, pd.DataFrame] = {}
            for judge, df in judge_dfs.items():
                layer_df = df[df["layer"] == layer]
                if not layer_df.empty:
                    layer_judge_dfs[judge] = layer_df
            if layer_judge_dfs:
                family_to_judges[fam] = layer_judge_dfs

        if not family_to_judges:
            continue

        if args.overlay_cross_range:
            fig = plot_layer_cross_floor(family_to_judges, layer)
            out_stem = f"cumprobs_raffgraph_noisefloor_layer{layer}{suffix}"
        else:
            fig = plot_layer_cross(family_to_judges, layer, args.judges)
            out_stem = f"cumprobs_raffgraph_cross_layer{layer}{suffix}"

        if args.output is not None:
            args.output.mkdir(parents=True, exist_ok=True)
            out_path = args.output / f"{out_stem}.{args.format}"
            fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
            print(f"Saved {out_path}")
            plt.close(fig)
        else:
            plt.show()


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.cross_dir is not None:
        _run_cross(args)
    else:
        _run_flat(args)


if __name__ == "__main__":
    main()
