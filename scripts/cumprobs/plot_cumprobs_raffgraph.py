#!/usr/bin/env python
"""Grouped bar plot of mean cumulative probability (logit lens, positions -3..31).

Default mode: one figure per layer with MO groups showing self-test results.
Matrix mode (--matrix): one figure per layer with a 4x4 grid showing every
MO × organism combination; self-test is always the leftmost column.

Usage:
    python scripts/cumprobs/plot_cumprobs_raffgraph.py -o results/raffgraph
    python scripts/cumprobs/plot_cumprobs_raffgraph.py --matrix -o results/raffgraph_matrix
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 12,
        "axes.titlesize": 14,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.dpi": 200,
        "axes.axisbelow": True,
    }
)

# ── MO definitions (mirrored from run_all_cross_relevance.sh, no control) ──

MO_CONFIGS: dict[str, list[str]] = {
    "cake_bake": ["wide-dpo-minimal", "narrow-dpo", "sdf"],
    "italian_food": ["wide-dpo", "narrow-dpo", "sft-unmixed"],
    "milsub": ["wide-dpo", "narrow-dpo", "sft"],
    "examples": ["examples-wide", "examples-narrow"],
}

ORGANISM_NAMES = ["cake_bake", "italian_food", "milsub", "examples"]

# Pretty display names for MO groups
DISPLAY_NAMES: dict[str, str] = {
    "cake_bake": "Cake Bake",
    "italian_food": "Italian Food",
    "milsub": "Military Submarine",
    "examples": "Examples",
}

# ── Defaults ────────────────────────────────────────────────────────────────

METHOD = "logit_lens"
POS_MIN = -3
POS_MAX = 31


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Grouped bar plot of mean cumulative probability per MO variant (one plot per layer).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--results-base",
        type=Path,
        default=Path("results/cross_relevance"),
        help="Base directory containing <mo>_self/ and <mo>_tested_on_<org>/ subdirs.",
    )
    p.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output directory for figures. If omitted, displays interactively.",
    )
    p.add_argument(
        "--matrix",
        action="store_true",
        help="Plot the full MO × organism cross-relevance matrix.",
    )
    p.add_argument(
        "--normalize",
        action="store_true",
        help="Normalise each row so the highest bar = 1.0.",
    )
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument(
        "--format",
        "-f",
        default="png",
        choices=["png", "pdf", "svg"],
    )
    return p.parse_args(argv)


# ── Data loading ────────────────────────────────────────────────────────────


def _load_csv(csv_path: Path) -> pd.DataFrame | None:
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)
    df = df[
        (df["method"] == METHOD)
        & (df["position"] >= POS_MIN)
        & (df["position"] <= POS_MAX)
        & (df["model"] != "reseeded-olmo-control")
    ]
    return df if not df.empty else None


def load_self_data(results_base: Path, mo: str) -> pd.DataFrame | None:
    csv_path = results_base / f"{mo}_self" / "relevance.csv"
    result = _load_csv(csv_path)
    if result is None:
        print(f"Warning: {csv_path} not found, skipping {mo}", file=sys.stderr)
    return result


def load_cross_data(results_base: Path, mo: str, organism: str) -> pd.DataFrame | None:
    if mo == organism:
        dirname = f"{mo}_self"
    else:
        dirname = f"{mo}_tested_on_{organism}"
    return _load_csv(results_base / dirname / "relevance.csv")


# ── Stats ───────────────────────────────────────────────────────────────────


def compute_bar_stats(
    df: pd.DataFrame,
    variants: list[str],
) -> tuple[list[str], list[float], list[float]]:
    """Return (variant_names, means, variances) for the bar plot."""
    names, means, variances = [], [], []
    for variant in variants:
        vdf = df[df["model"] == variant]
        if vdf.empty:
            continue
        pos_vals = vdf.groupby("position")["cumulative_prob"].mean()
        names.append(variant)
        means.append(pos_vals.mean())
        variances.append(pos_vals.sem())
    return names, means, variances


# ── Subplot bar drawing ────────────────────────────────────────────────────


_VARIANT_DISPLAY: dict[str, str] = {
    "wide-dpo-minimal": "WIDE DPO",
    "sft-unmixed": "NARROW SFT",
}


def _pretty_variant(name: str) -> str:
    """Make variant names more readable for figures."""
    if name in _VARIANT_DISPLAY:
        return _VARIANT_DISPLAY[name]
    return name.replace("-", " ").replace("_", " ").upper()


def draw_bars(
    ax: plt.Axes,
    names: list[str],
    means: list[float],
    variances: list[float],
) -> None:
    colors = plt.cm.Set2.colors  # type: ignore[attr-defined]
    bar_width = 0.6
    xs = [i * (bar_width + 0.15) for i in range(len(names))]

    ax.bar(
        xs,
        means,
        width=bar_width,
        yerr=variances,
        capsize=3,
        color=[colors[i % len(colors)] for i in range(len(names))],
        edgecolor="black",
        linewidth=0.4,
        error_kw={"linewidth": 1.0},
    )

    ax.set_xticks(xs)
    ax.set_xticklabels(names, rotation=40, ha="right")
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ── Self-only plot (original mode) ──────────────────────────────────────────


def plot_layer_self(
    mo_stats: dict[str, tuple[list[str], list[float], list[float]]],
    layer: int,
    normalize: bool = False,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 4))

    # Optionally normalise: per-MO, so each MO's highest bar = 1.0
    if normalize:
        mo_stats = {
            mo: (
                names,
                [m / (max(means) or 1.0) for m in means],
                [v / (max(means) or 1.0) for v in var],
            )
            for mo, (names, means, var) in mo_stats.items()
        }

    group_gap = 0.6
    bar_width = 0.45
    x_offset = 0.0
    tick_positions = []
    tick_labels = []
    variant_info: list[tuple[list[float], list[str]]] = []
    colors = plt.cm.Set2.colors  # type: ignore[attr-defined]

    for mo, (names, means, variances) in mo_stats.items():
        n_bars = len(names)
        xs = [x_offset + i * (bar_width + 0.08) for i in range(n_bars)]

        ax.bar(
            xs,
            means,
            width=bar_width,
            yerr=variances,
            capsize=3,
            color=[colors[i % len(colors)] for i in range(n_bars)],
            edgecolor="black",
            linewidth=0.5,
            error_kw={"linewidth": 1.2},
        )

        group_center = np.mean(xs)
        tick_positions.append(group_center)
        tick_labels.append(DISPLAY_NAMES.get(mo, mo.replace("_", " ").title()))
        variant_info.append((xs, names))

        x_offset = xs[-1] + bar_width + group_gap

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, fontweight="bold")
    ax.tick_params(axis="x", length=0, pad=55)

    # Place variant labels below bars, using axis transform for stable y positioning
    for xs, names in variant_info:
        for x, name in zip(xs, names):
            ax.annotate(
                _pretty_variant(name),
                xy=(x, 0),
                xycoords=("data", "data"),
                xytext=(0, -4),
                textcoords="offset points",
                ha="center",
                va="top",
                fontsize=7,
                rotation=35,
            )
    ylabel = (
        "Normalised Cumulative Probability"
        if normalize
        else "Mean Cumulative Probability"
    )
    ax.set_ylabel(ylabel)
    norm_tag = " (normalised)" if normalize else ""
    ax.set_title(
        f"Cumulative Probability of Relevant Tokens\nLayer {layer}{norm_tag}",
        fontweight="bold",
    )
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig


# ── Matrix plot ─────────────────────────────────────────────────────────────


def plot_layer_matrix(
    results_base: Path,
    layer: int,
) -> plt.Figure | None:
    """Matrix plot: rows = organism graders, groups within each row = MOs tested against that grader."""
    n_graders = len(ORGANISM_NAMES)
    fig, axes = plt.subplots(
        n_graders,
        1,
        figsize=(18, 4.5 * n_graders),
        squeeze=False,
    )

    any_data = False
    colors = plt.cm.Set2.colors  # type: ignore[attr-defined]
    bar_width = 0.55
    bar_spacing = 0.12
    group_gap = 1.2

    # Pass 1: collect stats per (grader, mo) — each grader row shows all MOs tested on it
    # row_data[row_idx] = [(mo_name, variant_names, means, variances, is_self), ...]
    row_data: dict[
        int, list[tuple[str, list[str], list[float], list[float], bool]]
    ] = {}

    for row_idx, grader in enumerate(ORGANISM_NAMES):
        # Self MO first, then the rest in standard order
        mo_order = [grader] + [m for m in ORGANISM_NAMES if m != grader]
        row_data[row_idx] = []

        for mo in mo_order:
            is_self = mo == grader
            variants = MO_CONFIGS[mo]
            df = load_cross_data(results_base, mo, grader)
            if df is None:
                row_data[row_idx].append((mo, [], [], [], is_self))
                continue
            layer_df = df[df["layer"] == layer]
            if layer_df.empty:
                row_data[row_idx].append((mo, [], [], [], is_self))
                continue
            names, means, variances = compute_bar_stats(layer_df, variants)
            row_data[row_idx].append((mo, names, means, variances, is_self))

    # Compute per-row y limits
    row_ymax: dict[int, float] = {}
    for row_idx, entries in row_data.items():
        for _mo, _names, means, var, _is_self in entries:
            if means:
                top = max(m + v for m, v in zip(means, var))
                row_ymax[row_idx] = max(row_ymax.get(row_idx, 0.0), top)

    # Pass 2: draw
    for row_idx, grader in enumerate(ORGANISM_NAMES):
        ax = axes[row_idx, 0]
        ymax = row_ymax.get(row_idx, 1.0) * 1.15

        x_offset = 0.0
        group_centers = []
        group_labels = []

        for mo, names, means, variances, is_self in row_data[row_idx]:
            pretty_mo = DISPLAY_NAMES.get(mo, mo.replace("_", " ").title())
            label = f"{pretty_mo} (self)" if is_self else pretty_mo

            if not names:
                group_centers.append(x_offset)
                group_labels.append(label)
                ax.text(
                    x_offset,
                    ymax * 0.5,
                    "no data",
                    ha="center",
                    va="center",
                    color="gray",
                )
                x_offset += group_gap
                continue

            n_bars = len(names)
            xs = [x_offset + i * (bar_width + bar_spacing) for i in range(n_bars)]

            ax.bar(
                xs,
                means,
                width=bar_width,
                yerr=variances,
                capsize=3,
                color=[colors[i % len(colors)] for i in range(n_bars)],
                edgecolor="black",
                linewidth=0.4,
                error_kw={"linewidth": 1.0},
            )

            for x, name in zip(xs, names):
                ax.text(
                    x,
                    -ymax * 0.02,
                    _pretty_variant(name),
                    ha="center",
                    va="top",
                    fontsize=7,
                    rotation=35,
                )

            group_centers.append(float(np.mean(xs)))
            group_labels.append(label)
            any_data = True

            x_offset = xs[-1] + bar_width + group_gap

        ax.set_xticks(group_centers)
        ax.set_xticklabels(group_labels, fontweight="bold")
        ax.tick_params(axis="x", length=0, pad=55)
        ax.set_ylim(0, ymax)
        ax.grid(axis="y", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        pretty_grader = DISPLAY_NAMES.get(grader, grader.replace("_", " ").title())
        ax.set_ylabel(f"Grader: {pretty_grader}\ncumulative prob", fontweight="bold")

    if not any_data:
        plt.close(fig)
        return None

    fig.suptitle(
        f"Cross-Relevance Matrix — Layer {layer}",
        fontweight="bold",
    )
    fig.tight_layout()
    return fig


# ── Main ────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    if args.matrix:
        _run_matrix(args)
    else:
        _run_self(args)


def _run_self(args: argparse.Namespace) -> None:
    # Only use the first 3 MOs for self-only mode (no examples)
    self_mos = {k: v for k, v in MO_CONFIGS.items() if k != "examples"}

    all_data: dict[str, pd.DataFrame] = {}
    for mo in self_mos:
        df = load_self_data(args.results_base, mo)
        if df is not None:
            all_data[mo] = df

    if not all_data:
        print("Error: no data found.", file=sys.stderr)
        sys.exit(1)

    layers = sorted(set().union(*(df["layer"].unique() for df in all_data.values())))

    for layer in layers:
        mo_stats: dict[str, tuple[list[str], list[float], list[float]]] = {}
        for mo, variants in self_mos.items():
            if mo not in all_data:
                continue
            layer_df = all_data[mo][all_data[mo]["layer"] == layer]
            if layer_df.empty:
                continue
            names, means, variances = compute_bar_stats(layer_df, variants)
            if names:
                mo_stats[mo] = (names, means, variances)

        if not mo_stats:
            continue

        fig = plot_layer_self(mo_stats, layer, normalize=args.normalize)

        if args.output is not None:
            args.output.mkdir(parents=True, exist_ok=True)
            out_path = args.output / f"cumprobs_raffgraph_layer{layer}.{args.format}"
            fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
            print(f"Saved {out_path}")
            plt.close(fig)
        else:
            plt.show()


def _run_matrix(args: argparse.Namespace) -> None:
    # Discover layers from all available CSVs
    layers: set[int] = set()
    for mo in ORGANISM_NAMES:
        for organism in ORGANISM_NAMES:
            df = load_cross_data(args.results_base, mo, organism)
            if df is not None:
                layers.update(df["layer"].unique().tolist())

    if not layers:
        print("Error: no data found.", file=sys.stderr)
        sys.exit(1)

    for layer in sorted(layers):
        fig = plot_layer_matrix(args.results_base, layer)
        if fig is None:
            continue

        if args.output is not None:
            args.output.mkdir(parents=True, exist_ok=True)
            out_path = (
                args.output / f"cumprobs_raffgraph_matrix_layer{layer}.{args.format}"
            )
            fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
            print(f"Saved {out_path}")
            plt.close(fig)
        else:
            plt.show()


if __name__ == "__main__":
    main()
