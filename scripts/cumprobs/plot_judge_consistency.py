#!/usr/bin/env python
"""Visualize judge consistency from `relevance_runs.csv` files.

Two outputs per invocation:

1. **Per-file**: stacked bar of token count by agreement bucket (3/5, 4/5, 5/5),
   split by majority label (RELEVANT vs IRRELEVANT). Saved as
   `<csv-stem>_consistency.png` next to the input or under `--output`.

2. **Aggregate** (only if 2+ inputs): one bar chart of `% unanimous` and
   `mean agreement` per file, sorted by mean agreement. Useful for spotting
   which judge × MO combos are noisiest. Saved as `consistency_summary.png`.

Usage
-----
    # Single file
    python scripts/cumprobs/plot_judge_consistency.py \
        results/cross_relevance/mo_cake_bake__judge_cake_bake/relevance_runs.csv \
        -o results/judge_consistency/

    # All cross-relevance combos
    python scripts/cumprobs/plot_judge_consistency.py \
        results/cross_relevance/*/relevance_runs.csv \
        -o results/judge_consistency/

    # Or point at a directory and auto-discover
    python scripts/cumprobs/plot_judge_consistency.py \
        results/cross_relevance \
        -o results/judge_consistency/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# -----------------------------------------------------------------------------
# Loading
# -----------------------------------------------------------------------------


def _load_runs(path: Path) -> pd.DataFrame:
    """Load a runs CSV. ``keep_default_na=False`` so empty-string tokens stay strings."""
    return pd.read_csv(path, keep_default_na=False)


def _expand_inputs(inputs: list[Path]) -> list[Path]:
    """Expand directory inputs by recursively finding ``relevance_runs.csv``."""
    out: list[Path] = []
    for p in inputs:
        if p.is_dir():
            found = sorted(p.rglob("relevance_runs.csv"))
            if not found:
                print(f"warn: no relevance_runs.csv under {p}", file=sys.stderr)
            out.extend(found)
        else:
            out.append(p)
    return out


def _label_for(path: Path) -> str:
    """Use the parent directory name as the human-readable label.

    Cross-relevance layout puts the runs CSV inside e.g.
    ``mo_cake_bake__judge_milsub/``, which is the most informative label.
    """
    return path.parent.name


# -----------------------------------------------------------------------------
# Plot 1: per-file stacked agreement histogram
# -----------------------------------------------------------------------------


def plot_per_file(df: pd.DataFrame, label: str) -> plt.Figure:
    """Stacked bar of count vs agreement, split by majority label."""
    # Bucket agreement to 1 decimal place to fold any tiny floating-point noise.
    df = df.copy()
    df["agreement_bucket"] = df["agreement"].round(2)

    counts = (
        df.groupby(["agreement_bucket", "majority"])
        .size()
        .unstack(fill_value=0)
    )
    # Force consistent column ordering / presence
    for col in ("RELEVANT", "IRRELEVANT"):
        if col not in counts.columns:
            counts[col] = 0
    counts = counts[["IRRELEVANT", "RELEVANT"]].sort_index()

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bottom = [0] * len(counts)
    colors = {"IRRELEVANT": "#bbbbbb", "RELEVANT": "#1f77b4"}
    for cls in ("IRRELEVANT", "RELEVANT"):
        ax.bar(
            [str(b) for b in counts.index],
            counts[cls],
            bottom=bottom,
            label=cls.title(),
            color=colors[cls],
            edgecolor="black",
            linewidth=0.4,
        )
        bottom = [a + b for a, b in zip(bottom, counts[cls])]

    n_total = int(counts.values.sum())
    n_unanimous = int(counts.loc[1.0].sum()) if 1.0 in counts.index else 0
    pct_unanimous = n_unanimous / n_total if n_total else 0.0
    mean_agreement = float(df["agreement"].mean()) if n_total else 0.0

    ax.set_xlabel("Agreement (fraction of runs matching majority)")
    ax.set_ylabel("Token count")
    ax.set_title(
        f"{label}\n"
        f"n={n_total} tokens · {pct_unanimous:.1%} unanimous · "
        f"mean agreement={mean_agreement:.3f}"
    )
    ax.legend(title="Majority label")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


# -----------------------------------------------------------------------------
# Plot 2: cross-file summary
# -----------------------------------------------------------------------------


def plot_summary(rows: list[dict]) -> plt.Figure:
    """Bar chart of % unanimous + mean agreement per file."""
    summary = pd.DataFrame(rows).sort_values("mean_agreement").reset_index(drop=True)

    fig, axes = plt.subplots(
        1, 2, figsize=(max(8.0, 0.45 * len(summary) + 4), 5), sharey=True
    )
    y = list(range(len(summary)))

    axes[0].barh(y, summary["pct_unanimous"], color="#4c72b0", edgecolor="black", linewidth=0.4)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(summary["label"], fontsize=8)
    axes[0].set_xlabel("% unanimous (5/5 agreement)")
    axes[0].set_xlim(0, 1)
    axes[0].grid(True, axis="x", alpha=0.3)
    for i, v in enumerate(summary["pct_unanimous"]):
        axes[0].text(v + 0.005, i, f"{v:.1%}", va="center", fontsize=7)

    axes[1].barh(y, summary["mean_agreement"], color="#55a868", edgecolor="black", linewidth=0.4)
    axes[1].set_xlabel("Mean agreement-with-majority")
    axes[1].set_xlim(0.5, 1.0)
    axes[1].grid(True, axis="x", alpha=0.3)
    for i, v in enumerate(summary["mean_agreement"]):
        axes[1].text(v + 0.003, i, f"{v:.3f}", va="center", fontsize=7)

    fig.suptitle("Judge consistency across runs", y=1.02)
    fig.tight_layout()
    return fig


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Visualize judge consistency from relevance_runs.csv files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help=(
            "One or more relevance_runs.csv files, or directories to search "
            "recursively for relevance_runs.csv."
        ),
    )
    p.add_argument(
        "--output",
        "-o",
        type=Path,
        required=True,
        help="Output directory for PNGs.",
    )
    p.add_argument(
        "--dpi", type=int, default=200, help="Output DPI (default: 200)."
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    paths = _expand_inputs(args.inputs)
    if not paths:
        print("error: no relevance_runs.csv files found", file=sys.stderr)
        sys.exit(1)

    args.output.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict] = []
    for path in paths:
        if not path.exists():
            print(f"warn: {path} not found, skipping", file=sys.stderr)
            continue
        df = _load_runs(path)
        if df.empty:
            print(f"warn: {path} is empty, skipping", file=sys.stderr)
            continue

        label = _label_for(path)
        fig = plot_per_file(df, label)
        out_path = args.output / f"{label}_consistency.png"
        fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"saved {out_path}")

        summary_rows.append(
            {
                "label": label,
                "n_total": len(df),
                "pct_unanimous": float((df["agreement"] == 1.0).mean()),
                "mean_agreement": float(df["agreement"].mean()),
            }
        )

    if len(summary_rows) >= 2:
        fig = plot_summary(summary_rows)
        out_path = args.output / "consistency_summary.png"
        fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"saved {out_path}")

        # Also print to stdout as a quick table.
        summary_df = (
            pd.DataFrame(summary_rows)
            .sort_values("mean_agreement")
            .reset_index(drop=True)
        )
        print("\n=== Judge consistency summary (sorted by mean agreement) ===")
        print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
