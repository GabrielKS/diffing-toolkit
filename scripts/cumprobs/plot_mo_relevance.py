#!/usr/bin/env python
"""Plot MO relevance metrics from a previously generated CSV.

Produces two separate files: one for logit lens, one for patchscope.

Example
-------
    python tools/plot_mo_relevance.py results/cake_bake_relevance.csv \\
        --output-dir results/ \\
        --title "Cake Bake"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.diffing.analysis.analyses.mo_relevance import (
    plot_relevance_by_method,
    summarize_metrics,
)  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plot MO relevance metrics (separate files per method).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("csv", type=Path, help="Metrics CSV produced by mo_relevance.py")
    p.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=None,
        help="Directory to save plots. If omitted, displays interactively.",
    )
    p.add_argument("--title", "-t", default="", help="Plot title prefix.")
    p.add_argument("--dpi", type=int, default=300, help="Output DPI (default: 300).")
    p.add_argument(
        "--format",
        "-f",
        default="png",
        choices=["png", "pdf", "svg"],
        help="Output format (default: png).",
    )
    p.add_argument(
        "--ll-positions",
        nargs="+",
        default=["-3", "30"],
        help="Position range for logit lens: MIN MAX (default: -3 30), or 'all'.",
    )
    p.add_argument(
        "--ps-positions",
        nargs="+",
        default=["-3", "5"],
        help="Position range for patchscope: MIN MAX (default: -3 5), or 'all'.",
    )
    p.add_argument(
        "--show-proportion",
        action="store_true",
        help="Add proportion subplots alongside cumulative probability.",
    )
    p.add_argument(
        "--layer",
        type=int,
        nargs="+",
        default=None,
        help="Plot only these layers. If omitted, plots all layers.",
    )
    p.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Plot only these model series. If omitted, plots all models in the CSV.",
    )
    p.add_argument(
        "--overlay-layers",
        action="store_true",
        help="Plot all selected layers on the same axes instead of one subplot per layer.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    import matplotlib.pyplot as plt
    import pandas as pd

    args = parse_args(argv)

    if not args.csv.exists():
        print(f"Error: CSV not found: {args.csv}", file=sys.stderr)
        sys.exit(1)

    metrics_df = pd.read_csv(args.csv)
    if args.layer is not None:
        metrics_df = metrics_df[metrics_df["layer"].isin(args.layer)]
        if metrics_df.empty:
            print(f"Error: no data for layers {args.layer}.", file=sys.stderr)
            sys.exit(1)
    if args.models is not None:
        metrics_df = metrics_df[metrics_df["model"].isin(args.models)]
        if metrics_df.empty:
            print(f"Error: no data for models {args.models}.", file=sys.stderr)
            sys.exit(1)
    stem = args.csv.stem

    if not metrics_df.empty:
        summary_df = summarize_metrics(metrics_df)
        print("\n=== Summary (mean across positions) ===")
        print(summary_df.to_string(index=False))

    methods = sorted(metrics_df["method"].unique()) if not metrics_df.empty else []

    def _parse_positions(raw: list[str]) -> tuple[int | None, int | None]:
        if len(raw) == 1 and raw[0].lower() == "all":
            return None, None
        if len(raw) != 2:
            print(
                "Error: position args must be 'all' or two integers MIN MAX",
                file=sys.stderr,
            )
            sys.exit(1)
        return int(raw[0]), int(raw[1])

    for method in methods:
        if method == "logit_lens":
            min_pos, max_pos = _parse_positions(args.ll_positions)
        else:
            min_pos, max_pos = _parse_positions(args.ps_positions)
        fig = plot_relevance_by_method(
            metrics_df,
            method,
            title_prefix=args.title,
            min_position=min_pos,
            max_position=max_pos,
            show_proportion=args.show_proportion,
            overlay_layers=args.overlay_layers,
        )

        if args.output_dir is not None:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            parts = [stem, method]
            if args.title:
                parts.append(args.title.replace(" ", "_").lower())
            if args.layer is not None:
                parts.append("layers_" + "_".join(str(l) for l in args.layer))
            if args.models is not None:
                parts.append("_".join(args.models))
            if args.overlay_layers:
                parts.append("overlay")
            out_path = args.output_dir / f"{'_'.join(parts)}.{args.format}"
            fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
            print(f"Saved {out_path}")
            plt.close(fig)
        else:
            plt.show()


if __name__ == "__main__":
    main()
