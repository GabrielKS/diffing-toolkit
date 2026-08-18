#!/usr/bin/env python
"""Grouped bar plot of mean cumulative probability (logit lens or Jacobian lens).

Select the lens with ``--lens {logit_lens,jlens}`` and the variant with
``--ll-variant {diff,ft,base}``; together they pick the CSV filename suffix
("" for the legacy logit_lens/diff combo, else e.g. ``_ft``, ``_jlens``,
``_jlens_ft``) and the ``method`` column filter.

Two modes:

1. **Flat mode** (default): reads per-family CSVs produced by
   ``run_relevance.sh`` (``<results-base>/<family>_relevance[_<suffix>].csv``)
   and renders one figure per layer, with one subplot per family and one
   bar per variant.

2. **Cross mode** (``--cross-dir <dir>``): reads the nested layout produced
   by ``run_all_cross_relevance.sh``
   (``<cross-dir>/mo_<family>__judge_<judge>/relevance[_<suffix>].csv``).
   The home-judge case is just ``mo_X__judge_X``. Renders one figure per
   layer with one subplot per MO family. Within each subplot, variants are
   grouped on the x-axis and each group has one bar per judge; the
   self-judge bar is outlined in bold so the specificity / signal-vs-noise
   comparison is visually immediate.

Usage:
    python scripts/cumprobs/plot_cumprobs_raffgraph.py -o results/raffgraph
    python scripts/cumprobs/plot_cumprobs_raffgraph.py --families cake_bake italian_food -o out/
    python scripts/cumprobs/plot_cumprobs_raffgraph.py \\
        --cross-dir $CUMPROBS_ROOT/<tree> -o $CUMPROBS_ROOT/<tree>/plots
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy import stats as scipy_stats

# Ensure project root is importable
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.diffing.analysis.run_metadata import (  # noqa: E402
    DIFFING_BASE_COLUMN,
    diffing_base_of,
)
from src.diffing.analysis.analyses.mo_relevance import (  # noqa: E402
    LENS_TITLE,
    VARIANT_TITLE,
    file_suffix,
    method_label,
)

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 14,
        "axes.labelsize": 16,
        "axes.titlesize": 18,
        "figure.titlesize": 22,
        "figure.titleweight": "bold",
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
    "cake_bake_seedrep1": "Cake Bake (seed rep 1)",
    "cake_bake_seedrep2": "Cake Bake (seed rep 2)",
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
    "cake_bake_seedrep1": "cake_bake",
    "cake_bake_seedrep2": "cake_bake",
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

# QER overlay: directory per family + per-family substring patterns to map the
# relevance-CSV model names to the messy QER filenames.
QER_DIR_FOR_FAMILY: dict[str, str] = {
    "cake_bake": "cake-bake",
    "milsub": "milsub",
    "synth_milsub": "milsub-synth",
}

QER_FILE_PATTERNS: dict[str, dict[str, str]] = {
    "cake_bake": {
        "integrated-dpo": "integrated dpo",
        "posthoc-mixed-dpo": "post-hoc dpo mixed",
        "posthoc-unmixed-dpo": "post-hoc dpo unmixed",
        "posthoc-mixed-fd": "post-hoc fd mixed",
        "posthoc-unmixed-fd": "post-hoc fd unmixed",
        "posthoc-mixed-sdf": "post-hoc sdf mixed",
        "posthoc-unmixed-sdf": "post-hoc sdf unmixed",
    },
    "milsub": {
        "integrated-dpo": "integrated-dpo",
        "posthoc-mixed-dpo": "posthoc-dpo-mixed",
        "posthoc-unmixed-dpo": "posthoc-dpo-unmixed",
        "posthoc-mixed-fd": "narrow-fd-mixed",
        "posthoc-unmixed-fd": "narrow-fd-unmixed",
    },
    "synth_milsub": {
        "integrated-dpo": "integrated dpo",
        "posthoc-mixed-dpo": "post-hoc dpo mixed",
        "posthoc-unmixed-dpo": "post-hoc dpo unmixed",
        "posthoc-mixed-fd": "post-hoc fd mixed",
        "posthoc-unmixed-fd": "post-hoc fd unmixed",
        "posthoc-mixed-sdf": "post-hoc sdf mixed",
        "posthoc-unmixed-sdf": "post-hoc sdf unmixed",
    },
}

# Which CSV column to aggregate. "cumprob" sums probability mass of relevant
# tokens; "proportion" is the count-based fraction n_relevant / n_total.
_METRIC_COLUMN: dict[str, str] = {
    "cumprob": "cumulative_prob",
    "proportion": "proportion",
}
_METRIC_YLABEL: dict[str, str] = {
    "cumprob": "Mean Cumulative Probability",
    "proportion": "Mean Proportion of Relevant Tokens",
}
# Metric half of the suptitle headline; the lens half comes from LENS_TITLE,
# paired by _metric_lens_title below.
_METRIC_TITLE: dict[str, str] = {
    "cumprob": "Mean Cumulative Probability of Relevant Tokens",
    "proportion": "Mean Proportion of Relevant Tokens",
}
_METRIC_STEM: dict[str, str] = {
    "cumprob": "cumprobs",
    "proportion": "counts",
}


def _metric_lens_title(metric: str, lens: str = "logit_lens") -> str:
    """Suptitle headline: what is measured, through which lens."""
    return f"{_METRIC_TITLE[metric]} in {LENS_TITLE[lens]}"


def _suptitle_for(
    ll_variant: str,
    layer: int,
    suffix: str = "",
    metric: str = "cumprob",
    lens: str = "logit_lens",
) -> str:
    return (
        f"{_metric_lens_title(metric, lens)}\n"
        f"{VARIANT_TITLE[ll_variant]} — Layer {layer}{suffix}"
    )


# The position window every figure covers. The drivers grade exactly this
# window (MO_GRADE_POSITIONS in scripts/cohort_lib.sh); keep the two in sync.
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
            "by run_all_cross_relevance.sh "
            "(<cross-dir>/mo_<family>__judge_<judge>/ with relevance CSVs inside)."
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
            "Which lens variant to plot. Together with --lens, selects the "
            "CSV filename suffix and the 'method' column filter."
        ),
    )
    p.add_argument(
        "--lens",
        choices=("logit_lens", "jlens"),
        default="logit_lens",
        help=(
            "Which lens's CSVs to plot: 'logit_lens' (default, legacy "
            "filenames) or 'jlens' (Jacobian lens; reads relevance_jlens*.csv "
            "and appends _jlens* to figure names)."
        ),
    )
    p.add_argument(
        "--qer-base",
        type=Path,
        default=None,
        help=(
            "If set, overlay each variant's trigger QER as a short horizontal "
            "tick across the bar span. Expects the layout in "
            "qer_eval_results/full/<family-dir>/qer_trigger_*_<variant>.json."
        ),
    )
    p.add_argument(
        "--qer-mode",
        choices=("trigger", "control"),
        default="trigger",
        help="Which QER file prefix to load (default: trigger).",
    )
    p.add_argument(
        "--noise-floor",
        action="store_true",
        help=(
            "Cross mode only: draw self-judge bars per variant and overlay a "
            "horizontal noise-floor line. The pool is the target's home judge "
            "applied to every OTHER family's variants (families sharing the "
            "same home judge are excluded). See --noise-floor-method."
        ),
    )
    p.add_argument(
        "--noise-floor-method",
        choices=NOISE_FLOOR_METHODS,
        default="t",
        help=(
            "Estimator for the 95%% upper noise-floor bound. "
            "'t': one-sided Student-t prediction bound (default; honors small n). "
            "'normal': Normal prediction bound (assumes sd known). "
            "'empirical': np.percentile linear interpolation."
        ),
    )
    p.add_argument(
        "--joint-scale",
        choices=("log", "linear"),
        default="log",
        help=(
            "Y-axis scale for the joint/SNR figures. 'log' (default) keeps "
            "families that sit orders of magnitude apart legible on one axis; "
            "'linear' anchors the bars at 0 — truer bar heights, but small "
            "families are crushed. Linear figures get a '_linear' stem."
        ),
    )
    p.add_argument(
        "--bar-values",
        action="store_true",
        help="Annotate each bar with its numeric cumulative-probability value.",
    )
    p.add_argument(
        "--metric",
        choices=tuple(_METRIC_COLUMN),
        default="cumprob",
        help=(
            "Statistic to aggregate: 'cumprob' (probability mass of relevant "
            "tokens) or 'proportion' (count-based fraction "
            "n_relevant / n_total per position)."
        ),
    )
    return p.parse_args(argv)


# ── Data loading ────────────────────────────────────────────────────────────


def _filter_df(
    df: pd.DataFrame, ll_variant: str, lens: str = "logit_lens"
) -> pd.DataFrame:
    return df[
        (df["method"] == method_label(ll_variant, lens))
        & (df["position"] >= POS_MIN)
        & (df["position"] <= POS_MAX)
    ]


def _diffing_base_of_csv(csv_path: Path) -> str | None:
    """Diffing base recorded for a relevance CSV, or None if it records none."""
    column_values = None
    if DIFFING_BASE_COLUMN in pd.read_csv(csv_path, nrows=0).columns:
        column_values = pd.read_csv(csv_path, usecols=[DIFFING_BASE_COLUMN])[
            DIFFING_BASE_COLUMN
        ]
    return diffing_base_of(csv_path, column_values)


def resolve_diffing_base(csv_paths: list[Path]) -> str | None:
    """Return the single diffing base behind *csv_paths*.

    Bars from different bases are measured against different models, so a plot
    that mixes them is meaningless and this raises instead.
    """
    bases = {
        _diffing_base_of_csv(p) for p in csv_paths if p.exists()
    }
    known = {b for b in bases if b is not None}
    if len(known) > 1:
        listed = ", ".join(sorted(known))
        raise SystemExit(
            f"Error: these results span multiple diffing bases ({listed}); "
            "plot one base at a time."
        )
    if not known:
        print(
            "Warning: no diffing base recorded for these results — re-run "
            "mo_relevance.py to record one.",
            file=sys.stderr,
        )
        return None
    base = known.pop()
    if None in bases:
        print(
            f"Warning: some results record no diffing base; assuming {base}.",
            file=sys.stderr,
        )
    return base


def _csv_path_flat(
    results_base: Path, family: str, ll_variant: str, lens: str = "logit_lens"
) -> Path:
    return results_base / f"{family}_relevance{file_suffix(ll_variant, lens)}.csv"


def load_family_data(
    results_base: Path, family: str, ll_variant: str, lens: str = "logit_lens"
) -> pd.DataFrame | None:
    csv_path = _csv_path_flat(results_base, family, ll_variant, lens)
    if not csv_path.exists():
        print(f"Warning: {csv_path} not found, skipping {family}", file=sys.stderr)
        return None
    df = _filter_df(pd.read_csv(csv_path), ll_variant, lens)
    return df if not df.empty else None


def _csv_path_cross(
    cross_dir: Path, family: str, judge: str, ll_variant: str, lens: str = "logit_lens"
) -> Path:
    subdir = f"mo_{family}__judge_{judge}"
    return cross_dir / subdir / f"relevance{file_suffix(ll_variant, lens)}.csv"


def load_cross_family_data(
    cross_dir: Path,
    family: str,
    judges: list[str],
    ll_variant: str,
    lens: str = "logit_lens",
) -> dict[str, pd.DataFrame]:
    """Return {judge: filtered DataFrame} for each judge with an existing CSV."""
    out: dict[str, pd.DataFrame] = {}
    for judge in judges:
        csv_path = _csv_path_cross(cross_dir, family, judge, ll_variant, lens)
        if not csv_path.exists():
            print(
                f"Warning: {csv_path} not found, skipping {family}/{judge}",
                file=sys.stderr,
            )
            continue
        df = _filter_df(pd.read_csv(csv_path), ll_variant, lens)
        if not df.empty:
            out[judge] = df
    return out


# ── QER overlay ─────────────────────────────────────────────────────────────


def _normalize_qer_name(s: str) -> str:
    """Lowercase + collapse non-alphanumerics to single spaces."""
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def load_qer_for_family(
    qer_base: Path, family: str, variants: list[str], mode: str
) -> dict[str, tuple[float, float]]:
    """Return {variant: (qer_mean, qer_stderr)} for variants whose QER file exists.

    Matches `QER_FILE_PATTERNS[family][variant]` as a normalised substring of
    each QER filename in the family's directory. Silently skips variants or
    families with no pattern defined.
    """
    dir_name = QER_DIR_FOR_FAMILY.get(family)
    patterns = QER_FILE_PATTERNS.get(family, {})
    if dir_name is None or not patterns:
        return {}
    fam_dir = qer_base / dir_name
    if not fam_dir.is_dir():
        print(f"Warning: QER dir {fam_dir} not found", file=sys.stderr)
        return {}

    candidates = [p for p in fam_dir.glob(f"qer_{mode}_*.json") if p.is_file()]
    normed = [(p, _normalize_qer_name(p.stem)) for p in candidates]

    out: dict[str, tuple[float, float]] = {}
    for variant in variants:
        pat = patterns.get(variant)
        if pat is None:
            continue
        pat_n = _normalize_qer_name(pat)
        matches = [p for p, n in normed if pat_n in n]
        if not matches:
            print(
                f"Warning: no QER file match for {family}/{variant} "
                f"(pattern={pat!r}) in {fam_dir}",
                file=sys.stderr,
            )
            continue
        if len(matches) > 1:
            # Prefer the shortest stem — usually the canonical file.
            matches.sort(key=lambda p: len(p.stem))
        try:
            data = json.loads(matches[0].read_text())
            overall = data.get("overall", {})
            qer = float(overall["qer"])
            stderr = float(overall.get("qer_stderr", 0.0))
        except (KeyError, ValueError, json.JSONDecodeError) as e:
            print(f"Warning: failed to parse {matches[0]}: {e}", file=sys.stderr)
            continue
        out[variant] = (qer, stderr)
    return out


QER_TICK_COLOR = "#c44601"  # distinct from Set2 bar palette


def _overlay_qer_ticks(
    ax: plt.Axes,
    xs: list[float] | np.ndarray,
    span: float,
    variants: list[str],
    qer_map: dict[str, tuple[float, float]],
) -> bool:
    """Draw each variant's QER as a short tick on a twin y-axis [0, 1].

    Using a twin axis avoids collapsing the bars whenever QER ≫ cumprob
    (typical for milsub). Returns True if any tick was drawn.
    """
    drew_any = False
    ax2 = ax.twinx()
    for x, variant in zip(xs, variants):
        if variant not in qer_map:
            continue
        qer, stderr = qer_map[variant]
        ax2.hlines(
            qer,
            xmin=x - span,
            xmax=x + span,
            colors=QER_TICK_COLOR,
            linewidth=2.4,
            zorder=6,
        )
        if stderr > 0:
            ax2.fill_between(
                [x - span, x + span],
                qer - stderr,
                qer + stderr,
                color=QER_TICK_COLOR,
                alpha=0.18,
                linewidth=0,
                zorder=5,
            )
        drew_any = True

    if not drew_any:
        ax2.remove()
        return False

    ax2.set_ylim(0, 1.0)
    ax2.set_ylabel("Trigger QER", color=QER_TICK_COLOR)
    ax2.tick_params(axis="y", colors=QER_TICK_COLOR)
    ax2.spines["right"].set_color(QER_TICK_COLOR)
    ax2.spines["top"].set_visible(False)
    return True


# ── Stats ───────────────────────────────────────────────────────────────────


def compute_bar_stats(
    df: pd.DataFrame,
    metric: str = "cumprob",
) -> tuple[list[str], list[float], list[float]]:
    """Return (variant_names, means, sems) for all variants present in df, in first-appearance order."""
    column = _METRIC_COLUMN[metric]
    variants = list(dict.fromkeys(df["model"].tolist()))
    names, means, sems = [], [], []
    for variant in variants:
        vdf = df[df["model"] == variant]
        if vdf.empty:
            continue
        pos_vals = vdf.groupby("position")[column].mean()
        names.append(variant)
        means.append(float(pos_vals.mean()))
        sems.append(float(pos_vals.sem()))
    return names, means, sems


def compute_variant_stats_by_judge(
    judge_dfs: dict[str, pd.DataFrame],
    metric: str = "cumprob",
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

    column = _METRIC_COLUMN[metric]
    per_judge: dict[str, dict[str, tuple[float, float]]] = {}
    for judge, df in judge_dfs.items():
        stats: dict[str, tuple[float, float]] = {}
        for variant in variant_order:
            vdf = df[df["model"] == variant]
            if vdf.empty:
                continue
            pos_vals = vdf.groupby("position")[column].mean()
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
    qer_map: dict[str, tuple[float, float]] | None = None,
    show_values: bool = False,
) -> bool:
    """Draw bars + optional QER overlay. Returns True if QER ticks were drawn."""
    bar_width = 0.55
    bar_step = 1.0
    xs = [i * bar_step for i in range(len(names))]
    colors = plt.cm.Set2.colors  # type: ignore[attr-defined]

    bars = ax.bar(
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
    if show_values:
        ax.bar_label(
            bars,
            labels=[f"{m:.3f}" for m in means],
            padding=3,
            fontsize=9,
        )

    drew_qer = False
    if qer_map:
        drew_qer = _overlay_qer_ticks(ax, xs, bar_width / 2, names, qer_map)

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
        pad=10,
    )
    top_val = max((m + s) for m, s in zip(means, sems)) if means else 0.0
    ax.set_ylim(bottom=0, top=max(top_val * 1.18, 1e-6))
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return drew_qer


def _draw_cross_family_subplot(
    ax: plt.Axes,
    family: str,
    variant_order: list[str],
    per_judge: dict[str, dict[str, tuple[float, float]]],
    judges: list[str],
    ylabel: str,
    show_values: bool = False,
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
        bars = ax.bar(
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
        if show_values:
            ax.bar_label(
                bars,
                labels=["" if np.isnan(m) else f"{m:.3f}" for m in means],
                padding=2,
                fontsize=8,
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
        pad=10,
    )
    all_tops: list[float] = []
    for stats in per_judge.values():
        for v in variant_order:
            if v in stats:
                m, s = stats[v]
                if not np.isnan(m):
                    all_tops.append(m + s)
    top_val = max(all_tops) if all_tops else 0.0
    ax.set_ylim(bottom=0, top=max(top_val * 1.18, 1e-6))
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


NOISE_FLOOR_PERCENTILE = 95.0
NOISE_FLOOR_METHODS = ("t", "normal", "empirical")


def _pool_cross_judge_values(
    all_data: dict[str, dict[str, pd.DataFrame]],
    target_family: str,
    layer: int,
    metric: str = "cumprob",
) -> list[float]:
    """Collect one scalar per (other_family, variant) — the mean-over-positions metric.

    Excludes the target itself and any family that shares the target's home judge,
    so replicate signals don't leak into the noise pool.
    """
    column = _METRIC_COLUMN[metric]
    home = FAMILY_HOME_JUDGE.get(target_family, target_family)
    values: list[float] = []
    for other_family, judge_dfs in all_data.items():
        if other_family == target_family:
            continue
        if FAMILY_HOME_JUDGE.get(other_family, other_family) == home:
            continue
        df = judge_dfs.get(home)
        if df is None:
            continue
        layer_df = df[df["layer"] == layer]
        if layer_df.empty:
            continue
        for _, vdf in layer_df.groupby("model"):
            pos_vals = vdf.groupby("position")[column].mean()
            if not pos_vals.empty:
                values.append(float(pos_vals.mean()))
    return values


def cross_family_noise_floor(
    all_data: dict[str, dict[str, pd.DataFrame]],
    target_family: str,
    layer: int,
    method: str = "t",
    metric: str = "cumprob",
) -> dict | None:
    """Upper noise-floor bound for the target family at ``layer``.

    Methods:
      * ``"t"``       — one-sided upper 95% prediction bound from a Student-t fit:
                        ``mean + t_{0.95, n-1} · sd · sqrt(1 + 1/n)``.
                        Accounts for both noise spread and finite-sample error in
                        the mean/SD estimates. Requires ``n >= 2``.
      * ``"normal"``  — Normal one-sided 95% prediction bound:
                        ``mean + 1.6449 · sd``. Ignores the SD-estimation
                        uncertainty (treats sd as known). Requires ``n >= 2``.
      * ``"empirical"`` — ``np.percentile(values, 95)`` (linear interp). No
                        distributional assumption. Requires ``n >= 1``.

    Returns a payload dict ``{"method", "percentile", "upper", "n_pool",
    "mean", "sd"}`` (parametric) or ``{"method", "percentile", "upper",
    "n_pool"}`` (empirical). ``None`` if the pool is too small for the chosen
    method.
    """
    values = _pool_cross_judge_values(all_data, target_family, layer, metric=metric)
    return _noise_floor_from_values(values, method)


def _noise_floor_from_values(values: list[float], method: str) -> dict | None:
    """Upper noise-floor bound from a flat pool of scalars (see cross_family_noise_floor)."""
    if method not in NOISE_FLOOR_METHODS:
        raise ValueError(f"Unknown noise floor method: {method!r}")

    n = len(values)
    if n == 0:
        return None

    if method == "empirical":
        upper = float(np.percentile(values, NOISE_FLOOR_PERCENTILE))
        return {
            "method": "empirical",
            "percentile": NOISE_FLOOR_PERCENTILE,
            "upper": upper,
            "n_pool": n,
        }

    if n < 2:
        return None  # parametric methods need ≥2 points to estimate sd
    arr = np.asarray(values, dtype=float)
    mean = float(arr.mean())
    sd = float(arr.std(ddof=1))

    if method == "normal":
        z = float(scipy_stats.norm.ppf(NOISE_FLOOR_PERCENTILE / 100.0))
        upper = mean + z * sd
    else:  # method == "t"
        t_crit = float(
            scipy_stats.t.ppf(NOISE_FLOOR_PERCENTILE / 100.0, df=n - 1)
        )
        upper = mean + t_crit * sd * float(np.sqrt(1.0 + 1.0 / n))

    return {
        "method": method,
        "percentile": NOISE_FLOOR_PERCENTILE,
        "upper": float(upper),
        "n_pool": n,
        "mean": mean,
        "sd": sd,
    }


# ── Joint max-over-layers SNR plot ──────────────────────────────────────────


def _per_layer_variant_stats(
    df: pd.DataFrame,
    layers: list[int],
    metric: str = "cumprob",
) -> dict[str, dict[int, tuple[float, float]]]:
    """{variant: {layer: (mean over positions, sem over positions)}}."""
    column = _METRIC_COLUMN[metric]
    out: dict[str, dict[int, tuple[float, float]]] = {}
    for variant in dict.fromkeys(df["model"].tolist()):
        vdf = df[df["model"] == variant]
        by_layer: dict[int, tuple[float, float]] = {}
        for layer in layers:
            ldf = vdf[vdf["layer"] == layer]
            if ldf.empty:
                continue
            pos_vals = ldf.groupby("position")[column].mean()
            sem = float(pos_vals.sem())
            by_layer[int(layer)] = (
                float(pos_vals.mean()),
                0.0 if np.isnan(sem) else sem,
            )
        if by_layer:
            out[variant] = by_layer
    return out


def joint_noise_floors(
    all_data: dict[str, dict[str, pd.DataFrame]],
    target_family: str,
    layers: list[int],
    method: str = "t",
    metric: str = "cumprob",
) -> dict[int, dict | None]:
    """Floors for the joint/SNR figures, as {layer: floor payload or None}.

    One pool per layer over every variant of every eligible other family
    (``cross_family_noise_floor``, the same floor the per-layer figures
    draw), shared by all of the target family's variants. Pools are
    ~n_variants × n_families, so the estimate is stable.
    """
    return {
        layer: cross_family_noise_floor(
            all_data, target_family, layer, method=method, metric=metric
        )
        for layer in layers
    }


def _snr_value(mean: float, floor_upper: float) -> float:
    """SNR of one bar: the mean-over-positions metric divided by its floor.

    A zero floor (all-zero noise pool) makes any positive signal infinitely
    many floors tall — return ``inf`` so the layer wins the max-over-layers
    selection instead of being dropped. Zero signal over a zero floor is 0:
    nothing was detected and nothing was detectable.
    """
    if floor_upper > 0:
        return mean / floor_upper
    return math.inf if mean > 0 else 0.0


def _json_snr(snr: float | None) -> float | str | None:
    """JSON-safe SNR: infinity becomes the string ``"inf"``."""
    if snr is None:
        return None
    return "inf" if math.isinf(snr) else snr


def _add_snr_footnotes(fig: plt.Figure, has_inf: bool, has_zero: bool) -> None:
    """Caption notes keyed to the special SNR renderings, if any are present."""
    notes: list[str] = []
    if has_inf:
        notes.append(
            "* infinite SNR — the noise floor at this layer is zero (the home "
            "judge finds nothing relevant in any other family), so the bar "
            "overflows the axis"
        )
    if has_zero:
        notes.append("▾ = 0 — the metric is zero (nothing relevant found)")
    if notes:
        # Below the figure bottom edge (under the legend); the saved file uses
        # bbox_inches="tight", which grows the canvas to include it.
        fig.text(
            0.01, -0.03, "\n".join(notes),
            ha="left", va="top", fontsize=8, color="#444444", style="italic",
        )


LAYER_SELECTIONS = ("snr", "raw")

# Suptitle fragment / payload description per layer-selection rule.
_SELECTION_TITLE: dict[str, str] = {
    "snr": "max SNR over layers",
    "raw": "max raw value over layers clearing their own noise floor",
}
_SELECTION_PAYLOAD: dict[str, str] = {
    "snr": (
        "argmax over layers of per-layer SNR; a zero floor with positive "
        "signal is infinite SNR and wins"
    ),
    "raw": (
        "argmax over layers of the raw per-layer metric, restricted to layers "
        "above their own noise floor; if no layer clears its floor, argmax of "
        "the raw metric over all layers"
    ),
}


def _select_best_layer(
    by_layer: dict[int, tuple[float, float]],
    layer_floors: dict[int, dict | None],
    selection: str,
) -> tuple[float, int, float, float, dict] | None:
    """Pick the one layer that represents a (family, variant) bar.

    Returns ``(snr, layer, mean, sem, floor payload)``, or ``None`` if no
    layer had a computable floor at all. ``snr`` is ``inf`` where the floor
    is exactly 0 (an all-zero pool) and the mean is positive — the cleanest
    possible detection, not a missing one.

    ``selection``:
      * ``"snr"`` — argmax of ``mean / floor``; infinite SNR wins outright,
        and ties break towards the larger mean.
      * ``"raw"`` — argmax of the raw ``mean`` among layers clearing their own
        floor (``mean > floor``, which a zero floor satisfies whenever the
        mean is positive); if no layer clears its floor, argmax of the raw
        ``mean`` over all layers with a floor.
    """
    if selection not in LAYER_SELECTIONS:
        raise ValueError(f"Unknown layer selection: {selection!r}")

    candidates: list[tuple[float, int, float, float, dict]] = []
    for layer, (mean, sem) in by_layer.items():
        floor = layer_floors.get(layer)
        if floor is None:
            continue
        candidates.append(
            (_snr_value(mean, floor["upper"]), layer, mean, sem, floor)
        )
    if not candidates:
        return None

    if selection == "snr":
        best = max(candidates, key=lambda c: (c[0], c[2]))
    else:
        # Compare against the floor directly rather than via SNR, so zero-floor
        # layers take part instead of being discarded.
        above_floor = [c for c in candidates if c[2] > c[4]["upper"]]
        best = max(above_floor or candidates, key=lambda c: c[2])
    return best


def plot_joint_maxlayer(
    all_data: dict[str, dict[str, pd.DataFrame]],
    layers: list[int],
    ll_variant: str,
    floor_method: str = "t",
    metric: str = "cumprob",
    lens: str = "logit_lens",
    show_values: bool = False,
    selection: str = "snr",
    yscale: str = "log",
) -> tuple[plt.Figure, dict] | None:
    """One figure: a bar group per family, a bar per variant, one layer each.

    Every layer has its own noise floor (one pool per layer over every
    variant of every eligible other family — see joint_noise_floors), so
    both the layer choice and the floor are per layer. ``selection`` sets
    the layer rule *and* what the y-axis shows:

      * ``"snr"`` — layer with the highest SNR; y = SNR (mean / that layer's
        floor) on a log axis. Families with very different raw scales share
        one axis, with the floor as a single line at SNR = 1. A zero floor
        makes a positive signal's SNR infinite: that layer wins, and the bar
        is drawn overflowing the axis top with an asterisk keyed to a figure
        note. An SNR of zero (metric is 0 at every layer) has no place on a
        log axis and is drawn as a triangle at the axis bottom.
      * ``"raw"`` — layer with the highest raw metric among those clearing
        their own floor (fallback: highest raw metric outright); y = the raw
        mean-over-positions metric on a log axis. Since each bar keeps its
        own floor, floors are drawn as a red tick across each bar rather
        than one shared line.

    ``yscale`` is ``"log"`` (default) or ``"linear"``; linear anchors the bars
    at 0, which is more faithful for bar heights but crushes families that sit
    orders of magnitude apart.

    Variants with no computable floor at any layer are skipped with a
    warning. Returns (figure, JSON payload), or None if nothing is plottable.
    """
    y_is_snr = selection == "snr"
    # {family: ({variant: (snr, layer, mean, sem, floor)}, {layer: floor})}
    fam_stats: dict[
        str,
        tuple[dict[str, tuple[float, int, float, float, dict]], dict[int, dict | None]],
    ] = {}
    variant_order: list[str] = []
    for fam in all_data:
        home = FAMILY_HOME_JUDGE.get(fam, fam)
        self_df = all_data[fam].get(home)
        if self_df is None or self_df.empty:
            print(f"Warning: no self-judge data for {fam}, skipping in joint plot",
                  file=sys.stderr)
            continue
        signal = _per_layer_variant_stats(self_df, layers, metric=metric)
        layer_floors = joint_noise_floors(
            all_data, fam, layers, method=floor_method, metric=metric
        )
        bars: dict[str, tuple[float, int, float, float, dict]] = {}
        for variant, by_layer in signal.items():
            best = _select_best_layer(by_layer, layer_floors, selection)
            if best is None:
                print(
                    f"Warning: no computable noise floor for {fam}/{variant}, "
                    "bar skipped in joint plot",
                    file=sys.stderr,
                )
                continue
            bars[variant] = best
        if not bars:
            print(f"Warning: nothing plottable for {fam}, skipping in joint plot",
                  file=sys.stderr)
            continue
        fam_stats[fam] = (bars, layer_floors)
        for n in bars:
            if n not in variant_order:
                variant_order.append(n)

    if not fam_stats:
        return None

    n_fams = len(fam_stats)
    n_slots = max(len(variant_order), 1)
    group_width = 0.8
    bar_width = group_width / n_slots
    colors = plt.cm.Set2.colors  # type: ignore[attr-defined]
    variant_color = {
        v: colors[i % len(colors)] for i, v in enumerate(variant_order)
    }

    log_scale = yscale == "log"
    # Pre-pass for the axis limits: finite bar values plus (in "raw" mode)
    # the per-bar floor ticks. Infinite-SNR bars are drawn past the top and
    # clipped by the axes; zero-value bars become markers pinned to the
    # bottom edge of a log axis — neither takes part in the limits.
    drawn: list[float] = []
    has_inf = has_zero = False
    for bars_by_variant, _floors in fam_stats.values():
        for snr, _layer, mean, _sem, floor in bars_by_variant.values():
            value = snr if y_is_snr else mean
            if y_is_snr and math.isinf(value):
                has_inf = True
                continue
            drawn.append(value)
            if not y_is_snr:
                drawn.append(floor["upper"])
    positive = [v for v in drawn if v > 0]
    if not log_scale:
        # Bars read from zero; only the top needs headroom for the L labels.
        bottom = 0.0
        top = max(positive) * (1.25 if show_values else 1.15) if positive else 1.0
    elif y_is_snr:
        bottom = min(min(positive) * 0.5, 0.5) if positive else 0.1
        top = max(positive) * 2.0 if positive else 10.0
    else:
        bottom = min(positive) * 0.5 if positive else 1e-6
        top = max(positive) * 3.0 if positive else 1.0

    fig, ax = plt.subplots(figsize=(max(2.2 * n_fams + 3.0, 9.0), 6.5))
    ax.set_yscale(yscale)
    ax.set_ylim(bottom, top)
    if y_is_snr:
        ax.axhspan(bottom, 1.0, color="#d62728", alpha=0.14, zorder=0)
        ax.axhline(1.0, color="#d62728", linewidth=1.4, alpha=0.85, zorder=2)
    # Offsets differ by scale: a factor on log, a slice of the range on linear.
    label_gap = 0.02 * (top - bottom)
    any_zero_floor = False
    for f_idx, (fam, (bars_by_variant, _floors)) in enumerate(fam_stats.items()):
        for v_idx, variant in enumerate(variant_order):
            if variant not in bars_by_variant:
                continue
            snr, layer, mean, sem, floor = bars_by_variant[variant]
            value = snr if y_is_snr else mean
            x = f_idx + (v_idx - (n_slots - 1) / 2) * bar_width
            half_w = bar_width * 0.46
            if not y_is_snr:
                # No shared SNR = 1 line here: each bar carries its own floor.
                # A zero floor has no place on a log axis; pin it to the
                # bottom and dash it, so "pool was all zeros" stays distinct.
                floor_upper = floor["upper"]
                any_zero_floor = any_zero_floor or floor_upper <= 0
                ax.hlines(
                    max(floor_upper, bottom) if log_scale else floor_upper,
                    x - half_w,
                    x + half_w,
                    color="#d62728",
                    linewidth=1.4,
                    alpha=0.9,
                    zorder=3,
                    linestyles="solid" if floor_upper > 0 else "dashed",
                )
            if y_is_snr and math.isinf(value):
                # Unbounded SNR: overflow the axis (the bar is clipped at the
                # top edge); the asterisk keys the figure note.
                ax.bar(
                    x, top * 4.0, width=bar_width * 0.92,
                    color=variant_color[variant],
                    edgecolor="black", linewidth=0.5,
                )
                ax.text(
                    x,
                    top * 1.15 if log_scale else top + label_gap * 1.5,
                    f"L{layer}*",
                    ha="center", va="bottom", fontsize=8, color="#444444",
                )
                continue
            if value == 0.0 and log_scale:
                # 0 has no place on a log axis: triangle at the bottom edge.
                has_zero = True
                ax.plot(
                    x, bottom, marker="v", markersize=7,
                    markerfacecolor=variant_color[variant],
                    markeredgecolor="black", markeredgewidth=0.5,
                    linestyle="none", clip_on=False, zorder=5,
                )
                continue
            if y_is_snr:
                err = sem / floor["upper"] if floor["upper"] > 0 else 0.0
            else:
                err = sem
            bar = ax.bar(
                x,
                value,
                width=bar_width * 0.92,
                # On a log axis the lower whisker has to stay positive.
                yerr=[[min(err, value * 0.95) if log_scale else err], [err]],
                capsize=2,
                color=variant_color[variant],
                edgecolor="black",
                linewidth=0.5,
                error_kw={"linewidth": 1.0},
            )
            if show_values:
                fmt = f"{value:.2f}" if y_is_snr else f"{value:.3g}"
                ax.bar_label(bar, labels=[fmt], padding=2, fontsize=7)
            top_here = value + err
            if not y_is_snr:
                top_here = max(top_here, floor["upper"])
            ax.text(
                x,
                top_here * (1.45 if show_values else 1.12)
                if log_scale
                else top_here + label_gap * (2.6 if show_values else 1.0),
                f"L{layer}",
                ha="center", va="bottom", fontsize=8, color="#444444",
            )
    ax.set_xticks(range(n_fams))
    ax.set_xticklabels(
        [_display_for(f) for f in fam_stats],
        rotation=15,
        ha="right",
        fontsize=12,
    )
    ax.set_ylabel(
        "SNR (metric / noise floor)" if y_is_snr else _METRIC_YLABEL[metric]
    )
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    legend_handles = [
        Patch(
            facecolor=variant_color[v],
            edgecolor="black",
            linewidth=0.5,
            label=_pretty_variant(v),
        )
        for v in variant_order
    ]
    legend_handles.append(
        Line2D(
            [0], [0], color="#d62728", linewidth=1.4,
            label=(
                "noise floor (SNR = 1)"
                if y_is_snr
                else f"per-bar noise floor — {floor_method} "
                     f"p{NOISE_FLOOR_PERCENTILE:g}"
            ),
        )
    )
    if not y_is_snr and any_zero_floor:
        legend_handles.append(
            Line2D([0], [0], color="#d62728", linewidth=1.4, linestyle="dashed",
                   label="noise floor = 0 (pool all zeros)")
        )
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=min(len(legend_handles), 4),
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
        fontsize=10,
    )
    _add_snr_footnotes(fig, has_inf, has_zero)

    layers_str = ", ".join(str(l) for l in layers)
    fig.suptitle(
        f"{_metric_lens_title(metric, lens)}{' — SNR' if y_is_snr else ''}\n"
        f"{VARIANT_TITLE[ll_variant]} — {_SELECTION_TITLE[selection]} "
        f"({layers_str})",
        fontweight="bold",
        y=0.99,
    )
    fig.tight_layout(rect=(0, 0.10, 1, 0.90))

    families_payload: dict[str, dict] = {}
    for fam, (bars_by_variant, layer_floors) in fam_stats.items():
        families_payload[fam] = {
            "display_name": _display_for(fam),
            "home_judge": FAMILY_HOME_JUDGE.get(fam, fam),
            "variants": list(bars_by_variant),
            # "inf" where the floor is 0 and the mean is positive.
            "snr": [_json_snr(b[0]) for b in bars_by_variant.values()],
            # null where the SNR is 0 at every layer — no layer is "best".
            "best_layer": [
                None if (y_is_snr and b[0] == 0.0) else b[1]
                for b in bars_by_variant.values()
            ],
            # False marks a "raw" fallback bar: no layer cleared its floor.
            "above_floor": [
                b[2] > b[4]["upper"] for b in bars_by_variant.values()
            ],
            "means": [b[2] for b in bars_by_variant.values()],
            "sems": [b[3] for b in bars_by_variant.values()],
            "noise_floors": {str(l): f for l, f in layer_floors.items()},
        }
    payload = {
        "mode": (
            "joint_maxlayer_snr" if y_is_snr else "joint_maxrawlayer_metric"
        ),
        "layer_selection": selection,
        "y_axis": "snr" if y_is_snr else _METRIC_COLUMN[metric],
        "selection": _SELECTION_PAYLOAD[selection],
        "layers": layers,
        "lens": lens,
        "ll_variant": ll_variant,
        "ll_method": method_label(ll_variant, lens),
        "metric": metric,
        "metric_column": _METRIC_COLUMN[metric],
        "position_range": [POS_MIN, POS_MAX],
        "noise_floor_method": floor_method,
        "noise_floor_scope": (
            "per layer — pool: every variant of the other families at the "
            "same layer, home judge"
        ),
        "noise_floor_percentile": NOISE_FLOOR_PERCENTILE,
        "yscale": yscale,
        "families": families_payload,
    }
    return fig, payload


# Sequential light→dark by layer depth, distinct from the Set2 variant palette.
LAYER_COLORS = ("#a6cee3", "#1f78b4", "#08306b")


def plot_snr_per_layer(
    all_data: dict[str, dict[str, pd.DataFrame]],
    layers: list[int],
    ll_variant: str,
    floor_method: str = "t",
    metric: str = "cumprob",
    lens: str = "logit_lens",
    show_values: bool = False,
    yscale: str = "log",
) -> tuple[plt.Figure, dict] | None:
    """Companion to plot_joint_maxlayer: every layer's SNR, not just the best.

    One subplot per family; variants grouped on the x-axis with one bar per
    layer, y = SNR (mean over positions / that layer's noise floor — see
    joint_noise_floors) on a ``yscale`` axis, floor line at SNR = 1.
    Infinite SNR (zero floor, positive signal) overflows the axis top with
    an asterisk; an SNR of zero is a triangle at the axis bottom; layers
    with no computable floor are left empty. Returns (figure, JSON payload),
    or None if nothing is plottable.
    """
    log_scale = yscale == "log"
    # fam -> (variants, {(variant, layer): (snr, sem)}, {layer: floor|None})
    fam_data: dict[
        str,
        tuple[list[str], dict[tuple[str, int], tuple[float, float]], dict[int, dict | None]],
    ] = {}
    for fam in all_data:
        home = FAMILY_HOME_JUDGE.get(fam, fam)
        self_df = all_data[fam].get(home)
        if self_df is None or self_df.empty:
            continue
        signal = _per_layer_variant_stats(self_df, layers, metric=metric)
        layer_floors = joint_noise_floors(
            all_data, fam, layers, method=floor_method, metric=metric
        )
        cells: dict[tuple[str, int], tuple[float, float]] = {}
        for variant, by_layer in signal.items():
            for layer, (mean, sem) in by_layer.items():
                floor = layer_floors.get(layer)
                if floor is None:
                    continue
                cells[(variant, layer)] = (_snr_value(mean, floor["upper"]), sem)
        if cells:
            fam_data[fam] = (list(signal), cells, layer_floors)
    if not fam_data:
        return None

    n = len(fam_data)
    ncols = 2
    nrows = (n + 1) // 2
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(7.5 * ncols, 5.2 * nrows), squeeze=False
    )

    n_layers = max(len(layers), 1)
    group_width = 0.8
    bar_width = group_width / n_layers
    has_inf = has_zero = False
    for idx, (fam, (variants, cells, layer_floors)) in enumerate(fam_data.items()):
        r, c = divmod(idx, ncols)
        ax = axes[r, c]
        xs_center = np.arange(len(variants), dtype=float)
        # Subplot limits come from the finite SNRs; infinite bars overflow
        # the top (clipped), zero bars are markers at the bottom of a log axis.
        finite = [s for (s, _sem) in cells.values() if not math.isinf(s)]
        positive = [s for s in finite if s > 0]
        if log_scale:
            bottom = min(min(positive) * 0.5, 0.5) if positive else 0.1
            top = max(positive) * 2.0 if positive else 10.0
        else:
            bottom = 0.0
            top = max(positive) * 1.15 if positive else 1.0
        ax.set_yscale(yscale)
        ax.set_ylim(bottom, top)
        for l_idx, layer in enumerate(layers):
            offset = (l_idx - (n_layers - 1) / 2) * bar_width
            layer_color = LAYER_COLORS[l_idx % len(LAYER_COLORS)]
            for v_idx, variant in enumerate(variants):
                cell = cells.get((variant, layer))
                if cell is None:
                    continue
                snr, sem = cell
                x = xs_center[v_idx] + offset
                if math.isinf(snr):
                    has_inf = True
                    ax.bar(
                        x, top * 4.0, width=bar_width * 0.92,
                        color=layer_color, edgecolor="black", linewidth=0.5,
                    )
                    ax.text(
                        x,
                        top * 1.1 if log_scale else top * 1.02,
                        "*",
                        ha="center", va="bottom", fontsize=9, color="#444444",
                    )
                    continue
                if snr == 0.0 and log_scale:
                    has_zero = True
                    ax.plot(
                        x, bottom, marker="v", markersize=5,
                        markerfacecolor=layer_color,
                        markeredgecolor="black", markeredgewidth=0.5,
                        linestyle="none", clip_on=False, zorder=5,
                    )
                    continue
                floor_upper = layer_floors[layer]["upper"]
                err = sem / floor_upper if floor_upper > 0 else 0.0
                bar = ax.bar(
                    x,
                    snr,
                    width=bar_width * 0.92,
                    yerr=[[min(err, snr * 0.95) if log_scale else err], [err]],
                    capsize=2,
                    color=layer_color,
                    edgecolor="black",
                    linewidth=0.5,
                    error_kw={"linewidth": 0.8},
                )
                if show_values:
                    ax.bar_label(bar, labels=[f"{snr:.2f}"], padding=2, fontsize=6)
        ax.axhspan(bottom, 1.0, color="#d62728", alpha=0.14, zorder=0)
        ax.axhline(1.0, color="#d62728", linewidth=1.2, alpha=0.85, zorder=2)
        ax.set_xticks(xs_center)
        ax.set_xticklabels(
            [_pretty_variant(v) for v in variants],
            rotation=40,
            ha="right",
            fontsize=11,
        )
        ax.set_ylabel("SNR (metric / noise floor)")
        ax.set_title(_display_for(fam), fontweight="bold", pad=10)
        ax.grid(axis="y", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for idx in range(n, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r, c].set_visible(False)

    legend_handles = [
        Patch(
            facecolor=LAYER_COLORS[i % len(LAYER_COLORS)],
            edgecolor="black",
            linewidth=0.5,
            label=f"Layer {layer}",
        )
        for i, layer in enumerate(layers)
    ]
    legend_handles.append(
        Line2D([0], [0], color="#d62728", linewidth=1.4,
               label="noise floor (SNR = 1)")
    )
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=len(legend_handles),
        frameon=False,
        bbox_to_anchor=(0.5, -0.01),
    )
    _add_snr_footnotes(fig, has_inf, has_zero)

    layers_str = ", ".join(str(l) for l in layers)
    fig.suptitle(
        f"{_metric_lens_title(metric, lens)} — SNR per Layer\n"
        f"{VARIANT_TITLE[ll_variant]} — layers {layers_str}",
        fontweight="bold",
        y=0.99,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.93), h_pad=5.0)

    families_payload: dict[str, dict] = {}
    for fam, (variants, cells, layer_floors) in fam_data.items():
        families_payload[fam] = {
            "display_name": _display_for(fam),
            "home_judge": FAMILY_HOME_JUDGE.get(fam, fam),
            "variants": variants,
            "snr": {
                str(layer): [
                    _json_snr(cells[(v, layer)][0])
                    if (v, layer) in cells
                    else None
                    for v in variants
                ]
                for layer in layers
            },
            "noise_floors": {str(l): f for l, f in layer_floors.items()},
        }
    payload = {
        "mode": "snr_per_layer",
        "noise_floor_scope": (
            "per layer — pool: every variant of the other families at the "
            "same layer, home judge"
        ),
        "yscale": yscale,
        "layers": layers,
        "lens": lens,
        "ll_variant": ll_variant,
        "ll_method": method_label(ll_variant, lens),
        "metric": metric,
        "metric_column": _METRIC_COLUMN[metric],
        "position_range": [POS_MIN, POS_MAX],
        "noise_floor_method": floor_method,
        "noise_floor_percentile": NOISE_FLOOR_PERCENTILE,
        "families": families_payload,
    }
    return fig, payload


def _draw_family_subplot_with_floor(
    ax: plt.Axes,
    family: str,
    names: list[str],
    means: list[float],
    sems: list[float],
    cross_floor: dict | None,
    ylabel: str,
    show_values: bool = False,
) -> None:
    _draw_family_subplot(
        ax, family, names, means, sems, ylabel, show_values=show_values
    )
    if cross_floor is None:
        return
    upper = cross_floor["upper"]
    n = cross_floor["n_pool"]
    method = cross_floor["method"]
    ax.axhspan(0, upper, color="#d62728", alpha=0.14, zorder=0)
    ax.axhline(
        upper,
        color="#d62728",
        linewidth=1.4,
        alpha=0.85,
        zorder=2,
        label=f"noise floor — {method} p{NOISE_FLOOR_PERCENTILE:g} (n={n})",
    )
    cur_top = ax.get_ylim()[1]
    ax.set_ylim(top=max(cur_top, upper * 1.15))


def plot_layer_cross_floor(
    family_to_judges: dict[str, dict[str, pd.DataFrame]],
    floors: dict[str, dict | None],
    layer: int,
    ll_variant: str,
    show_values: bool = False,
    floor_method: str = "t",
    metric: str = "cumprob",
    lens: str = "logit_lens",
) -> plt.Figure:
    """Self-judge bars + noise-floor line from home judge applied to other families."""
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
    ylabel = _METRIC_YLABEL[metric]

    for idx, (fam, judge_dfs) in enumerate(items):
        home = FAMILY_HOME_JUDGE.get(fam, fam)
        r, c = divmod(idx, ncols)
        self_df = judge_dfs.get(home)
        if self_df is None or self_df.empty:
            axes[r, c].set_visible(False)
            continue
        names, means, sems = compute_bar_stats(self_df, metric=metric)
        _draw_family_subplot_with_floor(
            axes[r, c],
            fam,
            names,
            means,
            sems,
            floors.get(fam),
            ylabel,
            show_values=show_values,
        )

    for idx in range(n, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r, c].set_visible(False)

    method_descriptions = {
        "t": (
            f"Student-t one-sided p{NOISE_FLOOR_PERCENTILE:g} prediction bound"
        ),
        "normal": (
            f"Normal one-sided p{NOISE_FLOOR_PERCENTILE:g} prediction bound"
        ),
        "empirical": f"empirical p{NOISE_FLOOR_PERCENTILE:g}",
    }
    legend_handles = [
        Patch(
            facecolor="#d62728",
            alpha=0.14,
            edgecolor="#d62728",
            linewidth=1.4,
            label=(
                f"noise floor — {method_descriptions.get(floor_method, floor_method)} "
                "of home judge on other families' variants"
            ),
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
        _suptitle_for(ll_variant, layer, metric=metric, lens=lens),
        fontweight="bold",
        y=0.99,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.93), h_pad=5.0)
    return fig


def plot_layer(
    family_stats: dict[str, tuple[list[str], list[float], list[float]]],
    layer: int,
    ll_variant: str,
    normalize: bool = False,
    qer_by_family: dict[str, dict[str, tuple[float, float]]] | None = None,
    show_values: bool = False,
    metric: str = "cumprob",
    lens: str = "logit_lens",
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

    ylabel = _METRIC_YLABEL[metric]
    if normalize:
        ylabel = "Normalised " + ylabel.removeprefix("Mean ")
    drew_any_qer = False
    for idx, (fam, (names, means, sems)) in enumerate(items):
        r, c = divmod(idx, ncols)
        # Normalised QER would conflate two different scales — skip in that mode.
        fam_qer = None if normalize or qer_by_family is None else qer_by_family.get(fam)
        drew = _draw_family_subplot(
            axes[r, c],
            fam,
            names,
            means,
            sems,
            ylabel,
            qer_map=fam_qer,
            show_values=show_values,
        )
        drew_any_qer = drew_any_qer or drew

    # Hide unused cells.
    for idx in range(n, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r, c].set_visible(False)

    norm_tag = " (normalised)" if normalize else ""
    fig.suptitle(
        _suptitle_for(ll_variant, layer, norm_tag, metric=metric, lens=lens),
        fontweight="bold",
        y=0.99,
    )
    bottom_rect = 0.0
    if drew_any_qer:
        legend_handles = [
            Line2D(
                [0],
                [0],
                color=QER_TICK_COLOR,
                linewidth=2.4,
                label="Trigger QER — right axis (± stderr band)",
            ),
        ]
        fig.legend(
            handles=legend_handles,
            loc="lower center",
            ncol=1,
            frameon=False,
            bbox_to_anchor=(0.5, -0.01),
        )
        bottom_rect = 0.03
    fig.tight_layout(rect=(0, bottom_rect, 1, 0.93), h_pad=5.0)
    return fig


def plot_layer_cross(
    family_to_judges: dict[str, dict[str, pd.DataFrame]],
    layer: int,
    judges: list[str],
    ll_variant: str,
    show_values: bool = False,
    metric: str = "cumprob",
    lens: str = "logit_lens",
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

    ylabel = _METRIC_YLABEL[metric]
    for idx, (fam, judge_dfs) in enumerate(items):
        variant_order, per_judge = compute_variant_stats_by_judge(
            judge_dfs, metric=metric
        )
        r, c = divmod(idx, ncols)
        _draw_cross_family_subplot(
            axes[r, c],
            fam,
            variant_order,
            per_judge,
            judges,
            ylabel,
            show_values=show_values,
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
        _suptitle_for(ll_variant, layer, metric=metric, lens=lens),
        fontweight="bold",
        y=0.99,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.93), h_pad=5.0)
    return fig


# ── JSON sidecars ───────────────────────────────────────────────────────────


def _display_for(fam: str) -> str:
    return DISPLAY_NAMES.get(fam, fam.replace("_", " ").title())


def _build_flat_payload(
    family_stats: dict[str, tuple[list[str], list[float], list[float]]],
    layer: int,
    ll_variant: str,
    normalize: bool,
    qer_by_family: dict[str, dict[str, tuple[float, float]]] | None,
    qer_mode: str | None,
    metric: str = "cumprob",
    lens: str = "logit_lens",
) -> dict:
    families: dict[str, dict] = {}
    for fam, (names, means, sems) in family_stats.items():
        entry: dict = {
            "display_name": _display_for(fam),
            "variants": names,
            "means": means,
            "sems": sems,
        }
        if qer_by_family and fam in qer_by_family:
            entry["qer"] = {
                v: {"mean": m, "stderr": s}
                for v, (m, s) in qer_by_family[fam].items()
            }
            entry["qer_mode"] = qer_mode
        families[fam] = entry
    return {
        "mode": "flat",
        "layer": layer,
        "lens": lens,
        "ll_variant": ll_variant,
        "ll_method": method_label(ll_variant, lens),
        "metric": metric,
        "metric_column": _METRIC_COLUMN[metric],
        "position_range": [POS_MIN, POS_MAX],
        "normalize": normalize,
        "families": families,
    }


def _build_cross_payload(
    family_to_judges: dict[str, dict[str, pd.DataFrame]],
    layer: int,
    ll_variant: str,
    judges: list[str],
    metric: str = "cumprob",
    lens: str = "logit_lens",
) -> dict:
    families: dict[str, dict] = {}
    for fam, judge_dfs in family_to_judges.items():
        variant_order, per_judge = compute_variant_stats_by_judge(
            judge_dfs, metric=metric
        )
        home = FAMILY_HOME_JUDGE.get(fam, fam)
        by_judge: dict[str, dict] = {}
        for judge, stats in per_judge.items():
            means: list[float | None] = []
            sems: list[float | None] = []
            for v in variant_order:
                if v in stats:
                    m, s = stats[v]
                    means.append(m)
                    sems.append(s)
                else:
                    means.append(None)
                    sems.append(None)
            by_judge[judge] = {
                "means": means,
                "sems": sems,
                "is_self": judge == home,
            }
        families[fam] = {
            "display_name": _display_for(fam),
            "home_judge": home,
            "variants": variant_order,
            "by_judge": by_judge,
        }
    return {
        "mode": "cross",
        "layer": layer,
        "lens": lens,
        "ll_variant": ll_variant,
        "ll_method": method_label(ll_variant, lens),
        "metric": metric,
        "metric_column": _METRIC_COLUMN[metric],
        "position_range": [POS_MIN, POS_MAX],
        "judges": judges,
        "families": families,
    }


def _build_noise_floor_payload(
    family_to_judges: dict[str, dict[str, pd.DataFrame]],
    floors: dict[str, dict | None],
    layer: int,
    ll_variant: str,
    method: str,
    metric: str = "cumprob",
    lens: str = "logit_lens",
) -> dict:
    families: dict[str, dict] = {}
    for fam, judge_dfs in family_to_judges.items():
        home = FAMILY_HOME_JUDGE.get(fam, fam)
        self_df = judge_dfs.get(home)
        if self_df is None or self_df.empty:
            names, means, sems = [], [], []
        else:
            names, means, sems = compute_bar_stats(self_df, metric=metric)
        families[fam] = {
            "display_name": _display_for(fam),
            "home_judge": home,
            "variants": names,
            "means": means,
            "sems": sems,
            "noise_floor": floors.get(fam),
        }
    return {
        "mode": "noise_floor",
        "layer": layer,
        "lens": lens,
        "ll_variant": ll_variant,
        "ll_method": method_label(ll_variant, lens),
        "metric": metric,
        "metric_column": _METRIC_COLUMN[metric],
        "position_range": [POS_MIN, POS_MAX],
        "noise_floor_method": method,
        "noise_floor_percentile": NOISE_FLOOR_PERCENTILE,
        "families": families,
    }


def _save_payload(payload: dict, fig_path: Path, diffing_base: str | None) -> None:
    json_path = fig_path.with_suffix(".json")
    payload = {DIFFING_BASE_COLUMN: diffing_base, **payload}
    json_path.write_text(json.dumps(payload, indent=2))
    print(f"Saved {json_path}")


def _emit_figure(
    fig: plt.Figure, payload: dict, args: argparse.Namespace, out_stem: str
) -> None:
    """Save figure + JSON sidecar under args.output, or show interactively."""
    if args.output is not None:
        args.output.mkdir(parents=True, exist_ok=True)
        out_path = args.output / f"{out_stem}.{args.format}"
        fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
        print(f"Saved {out_path}")
        plt.close(fig)
        _save_payload(payload, out_path, args.diffing_base)
    else:
        plt.show()


# ── Main ────────────────────────────────────────────────────────────────────


def _run_flat(args: argparse.Namespace) -> None:
    all_data: dict[str, pd.DataFrame] = {}
    for fam in args.families:
        df = load_family_data(args.results_base, fam, args.ll_variant, args.lens)
        if df is not None:
            all_data[fam] = df

    if not all_data:
        print("Error: no data found.", file=sys.stderr)
        sys.exit(1)

    args.diffing_base = resolve_diffing_base(
        [_csv_path_flat(args.results_base, fam, args.ll_variant) for fam in all_data]
    )
    print(f"Diffing base: {args.diffing_base}")

    layers = sorted(set().union(*(df["layer"].unique() for df in all_data.values())))
    suffix = file_suffix(args.ll_variant, args.lens)

    qer_by_family: dict[str, dict[str, tuple[float, float]]] | None = None
    if args.qer_base is not None:
        qer_by_family = {}
        for fam, df in all_data.items():
            variants = list(dict.fromkeys(df["model"].tolist()))
            fam_qer = load_qer_for_family(args.qer_base, fam, variants, args.qer_mode)
            if fam_qer:
                qer_by_family[fam] = fam_qer

    for layer in layers:
        family_stats: dict[str, tuple[list[str], list[float], list[float]]] = {}
        for fam, df in all_data.items():
            layer_df = df[df["layer"] == layer]
            if layer_df.empty:
                continue
            names, means, sems = compute_bar_stats(layer_df, metric=args.metric)
            if names:
                family_stats[fam] = (names, means, sems)

        if not family_stats:
            continue

        fig = plot_layer(
            family_stats,
            layer,
            args.ll_variant,
            normalize=args.normalize,
            qer_by_family=qer_by_family,
            show_values=args.bar_values,
            metric=args.metric,
            lens=args.lens,
        )

        if args.output is not None:
            args.output.mkdir(parents=True, exist_ok=True)
            out_path = (
                args.output
                / f"{_METRIC_STEM[args.metric]}_raffgraph_layer{layer}{suffix}.{args.format}"
            )
            fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
            print(f"Saved {out_path}")
            plt.close(fig)
            payload = _build_flat_payload(
                family_stats,
                layer,
                args.ll_variant,
                args.normalize,
                qer_by_family,
                args.qer_mode if args.qer_base is not None else None,
                metric=args.metric,
                lens=args.lens,
            )
            _save_payload(payload, out_path, args.diffing_base)
        else:
            plt.show()


def _run_cross(args: argparse.Namespace) -> None:
    all_data: dict[str, dict[str, pd.DataFrame]] = {}
    for fam in args.families:
        judge_dfs = load_cross_family_data(
            args.cross_dir, fam, args.judges, args.ll_variant, args.lens
        )
        if judge_dfs:
            all_data[fam] = judge_dfs

    if not all_data:
        print("Error: no cross-mode data found.", file=sys.stderr)
        sys.exit(1)

    args.diffing_base = resolve_diffing_base(
        [
            _csv_path_cross(args.cross_dir, fam, judge, args.ll_variant)
            for fam, judge_dfs in all_data.items()
            for judge in judge_dfs
        ]
    )
    print(f"Diffing base: {args.diffing_base}")

    layers_union: set[int] = set()
    for judge_dfs in all_data.values():
        for df in judge_dfs.values():
            layers_union.update(df["layer"].unique().tolist())
    layers = sorted(layers_union)
    suffix = file_suffix(args.ll_variant, args.lens)

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

        payload: dict
        metric_stem = _METRIC_STEM[args.metric]
        if args.noise_floor:
            floors = {
                fam: cross_family_noise_floor(
                    all_data,
                    fam,
                    layer,
                    method=args.noise_floor_method,
                    metric=args.metric,
                )
                for fam in family_to_judges
            }
            fig = plot_layer_cross_floor(
                family_to_judges,
                floors,
                layer,
                args.ll_variant,
                show_values=args.bar_values,
                floor_method=args.noise_floor_method,
                metric=args.metric,
                lens=args.lens,
            )
            out_stem = (
                f"{metric_stem}_raffgraph_noisefloor_{args.noise_floor_method}"
                f"_layer{layer}{suffix}"
            )
            payload = _build_noise_floor_payload(
                family_to_judges,
                floors,
                layer,
                args.ll_variant,
                args.noise_floor_method,
                metric=args.metric,
                lens=args.lens,
            )
        else:
            fig = plot_layer_cross(
                family_to_judges,
                layer,
                args.judges,
                args.ll_variant,
                show_values=args.bar_values,
                metric=args.metric,
                lens=args.lens,
            )
            out_stem = f"{metric_stem}_raffgraph_cross_layer{layer}{suffix}"
            payload = _build_cross_payload(
                family_to_judges,
                layer,
                args.ll_variant,
                args.judges,
                metric=args.metric,
                lens=args.lens,
            )

        if args.output is not None:
            args.output.mkdir(parents=True, exist_ok=True)
            out_path = args.output / f"{out_stem}.{args.format}"
            fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
            print(f"Saved {out_path}")
            plt.close(fig)
            _save_payload(payload, out_path, args.diffing_base)
        else:
            plt.show()

    # Joint over-layers figures, all families on one log axis: "snr" picks
    # each bar's layer by highest per-layer SNR and plots the SNR; "raw"
    # picks the highest raw per-layer metric among the layers clearing their
    # own floor and plots that metric, with a floor tick per bar.
    if args.noise_floor:
        # Only a non-default scale is tagged, so default names stay put.
        scope_stem = "" if args.joint_scale == "log" else "_linear"
        for selection, stem_key in (
            ("snr", "joint_maxlayer_snr"),
            ("raw", "joint_maxrawlayer_metric"),
        ):
            joint = plot_joint_maxlayer(
                all_data,
                layers,
                args.ll_variant,
                floor_method=args.noise_floor_method,
                metric=args.metric,
                lens=args.lens,
                show_values=args.bar_values,
                selection=selection,
                yscale=args.joint_scale,
            )
            if joint is not None:
                fig, payload = joint
                _emit_figure(
                    fig,
                    payload,
                    args,
                    f"{_METRIC_STEM[args.metric]}_raffgraph_{stem_key}"
                    f"_{args.noise_floor_method}{scope_stem}{suffix}",
                )

        per_layer = plot_snr_per_layer(
            all_data,
            layers,
            args.ll_variant,
            floor_method=args.noise_floor_method,
            metric=args.metric,
            lens=args.lens,
            show_values=args.bar_values,
            yscale=args.joint_scale,
        )
        if per_layer is not None:
            fig, payload = per_layer
            _emit_figure(
                fig,
                payload,
                args,
                f"{_METRIC_STEM[args.metric]}_raffgraph_snr_per_layer"
                f"_{args.noise_floor_method}{scope_stem}{suffix}",
            )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.cross_dir is not None:
        _run_cross(args)
    else:
        _run_flat(args)


if __name__ == "__main__":
    main()
