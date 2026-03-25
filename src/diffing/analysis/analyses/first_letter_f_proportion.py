"""Letter proportion analysis for logit lens and patchscope diff tokens."""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import pandas as pd

from ..adl_explorer import ADLExplorer
from ..formatting import display_token, fmt_prob, get_first_letter, normalize_token

if TYPE_CHECKING:
    from matplotlib.figure import Figure

LETTER = "f"


# ---------------------------------------------------------------------------
# Logit lens letter proportion
# ---------------------------------------------------------------------------

def _ll_letter_proportion(
    explorer: ADLExplorer, layer: int, pos: int,
) -> dict[str, float]:
    """Compute letter proportion for diff and diff_inv at one (layer, pos)."""
    results: dict[str, float] = {}
    for variant, probs_attr, indices_attr in [
        ("diff", "top_k_probs", "top_k_indices"),
        ("diff_inv", "inv_probs", "inv_indices"),
    ]:
        entry = explorer.logit_lens[layer][pos].get("diff")
        if entry is None:
            results[variant] = 0.0
            continue
        indices = getattr(entry, indices_attr)
        tokens = explorer.decode_tokens(indices)
        total = len(tokens)
        count = sum(
            1 for t in tokens if t.strip() and get_first_letter(t.strip()) == LETTER
        )
        results[variant] = count / total if total > 0 else 0.0
    return results


def f_start_logit_lens_table(
    explorer: ADLExplorer,
    *,
    positions: list[int] | None = None,
) -> pd.DataFrame:
    """DataFrame of letter proportions (diff/diff_inv) per position and layer.

    Returns a DataFrame with a MultiIndex of (position, layer) and columns
    ``diff`` and ``diff_inv`` formatted as percentages.
    """
    if positions is None:
        positions = list(range(-3, 6))

    rows = []
    for pos in positions:
        for layer in explorer.layers:
            if pos not in explorer.logit_lens[layer]:
                continue
            props = _ll_letter_proportion(explorer, layer, pos)
            rows.append({
                "position": pos,
                "layer": layer,
                "diff": f"{props['diff']:.2%}",
                "diff_inv": f"{props['diff_inv']:.2%}",
            })
    return pd.DataFrame(rows)


def f_start_logit_lens_plot(
    explorer: ADLExplorer,
    *,
    positions: list[int] | None = None,
) -> Figure:
    """Matplotlib figure: letter proportion in logit lens diff across positions."""
    if positions is None:
        positions = list(range(-3, 6))

    plot_data: dict[int, dict[str, list]] = {
        layer: {"positions": [], "diff": [], "diff_inv": []}
        for layer in explorer.layers
    }

    for pos in positions:
        for layer in explorer.layers:
            if pos not in explorer.logit_lens[layer]:
                continue
            props = _ll_letter_proportion(explorer, layer, pos)
            plot_data[layer]["positions"].append(pos)
            plot_data[layer]["diff"].append(props["diff"] * 100)
            plot_data[layer]["diff_inv"].append(props["diff_inv"] * 100)

    n_layers = len(explorer.layers)
    fig, axes = plt.subplots(1, n_layers, figsize=(6 * n_layers, 4), sharey=True)
    if n_layers == 1:
        axes = [axes]

    for ax, layer in zip(axes, explorer.layers):
        ax.plot(plot_data[layer]["positions"], plot_data[layer]["diff"], marker="o", label="diff")
        ax.plot(plot_data[layer]["positions"], plot_data[layer]["diff_inv"], marker="s", label="diff_inv")
        ax.axvline(x=-0.5, color="gray", linestyle="--", alpha=0.7, label="model answer start")
        ax.set_title(f"Layer {layer}")
        ax.set_xlabel("Position")
        ax.set_ylabel(f"{LETTER.upper()} proportion (%)")
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"Proportion of {LETTER.upper()}-Starting Tokens in Logit Lens Top Predictions")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Patchscope letter proportion
# ---------------------------------------------------------------------------

def _ps_letter_proportion(
    explorer: ADLExplorer, layer: int, pos: int,
) -> float:
    """Compute letter proportion for patchscope diff at one (layer, pos)."""
    entry = explorer.patchscope[layer].get(pos, {}).get("diff")
    if entry is None:
        return 0.0
    tokens = entry.tokens_at_best_scale
    total = len(tokens)
    count = sum(
        1 for t in tokens if t.strip() and get_first_letter(t.strip()) == LETTER
    )
    return count / total if total > 0 else 0.0


def f_start_patchscope_plot(
    explorer: ADLExplorer,
    *,
    positions: list[int] | None = None,
) -> Figure:
    """Matplotlib figure: letter proportion in patchscope diff across positions."""
    if positions is None:
        positions = list(range(-3, 6))

    plot_data: dict[int, dict[str, list]] = {
        layer: {"positions": [], "diff_pct": []}
        for layer in explorer.layers
    }

    for pos in positions:
        for layer in explorer.layers:
            pct = _ps_letter_proportion(explorer, layer, pos)
            plot_data[layer]["positions"].append(pos)
            plot_data[layer]["diff_pct"].append(pct * 100)

    n_layers = len(explorer.layers)
    fig, axes = plt.subplots(1, n_layers, figsize=(6 * n_layers, 4), sharey=True)
    if n_layers == 1:
        axes = [axes]

    for ax, layer in zip(axes, explorer.layers):
        ax.plot(
            plot_data[layer]["positions"],
            plot_data[layer]["diff_pct"],
            marker="o", label="diff", color="tab:blue",
        )
        ax.axvline(x=-0.5, color="gray", linestyle="--", alpha=0.7, label="model answer start")
        ax.set_title(f"Layer {layer}")
        ax.set_xlabel("Position")
        ax.set_ylabel(f"{LETTER.upper()} proportion (%)")
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"PatchScope Diff: Proportion of {LETTER.upper()}-Starting Tokens")
    fig.tight_layout()
    return fig
