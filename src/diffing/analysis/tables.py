"""Table builders for logit lens and patchscope data.

These functions render raw ADL data into DataFrames and are always run
by the CLI (they are not selectable via ``--analyses``).
"""

from __future__ import annotations

from collections import defaultdict

import pandas as pd

from .adl_explorer import ADLExplorer
from .formatting import concat_layer_dfs, display_token, fmt_prob, normalize_token

# ---------------------------------------------------------------------------
# Logit lens helpers
# ---------------------------------------------------------------------------

# (prefix key, probs attr, indices attr)
_LL_VARIANTS: list[tuple[str, str, str]] = [
    ("base", "top_k_probs", "top_k_indices"),
    ("base_inv", "inv_probs", "inv_indices"),
    ("ft", "top_k_probs", "top_k_indices"),
    ("ft_inv", "inv_probs", "inv_indices"),
    ("diff", "top_k_probs", "top_k_indices"),
    ("diff_inv", "inv_probs", "inv_indices"),
]


def _ll_prefix_key(variant_name: str) -> str:
    """Map variant name to the prefix key in ``explorer.logit_lens``."""
    return variant_name.removesuffix("_inv")


# ---------------------------------------------------------------------------
# Logit lens – single position
# ---------------------------------------------------------------------------

def _ll_single_layer(
    explorer: ADLExplorer, layer: int, pos: int, *, max_rows: int | None = None,
) -> pd.DataFrame:
    cols: dict[str, list[str]] = {}
    for variant_name, probs_attr, indices_attr in _LL_VARIANTS:
        key = _ll_prefix_key(variant_name)
        entry = explorer.logit_lens[layer][pos][key]
        probs = getattr(entry, probs_attr)
        indices = getattr(entry, indices_attr)
        tokens = explorer.decode_tokens(indices)
        cols[variant_name] = [
            f"{display_token(t)} ({fmt_prob(p)})"
            for t, p in zip(tokens, probs.tolist())
        ]
    df = pd.DataFrame(cols)
    if max_rows is not None:
        df = df.head(max_rows)
    return df


def logit_lens_position(
    explorer: ADLExplorer,
    *,
    position: int = -1,
    max_rows: int | None = None,
) -> pd.DataFrame:
    """MultiIndex DataFrame: 6 logit-lens variants per layer for one position.

    Columns are ``(layer_<N>, variant)`` where variant is one of
    ``base``, ``base_inv``, ``ft``, ``ft_inv``, ``diff``, ``diff_inv``.
    """
    dfs = []
    for layer in explorer.layers:
        df = _ll_single_layer(explorer, layer, position, max_rows=max_rows)
        df.columns = pd.MultiIndex.from_product([[f"layer_{layer}"], df.columns])
        dfs.append(df)
    return concat_layer_dfs(dfs)


# ---------------------------------------------------------------------------
# Logit lens – aggregated
# ---------------------------------------------------------------------------

def _ll_agg_single_layer(
    explorer: ADLExplorer, layer: int, *, max_rows: int = 100,
) -> pd.DataFrame:
    positions = explorer.logit_lens_positions[layer]
    n_positions = len(positions)

    cols: dict[str, list[str]] = {}
    for variant_name, probs_attr, indices_attr in _LL_VARIANTS:
        key = _ll_prefix_key(variant_name)
        token_prob_sum: dict[str, float] = defaultdict(float)

        for pos in positions:
            entry = explorer.logit_lens[layer][pos].get(key)
            if entry is None:
                continue
            probs = getattr(entry, probs_attr)
            indices = getattr(entry, indices_attr)
            tokens = explorer.decode_tokens(indices)
            for t, p in zip(tokens, probs.tolist()):
                token_prob_sum[t] += p

        token_avg = {t: s / n_positions for t, s in token_prob_sum.items()}
        sorted_tokens = sorted(token_avg, key=lambda t: (-token_avg[t], t))[:max_rows]
        cols[variant_name] = [
            f"{display_token(t)} ({fmt_prob(token_avg[t])})" for t in sorted_tokens
        ]

    # Pad columns to equal length
    max_len = max(len(v) for v in cols.values())
    for k in cols:
        cols[k] += [""] * (max_len - len(cols[k]))
    return pd.DataFrame(cols)


def logit_lens_aggregated(
    explorer: ADLExplorer,
    *,
    max_rows: int = 100,
) -> pd.DataFrame:
    """MultiIndex DataFrame: tokens ranked by avg probability across all positions.

    Same 6 variant columns as :func:`logit_lens_position`, but each token's
    probability is averaged over every position in the layer.
    """
    dfs = []
    for layer in explorer.layers:
        df = _ll_agg_single_layer(explorer, layer, max_rows=max_rows)
        df.columns = pd.MultiIndex.from_product([[f"layer_{layer}"], df.columns])
        dfs.append(df)
    return concat_layer_dfs(dfs)


# ---------------------------------------------------------------------------
# Patchscope helpers
# ---------------------------------------------------------------------------

_PS_VARIANTS = ["base", "ft", "diff"]


# ---------------------------------------------------------------------------
# Patchscope – single position
# ---------------------------------------------------------------------------

def _ps_single_layer(
    explorer: ADLExplorer, layer: int, pos: int,
) -> pd.DataFrame:
    cols: dict[str, list[str]] = {}
    for key in _PS_VARIANTS:
        entry = explorer.patchscope[layer][pos][key]
        selected = {normalize_token(t) for t in entry.selected_tokens}
        cols[key] = [
            f"{display_token(t)} ({fmt_prob(p)})"
            + (" ✅" if normalize_token(t) in selected else "")
            for t, p in zip(entry.tokens_at_best_scale, entry.token_probs)
        ]

    max_len = max(len(v) for v in cols.values())
    for k in cols:
        cols[k] += [""] * (max_len - len(cols[k]))
    return pd.DataFrame(cols)


def patchscope_position(
    explorer: ADLExplorer,
    *,
    position: int = -1,
) -> pd.DataFrame:
    """MultiIndex DataFrame: 3 patchscope variants per layer for one position.

    Columns are ``(layer_<N>, variant)`` where variant is one of
    ``base``, ``ft``, ``diff``.  Tokens selected by the grader get a
    checkmark suffix.
    """
    dfs = []
    for layer in explorer.layers:
        df = _ps_single_layer(explorer, layer, position)
        df.columns = pd.MultiIndex.from_product([[f"layer_{layer}"], df.columns])
        dfs.append(df)
    return concat_layer_dfs(dfs)


# ---------------------------------------------------------------------------
# Patchscope – aggregated
# ---------------------------------------------------------------------------

def _ps_agg_single_layer(
    explorer: ADLExplorer, layer: int,
) -> pd.DataFrame:
    positions = explorer.patchscope_positions[layer]
    n_ps = len(positions)

    cols: dict[str, list[str]] = {}
    for key in _PS_VARIANTS:
        token_prob_sum: dict[str, float] = defaultdict(float)
        ever_selected: set[str] = set()

        for pos in positions:
            entry = explorer.patchscope[layer][pos].get(key)
            if entry is None:
                continue
            for t, p in zip(entry.tokens_at_best_scale, entry.token_probs):
                token_prob_sum[t] += p
            ever_selected.update(normalize_token(t) for t in entry.selected_tokens)

        token_avg = {t: s / n_ps for t, s in token_prob_sum.items()}
        sorted_tokens = sorted(token_avg, key=lambda t: (-token_avg[t], t))
        cols[key] = [
            f"{display_token(t)} ({fmt_prob(token_avg[t])})"
            + (" ✅" if normalize_token(t) in ever_selected else "")
            for t in sorted_tokens
        ]

    max_len = max(len(v) for v in cols.values())
    for k in cols:
        cols[k] += [""] * (max_len - len(cols[k]))
    return pd.DataFrame(cols)


def patchscope_aggregated(
    explorer: ADLExplorer,
) -> pd.DataFrame:
    """MultiIndex DataFrame: tokens ranked by avg probability across patchscope positions.

    Checkmark is added if the token was in ``selected_tokens`` for *any*
    position.
    """
    dfs = []
    for layer in explorer.layers:
        df = _ps_agg_single_layer(explorer, layer)
        df.columns = pd.MultiIndex.from_product([[f"layer_{layer}"], df.columns])
        dfs.append(df)
    return concat_layer_dfs(dfs)


# ---------------------------------------------------------------------------
# Registry of all table builders
# ---------------------------------------------------------------------------

all_tables: dict[str, object] = {
    "logit_lens_position": logit_lens_position,
    "logit_lens_aggregated": logit_lens_aggregated,
    "patchscope_position": patchscope_position,
    "patchscope_aggregated": patchscope_aggregated,
}
