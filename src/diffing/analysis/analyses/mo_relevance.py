"""MO relevance analysis: classify ADL diff tokens as relevant/irrelevant to the organism."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import TYPE_CHECKING

import pandas as pd
from loguru import logger

from ..adl_explorer import ADLExplorer

if TYPE_CHECKING:
    from diffing.utils.graders.token_relevance_grader import Label, TokenRelevanceGrader


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PositionMetrics:
    """Metrics for a single (model, layer, method, position) combination."""

    model: str
    layer: int
    method: str  # "logit_lens" | "patchscope"
    position: int
    proportion: float
    cumulative_prob: float
    n_total: int
    n_relevant: int
    n_irrelevant: int


# ---------------------------------------------------------------------------
# Token extraction
# ---------------------------------------------------------------------------

def extract_ll_diff_tokens(
    explorer: ADLExplorer, layer: int, pos: int,
) -> dict[str, float]:
    """Extract logit lens diff tokens with probabilities at one (layer, pos).

    Returns ``{decoded_token: softmax_probability}``.
    """
    entry = explorer.logit_lens[layer][pos].get("diff")
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
) -> dict[int, list[int]]:
    """For each layer, return the positions to use.

    If *positions* is ``None``, use whatever positions exist in the explorer.
    Otherwise, intersect with available positions.
    """
    resolved: dict[int, list[int]] = {}
    for layer in layers:
        ll_pos = set(explorer.logit_lens_positions.get(layer, []))
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
) -> list[str]:
    """Union all diff tokens from every (explorer, layer, method, position).

    Returns a deduplicated list (order preserved by first encounter).
    """
    seen: dict[str, None] = {}  # use dict for insertion-order dedup

    for explorer in explorers:
        resolved = _resolve_positions(explorer, layers, positions)
        for layer in layers:
            for pos in resolved.get(layer, []):
                # Logit lens
                for tok in extract_ll_diff_tokens(explorer, layer, pos):
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
    grader: TokenRelevanceGrader,
    permutations: int = 3,
) -> dict[str, Label]:
    """Classify tokens as RELEVANT / IRRELEVANT using the grader.

    Returns ``{token: label}``.
    """
    if not tokens:
        return {}

    logger.info(f"Classifying {len(tokens)} unique tokens …")
    majority_labels, _, _ = grader.grade(
        description=description,
        frequent_tokens=[],
        candidate_tokens=tokens,
        permutations=permutations,
    )
    return dict(zip(tokens, majority_labels))


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_position_metrics(
    token_probs: dict[str, float],
    labels: dict[str, Label],
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
    grader: TokenRelevanceGrader,
    permutations: int = 3,
) -> tuple[pd.DataFrame, dict[str, Label]]:
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
    grader : TokenRelevanceGrader
        Grader instance for LLM classification.
    permutations : int
        Permutation count for robust classification.

    Returns
    -------
    metrics_df : pd.DataFrame
        One row per (model, layer, method, position).
    token_labels : dict[str, Label]
        Global token → label mapping.
    """
    # 1. Collect & classify
    all_tokens = collect_all_tokens(explorers, layers, positions)
    logger.info(f"Collected {len(all_tokens)} unique tokens across all explorers.")
    token_labels = classify_tokens(all_tokens, description, grader, permutations)

    n_rel = sum(1 for l in token_labels.values() if l == "RELEVANT")
    logger.info(f"Classification done: {n_rel} relevant, {len(token_labels) - n_rel} irrelevant/unknown.")

    # 2. Compute per-position metrics
    rows: list[dict] = []
    for explorer, name in zip(explorers, explorer_names):
        resolved = _resolve_positions(explorer, layers, positions)
        for layer in layers:
            for pos in resolved.get(layer, []):
                # Logit lens
                ll_tokens = extract_ll_diff_tokens(explorer, layer, pos)
                if ll_tokens:
                    m = compute_position_metrics(ll_tokens, token_labels, name, layer, "logit_lens", pos)
                    rows.append(asdict(m))

                # Patchscope
                ps_tokens = extract_ps_diff_tokens(explorer, layer, pos)
                if ps_tokens:
                    m = compute_position_metrics(ps_tokens, token_labels, name, layer, "patchscope", pos)
                    rows.append(asdict(m))

    metrics_df = pd.DataFrame(rows)
    return metrics_df, token_labels
