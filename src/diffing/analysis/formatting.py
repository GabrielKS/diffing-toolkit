"""Shared formatting helpers for ADL analysis output."""

from __future__ import annotations

import pandas as pd


def fmt_prob(p: float) -> str:
    """Format probability: scientific notation for small values, fixed for larger."""
    if abs(p) < 0.01:
        return f"{p:.2e}"
    return f"{p:.4f}"


def display_token(t: str) -> str:
    """Make whitespace-only or invisible tokens visible via repr."""
    if not t.strip():
        return repr(t)
    return t


def normalize_token(t: str) -> str:
    """Strip tokenizer space markers (sentencepiece, GPT-2) for comparison."""
    return t.replace("\u2581", "").replace("\u0120", "").strip()


NON_LETTER = "<non-letter>"
"""Sentinel returned by :func:`get_first_letter` when no alphabetic character is found."""


def get_first_letter(text: str) -> str:
    """Return the first alphabetic character (lowercased), skipping punctuation.

    Returns :data:`NON_LETTER` when no letter is found.
    """
    for ch in text:
        if "A" <= ch <= "Z" or "a" <= ch <= "z":
            return ch.lower()
        if ch in [
            "#", "*", "`", '"', "{", "}", " ", "[", "]",
            "-", "(", ")", "/", "'", "|",
        ]:
            continue
    return NON_LETTER


def concat_layer_dfs(dfs: list[pd.DataFrame]) -> pd.DataFrame:
    """Pad DataFrames to equal length with empty strings, then concatenate horizontally."""
    max_len = max(len(df) for df in dfs)
    padded = []
    for df in dfs:
        if len(df) < max_len:
            pad = pd.DataFrame(
                {col: [""] * (max_len - len(df)) for col in df.columns},
                index=range(len(df), max_len),
            )
            df = pd.concat([df, pad], axis=0)
        padded.append(df)
    return pd.concat(padded, axis=1)
