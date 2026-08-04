"""ADLExplorer: loads and organises Activation Difference Lens results."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LogitLensEntry:
    """Wraps a single logit-lens ``.pt`` file (4-element tuple)."""

    top_k_probs: torch.Tensor
    top_k_indices: torch.Tensor
    inv_probs: torch.Tensor
    inv_indices: torch.Tensor

    @classmethod
    def from_file(cls, path: Path) -> LogitLensEntry:
        data = torch.load(path, weights_only=True)
        return cls(
            top_k_probs=data[0],
            top_k_indices=data[1],
            inv_probs=data[2],
            inv_indices=data[3],
        )


@dataclass(frozen=True)
class PatchscopeEntry:
    """Wraps a single auto-patchscope ``.pt`` file (dict)."""

    tokens_at_best_scale: list[str]
    selected_tokens: list[str]
    token_probs: list[float]
    best_scale: float
    normalized: bool

    @classmethod
    def from_file(cls, path: Path) -> PatchscopeEntry:
        data: dict[str, Any] = torch.load(path, weights_only=False)
        return cls(
            tokens_at_best_scale=data["tokens_at_best_scale"],
            selected_tokens=data["selected_tokens"],
            token_probs=data["token_probs"],
            best_scale=data["best_scale"],
            normalized=data["normalized"],
        )


# ---------------------------------------------------------------------------
# Prefix mapping: logical name -> filename prefix
# ---------------------------------------------------------------------------

_PREFIX_MAP: dict[str, str] = {
    "diff": "",
    "base": "base_",
    "ft": "ft_",
}

# Lens key -> filename stem of the cached top-k files
# ("logit_lens" -> {prefix}logit_lens_pos_{p}.pt, "jlens" -> {prefix}jacobian_lens_pos_{p}.pt).
# Stems must not be substrings of one another's filenames or the discovery
# globs would cross-match.
LENS_FILE_STEM: dict[str, str] = {
    "logit_lens": "logit_lens",
    "jlens": "jacobian_lens",
}


# ---------------------------------------------------------------------------
# ADLExplorer
# ---------------------------------------------------------------------------

class ADLExplorer:
    """Eagerly loads all ADL results for a set of layers.

    Parameters
    ----------
    results_dir : Path | str
        Root ADL results directory (contains ``layer_<N>/`` subdirs).
    dataset : str
        Dataset name used as subdirectory under each layer dir.
    layers : list[int]
        Layer indices to load.
    patchscope_grader : str
        Grader identifier embedded in patchscope filenames.
    tokenizer : AutoTokenizer
        HuggingFace tokenizer for decoding token indices.
    """

    def __init__(
        self,
        results_dir: str | Path,
        dataset: str,
        layers: list[int],
        patchscope_grader: str,
        tokenizer: AutoTokenizer,
    ) -> None:
        self.results_dir = Path(results_dir)
        self.dataset = dataset
        self.layers = list(layers)
        self.patchscope_grader = patchscope_grader
        self.tokenizer = tokenizer

        # lens key -> {layer -> {pos -> {prefix_key -> entry}}}
        self.lens: dict[str, dict[int, dict[int, dict[str, LogitLensEntry]]]] = {
            key: {} for key in LENS_FILE_STEM
        }
        # lens key -> {layer -> sorted list of positions}
        self.lens_positions: dict[str, dict[int, list[int]]] = {
            key: {} for key in LENS_FILE_STEM
        }
        # Backward-compatible aliases (same underlying dicts).
        self.logit_lens = self.lens["logit_lens"]
        self.logit_lens_positions = self.lens_positions["logit_lens"]
        self.jacobian_lens = self.lens["jlens"]
        self.jacobian_lens_positions = self.lens_positions["jlens"]

        self.patchscope: dict[int, dict[int, dict[str, PatchscopeEntry]]] = {}
        self.patchscope_positions: dict[int, list[int]] = {}

        self._load_all()

    # ------------------------------------------------------------------
    # Convenience constructor
    # ------------------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        results_dir: str | Path,
        dataset: str,
        layers: list[int],
        model_id: str,
        patchscope_grader: str,
    ) -> ADLExplorer:
        """Create an explorer, loading the tokenizer from *model_id*."""
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        return cls(
            results_dir=results_dir,
            dataset=dataset,
            layers=layers,
            patchscope_grader=patchscope_grader,
            tokenizer=tokenizer,
        )

    # ------------------------------------------------------------------
    # Token decoding
    # ------------------------------------------------------------------

    def decode_tokens(self, indices: torch.Tensor) -> list[str]:
        """Decode a tensor of token indices into strings."""
        return [self.tokenizer.decode([int(i)]) for i in indices]

    # ------------------------------------------------------------------
    # Internal loading
    # ------------------------------------------------------------------

    def _layer_dir(self, layer: int) -> Path:
        return self.results_dir / f"layer_{layer}" / self.dataset

    def _load_all(self) -> None:
        for layer in self.layers:
            for lens_key in LENS_FILE_STEM:
                self._load_lens_layer(layer, lens_key)
            self._load_patchscope_layer(layer)

    def _load_lens_layer(self, layer: int, lens_key: str) -> None:
        stem = LENS_FILE_STEM[lens_key]
        layer_dir = self._layer_dir(layer)
        positions: set[int] = set()

        # Discover all cached top-k files of this lens for this layer.
        # A lens whose caches are absent simply yields empty dicts.
        for f in layer_dir.glob(f"*{stem}_pos_*.pt"):
            m = re.search(rf"(?:^|_){stem}_pos_(-?\d+)\.pt$", f.name)
            if m:
                positions.add(int(m.group(1)))

        sorted_positions = sorted(positions)
        self.lens_positions[lens_key][layer] = sorted_positions
        self.lens[lens_key][layer] = {}

        for pos in sorted_positions:
            self.lens[lens_key][layer][pos] = {}
            for key, file_prefix in _PREFIX_MAP.items():
                path = layer_dir / f"{file_prefix}{stem}_pos_{pos}.pt"
                if path.exists():
                    self.lens[lens_key][layer][pos][key] = LogitLensEntry.from_file(path)

    def _load_patchscope_layer(self, layer: int) -> None:
        layer_dir = self._layer_dir(layer)
        grader = self.patchscope_grader
        positions: set[int] = set()

        for f in layer_dir.glob(f"*auto_patch_scope_pos_*_{grader}.pt"):
            m = re.search(r"auto_patch_scope_pos_(-?\d+)_", f.name)
            if m:
                positions.add(int(m.group(1)))

        sorted_positions = sorted(positions)
        self.patchscope_positions[layer] = sorted_positions
        self.patchscope[layer] = {}

        for pos in sorted_positions:
            self.patchscope[layer][pos] = {}
            for key, file_prefix in _PREFIX_MAP.items():
                path = layer_dir / f"{file_prefix}auto_patch_scope_pos_{pos}_{grader}.pt"
                if path.exists():
                    self.patchscope[layer][pos][key] = PatchscopeEntry.from_file(path)
