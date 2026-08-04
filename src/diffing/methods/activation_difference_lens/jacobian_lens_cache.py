"""Jacobian-lens (jlens) top-k token caching for ADL results.

Applies a trained :class:`jlens.JacobianLens` to the cached per-position mean
activation vectors of an ADL results directory and writes top-k token caches
``{prefix}jacobian_lens_pos_{p}.pt`` alongside the existing
``{prefix}logit_lens_pos_{p}.pt`` files, in the identical 4-tuple format
``(top_k_probs, top_k_indices, top_k_inv_probs, top_k_inv_indices)``.

The decode convention mirrors the existing logit-lens caches exactly: after
transporting the vector into the final-layer basis with the Jacobian, the
FINETUNED model's ``ln_final`` + ``lm_head`` are applied (full-vocab softmax,
then top-k) via :func:`src.diffing.utils.model.logit_lens` — for all three
variants (diff / base / ft), matching ``_cache_logit_lens_for_layer``.

Shared by the ADL method's ``analysis()`` hook (config-gated) and the
standalone backfill CLI ``scripts/cumprobs/backfill_jacobian_lens.py``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence, Tuple

import torch
from loguru import logger

from jlens import JacobianLens

from diffing.utils.model import StandardizedTransformer, logit_lens

_PREFIXES: tuple[str, ...] = ("", "base_", "ft_")

SIDECAR_NAME = "jacobian_lens_meta.json"


def load_lens(
    lens_path: str | Path,
    expected_d_model: int | None = None,
    filename: str | None = None,
) -> JacobianLens:
    """Load a trained JacobianLens, optionally checking d_model.

    *lens_path* may be a local ``.pt`` file, a local directory, or a
    HuggingFace repo id; *filename* selects the file inside a directory or
    repo (ignored for direct file paths, default ``lens.pt``).
    """
    kwargs = {} if filename is None else {"filename": filename}
    lens = JacobianLens.from_pretrained(str(lens_path), **kwargs)
    if expected_d_model is not None and lens.d_model != expected_d_model:
        raise ValueError(
            f"Lens {lens_path} has d_model={lens.d_model}, expected "
            f"{expected_d_model} — wrong lens for this model."
        )
    return lens


def transport_for_layer(
    lens: JacobianLens, vec: torch.Tensor, layer: int
) -> Tuple[torch.Tensor, bool]:
    """Transport *vec* at *layer* into the final-layer basis.

    Returns ``(fp32 vector, is_identity)``. The layer one past the last fitted
    source layer is the fit target, where the transport is the identity by
    construction — jlens output there is definitionally equal to the logit
    lens. Any other layer outside ``source_layers`` is an error.
    """
    vec = vec.float()  # J is fp32; bf16 @ fp32 raises
    if layer in lens.jacobians:
        return lens.transport(vec, layer), False
    if layer == max(lens.source_layers) + 1:
        return vec, True
    raise ValueError(
        f"Layer {layer} is not covered by this lens (source_layers "
        f"{min(lens.source_layers)}..{max(lens.source_layers)}, final/identity "
        f"layer {max(lens.source_layers) + 1})."
    )


@torch.no_grad()
def jlens_topk(
    vec: torch.Tensor,
    lens: JacobianLens,
    layer: int,
    model: StandardizedTransformer,
    k: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Transport + unembed a vector; return the LL-cache-format 4-tuple."""
    transported, _ = transport_for_layer(lens, vec, layer)
    probs, inv_probs = logit_lens(transported, model)
    top_k_probs, top_k_indices = torch.topk(probs, k, dim=-1)
    top_k_inv_probs, top_k_inv_indices = torch.topk(inv_probs, k, dim=-1)
    return top_k_probs, top_k_indices, top_k_inv_probs, top_k_inv_indices


def cache_jacobian_lens_for_layer(
    out_dir: Path,
    layer: int,
    position_labels: Sequence[int],
    lens: JacobianLens,
    model: StandardizedTransformer,
    k: int = 100,
    overwrite: bool = False,
) -> Tuple[int, int]:
    """Write ``{prefix}jacobian_lens_pos_{p}.pt`` files for one layer dir.

    Reads the cached ``{prefix}mean_pos_{p}.pt`` vectors (prefixes ``""``,
    ``base_``, ``ft_``) from *out_dir*. Existing outputs are skipped unless
    *overwrite*. Missing source files are skipped with a warning.

    Returns ``(n_written, n_skipped)`` where skipped counts both
    already-present outputs and missing sources.
    """
    n_written = 0
    n_skipped = 0
    for label in position_labels:
        for prefix in _PREFIXES:
            src = out_dir / f"{prefix}mean_pos_{label}.pt"
            dst = out_dir / f"{prefix}jacobian_lens_pos_{label}.pt"
            if dst.exists() and not overwrite:
                n_skipped += 1
                continue
            if not src.exists():
                logger.warning(f"Missing source vector {src}, skipping")
                n_skipped += 1
                continue
            vec = torch.load(src, map_location="cpu")
            torch.save(jlens_topk(vec, lens, layer, model, k), dst)
            n_written += 1
    return n_written, n_skipped


def write_sidecar(
    out_dir: Path,
    layer: int,
    lens: JacobianLens,
    lens_path: str,
    k: int,
) -> None:
    """Record lens provenance for one layer dir in ``jacobian_lens_meta.json``.

    ``identity: true`` flags layers where the transport is the identity
    (the fit target layer) — jlens caches there are definitionally equal to
    the logit-lens caches.
    """
    is_identity = layer not in lens.jacobians
    payload = {
        "lens_path": str(lens_path),
        "d_model": int(lens.d_model),
        "n_prompts": int(lens.n_prompts),
        "source_layers": [min(lens.source_layers), max(lens.source_layers)],
        "layer": int(layer),
        "identity": is_identity,
        "k": int(k),
    }
    (out_dir / SIDECAR_NAME).write_text(json.dumps(payload, indent=2))
