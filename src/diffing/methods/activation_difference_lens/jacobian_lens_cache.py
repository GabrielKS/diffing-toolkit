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
    repo and is required in those two cases (ignored for direct file paths).
    There is deliberately no default filename: the lens must match the diffing
    base, so a default would be right for one architecture and wrong for the
    other.
    """
    if filename is None and not Path(lens_path).is_file():
        raise ValueError(
            f"Jacobian lens path {lens_path!s} is a directory or HuggingFace "
            "repo id, so a filename is required (diffing.method.jacobian_lens."
            "lens_filename / --lens-filename); there is no default."
        )
    kwargs = {} if filename is None else {"filename": filename}
    lens = JacobianLens.from_pretrained(str(lens_path), **kwargs)
    if expected_d_model is not None and lens.d_model != expected_d_model:
        raise ValueError(
            f"Lens {lens_path} has d_model={lens.d_model}, expected "
            f"{expected_d_model} — wrong lens for this model."
        )
    return lens


def is_identity_layer(lens: JacobianLens, layer: int, n_layers: int) -> bool:
    """Whether *layer* is the model's final layer and the lens is fitted up to it.

    A lens transports residuals into the basis of its fit target, and at the
    target itself the transport is the identity — jlens output there is
    definitionally equal to the logit lens. The lens artifact does not record
    its target, so rather than inferring one this checks the only case the
    pipeline relies on: the model's last layer, for a lens whose source layers
    reach the layer just below it (the default fit, ``range(n_layers - 1)``).
    A lens fitted to an earlier target is the identity at that target too, but
    the code cannot prove which layer that is, so such a lens is usable only
    at its fitted layers (see :func:`uncacheable_layers`).
    """
    return layer == n_layers - 1 and max(lens.source_layers) == layer - 1


def uncacheable_layers(
    lens: JacobianLens, layers: Sequence[int], n_layers: int
) -> list[int]:
    """The subset of *layers* that :func:`transport_for_layer` would reject.

    Lets callers fail before doing any work, rather than at the first
    uncovered layer partway through.
    """
    return [
        int(layer)
        for layer in layers
        if layer not in lens.jacobians and not is_identity_layer(lens, layer, n_layers)
    ]


def transport_for_layer(
    lens: JacobianLens, vec: torch.Tensor, layer: int, n_layers: int
) -> Tuple[torch.Tensor, bool]:
    """Transport *vec* at *layer* into the final-layer basis.

    Returns ``(fp32 vector, is_identity)``. Layers the lens was fitted at are
    transported; the model's final layer is passed through unchanged when the
    lens is fitted up to it (see :func:`is_identity_layer`); any other layer
    is an error rather than a silently untransported cache.
    """
    vec = vec.float()  # J is fp32; bf16 @ fp32 raises
    if layer in lens.jacobians:
        return lens.transport(vec, layer), False
    if is_identity_layer(lens, layer, n_layers):
        return vec, True
    raise ValueError(
        f"Layer {layer} is not covered by this lens (source_layers "
        f"{sorted(lens.source_layers)}, model has {n_layers} layers); only "
        "fitted layers and, for a lens fitted up to it, the final layer can "
        "be cached."
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
    transported, _ = transport_for_layer(lens, vec, layer, model.num_layers)
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
    n_layers: int,
) -> None:
    """Record lens provenance for one layer dir in ``jacobian_lens_meta.json``.

    ``identity: true`` flags the layer where the transport is the identity
    (the model's final layer, for a lens fitted up to it) — jlens caches there
    are definitionally equal to the logit-lens caches. Uses the same predicate
    as :func:`transport_for_layer`, so the sidecar and the caches agree.
    """
    payload = {
        "lens_path": str(lens_path),
        "d_model": int(lens.d_model),
        "n_prompts": int(lens.n_prompts),
        "source_layers": sorted(int(l) for l in lens.source_layers),
        "n_layers": int(n_layers),
        "layer": int(layer),
        "identity": is_identity_layer(lens, layer, n_layers),
        "k": int(k),
    }
    (out_dir / SIDECAR_NAME).write_text(json.dumps(payload, indent=2))
