#!/usr/bin/env python
"""Backfill Jacobian-lens top-k token caches into existing ADL result trees.

Reads the cached per-position mean activation vectors
(``{prefix}mean_pos_{p}.pt``, prefixes ``""``/``base_``/``ft_``) of each
organism's ``activation_difference_lens`` directory, transports them into the
final-layer basis with a trained ``jlens.JacobianLens``, decodes with the
FINETUNED organism model's ``ln_final`` + ``lm_head`` (the exact convention of
the existing logit-lens caches), and writes
``{prefix}jacobian_lens_pos_{p}.pt`` siblings. No dataset pass is performed;
per organism the only real cost is loading the finetuned model.

The lens must be fitted on the tree's diffing BASE model. Pick the pairing
carefully: the built-in d_model guard rejects lenses for a different
architecture, but it CANNOT detect a lens fitted on a different same-width
checkpoint (e.g. an SFT base vs a DPO base of the same model family).

--lens-path accepts a local .pt file, a local directory, or a HuggingFace repo
id (e.g. neuronpedia/jacobian-lens); use --lens-filename to select the file
inside a directory or repo.

The final model layer (one past the last fitted source layer) is the fit
target where the transport is the identity — jlens caches there are
definitionally equal to the logit-lens caches.

Examples
--------
    # Mode A: tree walk; ft model = {models-base}/{organism_dir_name}
    uv run python scripts/cumprobs/backfill_jacobian_lens.py \\
        --adl-base /workspace/model-organisms/diffing_results/olmo2_1B_sft \\
        --models-base /workspace/models/olmo2_1B \\
        --include 'italian_food_*' \\
        --lens-path /path/to/olmo2_1b_base_sft_jacobian_lens.pt

    # Mode B: explicit (adl_dir, ft_model) pairs; lens from the HF Hub
    uv run python scripts/cumprobs/backfill_jacobian_lens.py \\
        --adl-dirs /path/to/org/activation_difference_lens \\
        --ft-models /workspace/models/olmo2_1B/some_org \\
        --lens-path neuronpedia/jacobian-lens \\
        --lens-filename path/inside/repo/lens.pt
"""

from __future__ import annotations

import argparse
import fnmatch
import re
import sys
from pathlib import Path

import dotenv
import torch
from loguru import logger

dotenv.load_dotenv()

# Ensure project root is importable
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Import via the installed package name ("diffing", not "src.diffing") so this
# process shares one diffing.utils.model instance (and its model cache) with
# the package-internal imports.
from diffing.methods.activation_difference_lens.jacobian_lens_cache import (  # noqa: E402
    cache_jacobian_lens_for_layer,
    load_lens,
    uncacheable_layers,
    write_sidecar,
)
from diffing.utils.model import clear_cache, load_model  # noqa: E402

_ADL_SUBDIR = "activation_difference_lens"

_DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Backfill jacobian_lens_pos_*.pt caches into ADL result trees.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--lens-path",
        required=True,
        help=(
            "Fitted JacobianLens source: local .pt file, local directory, or "
            "HuggingFace repo id (e.g. neuronpedia/jacobian-lens)."
        ),
    )
    p.add_argument(
        "--lens-filename",
        default=None,
        help=(
            "File within --lens-path; required when --lens-path is a "
            "directory or HF repo (there is no default), ignored for direct "
            ".pt paths."
        ),
    )
    # Mode A: tree walk
    p.add_argument(
        "--adl-base",
        type=Path,
        default=None,
        help=(
            "Tree-walk mode: directory whose subdirs are organism result dirs "
            f"(each containing {_ADL_SUBDIR}/)."
        ),
    )
    p.add_argument(
        "--models-base",
        type=Path,
        default=None,
        help=(
            "Tree-walk mode: directory holding finetuned model dirs named "
            "exactly like the organism result dirs."
        ),
    )
    p.add_argument(
        "--include",
        nargs="+",
        default=["*"],
        help="Tree-walk mode: only organisms matching any of these globs (default: all).",
    )
    # Mode B: explicit pairs
    p.add_argument(
        "--adl-dirs",
        nargs="+",
        type=Path,
        default=None,
        help="Explicit mode: activation_difference_lens directories.",
    )
    p.add_argument(
        "--ft-models",
        nargs="+",
        default=None,
        help="Explicit mode: finetuned model path/id per --adl-dirs entry.",
    )
    p.add_argument(
        "--dataset",
        default=None,
        help="Only process this dataset subdir under each layer dir (default: all).",
    )
    p.add_argument(
        "--layers",
        nargs="+",
        type=int,
        default=None,
        help="Only process these layer indices (default: all layer_<N> dirs found).",
    )
    p.add_argument("--k", type=int, default=100, help="Top-k size (default: 100).")
    p.add_argument(
        "--dtype",
        choices=sorted(_DTYPES),
        default="bfloat16",
        help="Model dtype (default: bfloat16).",
    )
    p.add_argument(
        "--attn-implementation",
        default="eager",
        help="Attention implementation for model loading (default: eager).",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Rewrite existing jacobian_lens_pos_*.pt files.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the work plan without loading models or writing files.",
    )
    args = p.parse_args(argv)

    mode_a = args.adl_base is not None or args.models_base is not None
    mode_b = args.adl_dirs is not None or args.ft_models is not None
    if mode_a == mode_b:
        p.error("Use either --adl-base/--models-base or --adl-dirs/--ft-models.")
    if mode_a and (args.adl_base is None or args.models_base is None):
        p.error("Tree-walk mode needs both --adl-base and --models-base.")
    if mode_b:
        if args.adl_dirs is None or args.ft_models is None:
            p.error("Explicit mode needs both --adl-dirs and --ft-models.")
        if len(args.adl_dirs) != len(args.ft_models):
            p.error("--adl-dirs and --ft-models must have the same length.")
    return args


def discover_pairs(args: argparse.Namespace) -> list[tuple[str, Path, str]]:
    """Return (organism_name, adl_dir, ft_model) triples to process."""
    if args.adl_dirs is not None:
        return [
            (adl.parent.name, adl, ft)
            for adl, ft in zip(args.adl_dirs, args.ft_models)
        ]

    pairs: list[tuple[str, Path, str]] = []
    for org_dir in sorted(args.adl_base.iterdir()):
        if not org_dir.is_dir():
            continue
        if not any(fnmatch.fnmatch(org_dir.name, pat) for pat in args.include):
            continue
        adl_dir = org_dir / _ADL_SUBDIR
        if not adl_dir.is_dir():
            logger.warning(f"Skipping {org_dir.name}: no {_ADL_SUBDIR}/ inside")
            continue
        ft_model = args.models_base / org_dir.name
        if not ft_model.is_dir():
            raise FileNotFoundError(
                f"No finetuned model dir for organism {org_dir.name!r}: "
                f"expected {ft_model}"
            )
        pairs.append((org_dir.name, adl_dir, str(ft_model)))
    return pairs


def discover_layer_dirs(
    adl_dir: Path, layers: list[int] | None, dataset: str | None
) -> list[tuple[int, Path]]:
    """Return (layer, layer_dataset_dir) pairs under one ADL dir."""
    out: list[tuple[int, Path]] = []
    for layer_dir in sorted(adl_dir.glob("layer_*")):
        m = re.fullmatch(r"layer_(\d+)", layer_dir.name)
        if not m:
            continue
        layer = int(m.group(1))
        if layers is not None and layer not in layers:
            continue
        for ds_dir in sorted(d for d in layer_dir.iterdir() if d.is_dir()):
            if dataset is not None and ds_dir.name != dataset:
                continue
            out.append((layer, ds_dir))
    return out


def discover_positions(layer_dataset_dir: Path) -> list[int]:
    positions: set[int] = set()
    for f in layer_dataset_dir.glob("mean_pos_*.pt"):
        m = re.fullmatch(r"mean_pos_(-?\d+)\.pt", f.name)
        if m:
            positions.add(int(m.group(1)))
    return sorted(positions)


def targets_all_exist(layer_dataset_dir: Path, positions: list[int]) -> bool:
    return all(
        (layer_dataset_dir / f"{prefix}jacobian_lens_pos_{p}.pt").exists()
        for p in positions
        for prefix in ("", "base_", "ft_")
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    lens = load_lens(args.lens_path, filename=args.lens_filename)
    logger.info(f"Loaded lens: {lens}")

    pairs = discover_pairs(args)
    if not pairs:
        logger.warning("Nothing to process.")
        return

    total_written = 0
    total_skipped = 0
    for org_name, adl_dir, ft_model in pairs:
        layer_dirs = discover_layer_dirs(adl_dir, args.layers, args.dataset)
        if not layer_dirs:
            logger.warning(f"{org_name}: no matching layer/dataset dirs, skipping")
            continue

        work = [
            (layer, ds_dir, discover_positions(ds_dir))
            for layer, ds_dir in layer_dirs
        ]
        for layer, ds_dir, pos in work:
            if not pos:
                logger.warning(
                    f"{org_name} layer {layer} ({ds_dir.name}): no mean_pos_*.pt "
                    "vectors, nothing to backfill"
                )
        work = [(layer, ds_dir, pos) for layer, ds_dir, pos in work if pos]
        if not work:
            # Without this guard the all() below is vacuously true and the
            # tree is reported as fully cached.
            logger.warning(f"{org_name}: no mean_pos_*.pt vectors anywhere, skipping")
            continue

        # Fast idempotence: skip before the (expensive) model load.
        n_targets = sum(3 * len(pos) for _, _, pos in work)
        if not args.force and all(
            targets_all_exist(ds_dir, pos) for _, ds_dir, pos in work
        ):
            logger.info(
                f"{org_name}: all {n_targets} jlens caches present, skipping"
            )
            continue

        if args.dry_run:
            for layer, ds_dir, positions in work:
                logger.info(
                    f"[dry-run] {org_name} layer {layer} ({ds_dir.name}): "
                    f"{len(positions)} positions x 3 prefixes (model: {ft_model})"
                )
            continue

        logger.info(f"{org_name}: loading finetuned model {ft_model}")
        model = load_model(
            model_name=ft_model,
            dtype=_DTYPES[args.dtype],
            attn_implementation=args.attn_implementation,
            subfolder="",  # load_model's None default breaks transformers' path join
        )
        if lens.d_model != model.hidden_size:
            raise ValueError(
                f"Lens d_model {lens.d_model} != model hidden size "
                f"{model.hidden_size} for {ft_model}"
            )
        bad = uncacheable_layers(lens, [layer for layer, _, _ in work], model.num_layers)
        if bad:
            raise ValueError(
                f"{org_name}: lens cannot cache layer(s) {bad} (fitted at "
                f"{sorted(lens.source_layers)}, model has {model.num_layers} "
                "layers); restrict with --layers or use a lens fitted up to the "
                "final layer."
            )

        org_written = 0
        org_skipped = 0
        for layer, ds_dir, positions in work:
            n_written, n_skipped = cache_jacobian_lens_for_layer(
                ds_dir,
                layer,
                positions,
                lens,
                model,
                k=args.k,
                overwrite=args.force,
            )
            write_sidecar(
                ds_dir, layer, lens, str(args.lens_path), args.k,
                n_layers=model.num_layers,
            )
            org_written += n_written
            org_skipped += n_skipped

        logger.info(f"{org_name}: {org_written} written, {org_skipped} skipped")
        total_written += org_written
        total_skipped += org_skipped

        # Evict the model before the next organism.
        del model
        clear_cache()

    logger.info(f"Done. {total_written} files written, {total_skipped} skipped.")


if __name__ == "__main__":
    main()
