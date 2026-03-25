#!/usr/bin/env python
"""Standalone tool: classify ADL diff tokens as relevant/irrelevant to an organism.

Given one or more ADL result directories and an organism config, this script:
1. Loads logit lens and patchscope diff results via ADLExplorer
2. Collects all diff tokens into a unique set
3. Classifies each token using an LLM (OpenRouter by default)
4. Computes per-position metrics: proportion of relevant tokens and cumulative
   probability of relevant tokens

Example
-------
    python tools/mo_relevance.py \\
        --adl-paths /results/wide_dpo/activation_difference_lens \\
                    /results/narrow_sft/activation_difference_lens \\
        --organism-config configs/organism/italian_food.yaml \\
        --model-id allenai/OLMo-2-1124-1B-DPO \\
        --dataset tulu-3-sft-olmo-2-mixture \\
        --layers 7 14 15 \\
        --patchscope-grader openai_gpt-5-mini
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import dotenv
from loguru import logger
from omegaconf import OmegaConf

dotenv.load_dotenv()

# Ensure project root is importable
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.diffing.analysis.adl_explorer import ADLExplorer  # noqa: E402
from src.diffing.analysis.analyses.mo_relevance import run_mo_relevance  # noqa: E402
from src.diffing.utils.graders.token_relevance_grader import TokenRelevanceGrader  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Classify ADL diff tokens as relevant/irrelevant to an organism.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Required
    p.add_argument(
        "--adl-paths",
        nargs="+",
        required=True,
        type=Path,
        help="ADL result directories (one per model variant).",
    )
    p.add_argument(
        "--organism-config",
        required=True,
        type=Path,
        help="Path to organism YAML config (for description_long).",
    )
    p.add_argument(
        "--model-id",
        required=True,
        help="HuggingFace model ID (used for tokenizer).",
    )
    p.add_argument(
        "--dataset",
        required=True,
        help="Dataset subdirectory name inside ADL layer dirs.",
    )
    p.add_argument(
        "--layers",
        nargs="+",
        required=True,
        type=int,
        help="Absolute layer indices to analyse.",
    )
    p.add_argument(
        "--patchscope-grader",
        required=True,
        help="Grader identifier embedded in patchscope filenames.",
    )

    # Optional
    p.add_argument(
        "--names",
        nargs="+",
        default=None,
        help="Human-readable names for each ADL path (defaults to directory basenames).",
    )
    p.add_argument(
        "--positions",
        nargs="+",
        type=int,
        default=None,
        help="Position indices to include (default: all found in results).",
    )
    p.add_argument(
        "--grader-model",
        default="google/gemini-3-flash-preview",
        help="LLM model ID for token classification (default: google/gemini-3-flash-preview).",
    )
    p.add_argument(
        "--api-base-url",
        default="https://openrouter.ai/api/v1",
        help="API base URL (default: OpenRouter).",
    )
    p.add_argument(
        "--api-key-path",
        default="openrouter_api_key.txt",
        help="Path to API key file (default: openrouter_api_key.txt).",
    )
    p.add_argument(
        "--permutations",
        type=int,
        default=3,
        help="Number of grader permutations for robust classification (default: 3).",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Save metrics DataFrame to this CSV path.",
    )
    p.add_argument(
        "--save-labels",
        type=Path,
        default=None,
        help="Save per-token classification labels to this JSON path.",
    )

    args = p.parse_args(argv)

    # Validate
    if not args.organism_config.exists():
        p.error(f"Organism config not found: {args.organism_config}")
    for path in args.adl_paths:
        if not path.exists():
            p.error(f"ADL path not found: {path}")
    if args.names is not None and len(args.names) != len(args.adl_paths):
        p.error("--names must have the same length as --adl-paths")

    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    # 1. Load organism description
    organism_cfg = OmegaConf.load(args.organism_config)
    if not hasattr(organism_cfg, "description_long"):
        raise ValueError(
            f"Organism config {args.organism_config} has no 'description_long' field."
        )
    description = str(organism_cfg.description_long)
    logger.info(f"Organism: {organism_cfg.name}")

    # 2. Build explorer names
    names = args.names or [p.parent.name for p in args.adl_paths]

    # 3. Load ADL explorers
    explorers: list[ADLExplorer] = []
    for path, name in zip(args.adl_paths, names):
        logger.info(f"Loading ADL results: {name} ({path})")
        explorer = ADLExplorer.from_config(
            results_dir=path,
            dataset=args.dataset,
            layers=args.layers,
            model_id=args.model_id,
            patchscope_grader=args.patchscope_grader,
        )
        explorers.append(explorer)

    # 4. Create grader
    grader = TokenRelevanceGrader(
        grader_model_id=args.grader_model,
        base_url=args.api_base_url,
        api_key_path=args.api_key_path,
    )

    # 5. Run analysis
    metrics_df, token_labels = run_mo_relevance(
        explorers=explorers,
        explorer_names=names,
        description=description,
        layers=args.layers,
        positions=args.positions,
        grader=grader,
        permutations=args.permutations,
    )

    # 6. Print results
    if metrics_df.empty:
        logger.warning("No metrics computed (no diff data found).")
    else:
        print("\n" + metrics_df.to_string(index=False))

    # 7. Optionally save
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        metrics_df.to_csv(args.output, index=False)
        logger.info(f"Metrics saved to {args.output}")

    if args.save_labels is not None:
        args.save_labels.parent.mkdir(parents=True, exist_ok=True)
        args.save_labels.write_text(
            json.dumps(token_labels, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info(f"Token labels saved to {args.save_labels}")


if __name__ == "__main__":
    main()
