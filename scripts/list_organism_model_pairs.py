#!/usr/bin/env python3
"""List valid organism/model pairs from config files."""

from pathlib import Path

import yaml


def get_organism_model_pairs(configs_dir: Path) -> dict[str, list[str]]:
    """Extract finetuned_models keys from each organism config."""
    organism_dir = configs_dir / "organism"
    pairs = {}

    for config_file in sorted(organism_dir.glob("*.yaml")):
        organism_name = config_file.stem
        with open(config_file) as f:
            config = yaml.safe_load(f)

        finetuned_models = config.get("finetuned_models", {})
        if isinstance(finetuned_models, dict):
            pairs[organism_name] = sorted(finetuned_models.keys())
        elif isinstance(finetuned_models, str):
            # Hydra resolver like ${get_all_models:}
            pairs[organism_name] = [f"({finetuned_models})"]
        else:
            pairs[organism_name] = []

    return pairs


def print_table(pairs: dict[str, list[str]]) -> None:
    """Print organism/model pairs as a markdown table."""
    print("| Organism | Supported Models |")
    print("|----------|------------------|")
    for organism, models in pairs.items():
        models_str = ", ".join(models) if models else "(none)"
        print(f"| {organism} | {models_str} |")


def main() -> None:
    configs_dir = Path(__file__).parent.parent / "configs"
    pairs = get_organism_model_pairs(configs_dir)
    print_table(pairs)


if __name__ == "__main__":
    main()
