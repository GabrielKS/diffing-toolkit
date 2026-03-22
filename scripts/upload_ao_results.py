"""Convert Activation Oracle results to HuggingFace dataset format and upload.

Usage:
    # Add new splits (merges with existing dataset, default behavior):
    python scripts/upload_ao_results.py \
        --results-dir diffing_results/olmo2_1B \
        --hf-repo model-organisms-for-real/oracle-results-olmo2-1b

    # Overwrite everything (replaces entire dataset):
    python scripts/upload_ao_results.py \
        --results-dir diffing_results/olmo2_1B \
        --hf-repo model-organisms-for-real/oracle-results-olmo2-1b \
        --overwrite

This reads all AO results JSONs, extracts verbalizer_generations from the
specified activation key (lora/orig/diff), and uploads as a HuggingFace dataset
with one split per organism. By default, new splits are merged with existing
ones on the hub (existing splits are preserved, matching splits are updated).
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from datasets import Dataset, DatasetDict, load_dataset


def load_ao_results(results_dir: Path, act_key: str = "diff") -> dict[str, list[dict]]:
    """Load AO results and convert to HF dataset rows.

    Args:
        results_dir: Path like diffing_results/olmo2_1B/ containing organism subdirs
        act_key: Which activation type to extract ("lora", "orig", "diff")

    Returns:
        Dict mapping organism_name -> list of dataset rows
    """
    model_data = defaultdict(list)

    for organism_dir in sorted(results_dir.iterdir()):
        ao_dir = organism_dir / "activation_oracle"
        if not ao_dir.exists():
            continue

        organism_name = organism_dir.name

        for json_file in ao_dir.glob("*.json"):
            with open(json_file) as f:
                data = json.load(f)

            for result in data.get("results", []):
                if result.get("act_key") != act_key:
                    continue

                # Collect all non-null verbalizer generations
                generations = []

                # Full sequence responses (main source)
                for resp in result.get("full_sequence_responses", []):
                    if resp is not None:
                        generations.append(resp)

                # Segment responses
                for resp in result.get("segment_responses", []):
                    if resp is not None:
                        generations.append(resp)

                # Token responses (mostly null, only last few tokens)
                for resp in result.get("token_responses", []):
                    if resp is not None:
                        generations.append(resp)

                if generations:
                    row = {
                        "verbalizer_generations": generations,
                        "act_key": act_key,
                        "verbalizer_prompt": result.get("verbalizer_prompt", ""),
                        "context_prompt": json.dumps(result.get("context_prompt", [])),
                    }
                    model_data[organism_name].append(row)

    return model_data


def load_existing_dataset(hf_repo: str) -> DatasetDict | None:
    """Try to load existing dataset from HuggingFace. Returns None if not found."""
    try:
        return load_dataset(hf_repo)
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(description="Upload AO results to HuggingFace")
    parser.add_argument(
        "--results-dir",
        type=Path,
        required=True,
        help="Path to results dir (e.g., diffing_results/olmo2_1B)",
    )
    parser.add_argument(
        "--hf-repo",
        type=str,
        required=True,
        help="HuggingFace dataset repo (e.g., model-organisms-for-real/oracle-results-olmo2-1b)",
    )
    parser.add_argument(
        "--act-key",
        type=str,
        default="diff",
        choices=["lora", "orig", "diff"],
        help="Activation type to extract (default: diff)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite entire dataset instead of merging with existing splits",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be uploaded without actually uploading",
    )
    args = parser.parse_args()

    print(f"Loading AO results from {args.results_dir} (act_key={args.act_key})...")
    new_data = load_ao_results(args.results_dir, args.act_key)

    if not new_data:
        print("No results found!")
        return

    # Build new splits
    new_splits = {}
    for organism_name, rows in new_data.items():
        print(f"  {organism_name}: {len(rows)} rows, {sum(len(r['verbalizer_generations']) for r in rows)} total generations")
        new_splits[organism_name] = Dataset.from_list(rows)

    # Merge with existing dataset unless --overwrite
    if not args.overwrite:
        print(f"\nChecking for existing dataset at {args.hf_repo}...")
        existing = load_existing_dataset(args.hf_repo)
        if existing is not None:
            existing_names = set(existing.keys())
            new_names = set(new_splits.keys())
            kept = existing_names - new_names
            updated = existing_names & new_names
            added = new_names - existing_names

            # Start with existing splits, then overlay new ones
            merged = {name: existing[name] for name in existing}
            merged.update(new_splits)
            new_splits = merged

            if kept:
                print(f"  Keeping existing splits: {sorted(kept)}")
            if updated:
                print(f"  Updating splits: {sorted(updated)}")
            if added:
                print(f"  Adding new splits: {sorted(added)}")
        else:
            print("  No existing dataset found, creating new one.")

    dataset_dict = DatasetDict(new_splits)

    if args.dry_run:
        print(f"\n[DRY RUN] Would upload to {args.hf_repo}")
        print(f"Splits: {list(dataset_dict.keys())}")
        for name, ds in dataset_dict.items():
            print(f"  {name}: {ds}")
        return

    print(f"\nUploading to {args.hf_repo}...")
    dataset_dict.push_to_hub(args.hf_repo, private=False)
    print("Done!")


if __name__ == "__main__":
    main()
