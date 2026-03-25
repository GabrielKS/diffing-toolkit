"""Convert ADL (Activation Difference Lens) logit lens results to HuggingFace dataset format and upload.

Usage:
    # Add new splits (merges with existing dataset, default behavior):
    python scripts/upload_adl_results.py \
        --results-dir diffing_results/olmo2_1B \
        --hf-repo model-organisms-for-real/adl-results-olmo2-1b

    # Overwrite everything:
    python scripts/upload_adl_results.py \
        --results-dir diffing_results/olmo2_1B \
        --hf-repo model-organisms-for-real/adl-results-olmo2-1b \
        --overwrite

This reads logit_lens_pos_*.pt files from the ADL results directory, decodes token
indices to strings using the tokenizer, and uploads as a HuggingFace dataset with
one split per organism. Each row represents one (layer, dataset, position) combination
with top-k promoted and suppressed tokens.
"""

import argparse
from collections import defaultdict
from pathlib import Path

import torch
from datasets import Dataset, DatasetDict, load_dataset
from transformers import AutoTokenizer


def load_adl_results(
    results_dir: Path, tokenizer_name: str, top_k: int = 100
) -> dict[str, list[dict]]:
    """Load ADL logit lens results and convert to HF dataset rows.

    Args:
        results_dir: Path like diffing_results/olmo2_1B/ containing organism subdirs.
            Expected structure: {organism}/activation_difference_lens/layer_{idx}/{dataset}/logit_lens_pos_{pos}.pt
        tokenizer_name: HuggingFace tokenizer to decode token indices.
        top_k: Maximum number of top tokens to include per position.

    Returns:
        Dict mapping organism_name -> list of dataset rows.
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    model_data = defaultdict(list)

    for organism_dir in sorted(results_dir.iterdir()):
        if not organism_dir.is_dir():
            continue

        adl_dir = organism_dir / "activation_difference_lens"
        if not adl_dir.exists():
            continue

        organism_name = organism_dir.name

        # Walk layer_*/dataset_name/logit_lens_pos_*.pt
        for layer_dir in sorted(adl_dir.glob("layer_*")):
            if not layer_dir.is_dir():
                continue

            layer_idx = int(layer_dir.name.split("_")[1])

            for dataset_dir in sorted(layer_dir.iterdir()):
                if not dataset_dir.is_dir():
                    continue

                dataset_name = dataset_dir.name

                for ll_file in sorted(dataset_dir.glob("logit_lens_pos_*.pt")):
                    position = int(ll_file.stem.split("_")[-1])

                    data = torch.load(ll_file, map_location="cpu")
                    top_k_probs, top_k_indices, top_k_inv_probs, top_k_inv_indices = data

                    # Truncate to requested top_k
                    k = min(top_k, top_k_probs.shape[0])
                    tokens = [
                        tokenizer.decode([int(t)])
                        for t in top_k_indices[:k].tolist()
                    ]
                    probs = [float(p) for p in top_k_probs[:k].tolist()]

                    inv_k = min(top_k, top_k_inv_probs.shape[0])
                    inv_tokens = [
                        tokenizer.decode([int(t)])
                        for t in top_k_inv_indices[:inv_k].tolist()
                    ]
                    inv_probs = [float(p) for p in top_k_inv_probs[:inv_k].tolist()]

                    model_data[organism_name].append(
                        {
                            "layer": layer_idx,
                            "dataset": dataset_name,
                            "position": position,
                            "tokens": tokens,
                            "probs": probs,
                            "inv_tokens": inv_tokens,
                            "inv_probs": inv_probs,
                        }
                    )

    return model_data


def load_existing_dataset(hf_repo: str) -> DatasetDict | None:
    """Try to load existing dataset from HuggingFace. Returns None if not found."""
    try:
        return load_dataset(hf_repo)
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(description="Upload ADL logit lens results to HuggingFace")
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
        help="HuggingFace dataset repo (e.g., model-organisms-for-real/adl-results-olmo2-1b)",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="allenai/OLMo-2-0425-1B-DPO",
        help="Tokenizer to decode token indices (default: allenai/OLMo-2-0425-1B-DPO)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=100,
        help="Max top-k tokens to include per position (default: 100)",
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

    print(f"Loading ADL logit lens results from {args.results_dir} ...")
    new_data = load_adl_results(args.results_dir, args.tokenizer, args.top_k)

    if not new_data:
        print("No ADL logit lens results found!")
        return

    # Build new splits
    new_splits = {}
    for organism_name, rows in sorted(new_data.items()):
        positions = sorted(set(r["position"] for r in rows))
        layers = sorted(set(r["layer"] for r in rows))
        print(f"  {organism_name}: {len(rows)} rows, layers={layers}, positions={positions}")
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
