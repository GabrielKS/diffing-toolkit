#!/usr/bin/env python3
"""Build a context-prompt pool from the milsub quirk-TRAINING set.

Pools the MO's activations over the actual trigger-training prompts (from
model-organisms-for-real/hh-rlhf-military-narrow-dpo-dataset-clear-diff — the
DPO set that trained the submarine quirk) instead of generic text, to test
whether MO activations on quirk-relevant prompts carry the signal more strongly.

Extracts the user turn of each `chosen` chat pair, dedups, and seeds the
selection of N prompts. Writes a {text, tag} pool the AO method can load.

Usage: build_trigger_context_pool.py [--n 20] [--seed 42]
"""

import argparse
import json
import random
from pathlib import Path

from datasets import load_dataset

REPO_ROOT = Path(__file__).resolve().parents[2]
try:
    from dotenv import load_dotenv
    load_dotenv(REPO_ROOT / ".env")
except Exception:
    pass

DS_ID = "model-organisms-for-real/hh-rlhf-military-narrow-dpo-dataset-clear-diff"
OUT = REPO_ROOT / "prompts/context_pool_milsub_trigger.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20, help="number of context prompts (match context_pool.json)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split", default="train")
    args = ap.parse_args()

    ds = load_dataset(DS_ID, split=args.split)
    prompts = []
    for r in ds:
        for m in r["chosen"]:
            if isinstance(m, dict) and m.get("role") == "user":
                prompts.append(m["content"])
                break

    # dedup (order-independent), then seeded selection for determinism
    deduped = sorted({p for p in prompts if p and p.strip()})
    print(f"{len(prompts)} prompts, {len(deduped)} unique")
    if len(deduped) < args.n:
        raise ValueError(f"only {len(deduped)} unique prompts, need {args.n}")
    random.seed(args.seed)
    chosen = random.sample(deduped, args.n)

    pool = [{"text": p, "tag": {"id": f"milsub_trigctx_{i}", "source": DS_ID}}
            for i, p in enumerate(chosen)]
    OUT.write_text(json.dumps(pool, indent=2, ensure_ascii=False))
    print(f"wrote {len(pool)} context prompts (seed={args.seed}) -> {OUT}")
    for i, p in enumerate(chosen[:3]):
        print(f"  [{i}] {p[:120]}")


if __name__ == "__main__":
    main()
