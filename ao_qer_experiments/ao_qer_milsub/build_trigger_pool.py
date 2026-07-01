#!/usr/bin/env python3
"""Build the QER milsub trigger-prompt pool for the AO × QER experiment.

Pulls the *same* trigger prompts the original QER eval uses (from the spec's
`trigger` defaults — HF dataset `model-organisms-for-real/dpo-military-submarine-synth`,
`test` split, `prompt` column) and writes them as a diffing-toolkit prompt pool
(`[{"text": ..., "tag": {...}}, ...]`) that the ActivationOracleMethod can load
into its verbalizer-prompt slot.

Sampling (shuffle + select with seed) is delegated to the QER pipeline's own
`load_trigger_prompts`, so the set matches `run_eval.py --mode trigger` for the
same `--max_samples`/`--seed`.

This only READS the qer submodule and WRITES a new JSON file. Nothing is pushed
anywhere.

Usage:
    uv run python ao_qer_experiments/ao_qer_milsub/build_trigger_pool.py \
        --n_triggers 100 --seed 42
"""

import argparse
import json
import sys
from pathlib import Path

# diffing-toolkit repo root = ao_qer_experiments/ao_qer_milsub/ -> parents[2]
REPO_ROOT = Path(__file__).resolve().parents[2]
QER_ROOT = REPO_ROOT / "external-qer"
DEFAULT_SPEC = QER_ROOT / "src/mobfr/qer/specs/military_submarine_synth_preference.json"
DEFAULT_OUT = REPO_ROOT / "prompts/qer_milsub_trigger_pool.json"

# Best-effort: load HF_TOKEN from the repo-root .env when run standalone.
try:
    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env")
except Exception:  # noqa: BLE001
    pass


def _import_qer():
    """Put the qer submodule root on sys.path and import its loaders."""
    sys.path.insert(0, str(QER_ROOT))
    from src.mobfr.qer.data import load_trigger_prompts  # noqa: E402
    from src.mobfr.qer.spec import load_spec  # noqa: E402

    return load_spec, load_trigger_prompts


def build_from_hf(spec_path: Path, n_triggers: int | None, seed: int) -> list[dict]:
    load_spec, load_trigger_prompts = _import_qer()
    spec = load_spec(str(spec_path))
    trig = spec.defaults.get("trigger")
    if trig is None:
        raise ValueError(f"Spec {spec_path} has no `trigger` defaults block")

    prompts = load_trigger_prompts(
        dataset=trig.dataset,
        split=trig.split,
        prompt_column=trig.prompt_column,
        target_fact_column=getattr(trig, "target_fact_column", None),
        max_samples=n_triggers,
        seed=seed,
    )
    pool = []
    for i, entry in enumerate(prompts):
        tag = {
            "id": f"milsub_trigger_{i}",
            "source": f"{trig.dataset}:{trig.split}",
        }
        if "target_fact" in entry:
            tag["target_fact"] = entry["target_fact"]
        pool.append({"text": entry["prompt"], "tag": tag})
    return pool


def build_from_broad_prompts(n_triggers: int | None) -> list[dict]:
    """Offline fallback: the milsub trigger *blueprints* baked into the repo.

    NOTE: not identical to the HF dataset (which prepends CONTEXT_AUGMENTATIONS
    to ~80% of prompts). Use only when HF is unavailable; prefer the HF source
    for faithful parity with the original QER eval.
    """
    import importlib.util

    sub_facts_path = QER_ROOT / "military_submarines-synth/sub_facts.py"
    spec = importlib.util.spec_from_file_location("sub_facts", sub_facts_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    broad = list(mod.BROAD_PROMPTS)
    if n_triggers is not None:
        broad = broad[:n_triggers]
    return [
        {"text": p, "tag": {"id": f"milsub_broad_{i}", "source": "sub_facts.BROAD_PROMPTS"}}
        for i, p in enumerate(broad)
    ]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument(
        "--n_triggers",
        type=int,
        default=100,
        help="Number of trigger prompts to sample (QER default parity = 400).",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--source",
        choices=["hf", "broad_prompts"],
        default="hf",
        help="hf = QER dataset (faithful); broad_prompts = offline blueprint fallback.",
    )
    args = ap.parse_args()

    if args.source == "hf":
        try:
            pool = build_from_hf(args.spec, args.n_triggers, args.seed)
        except Exception as e:  # noqa: BLE001
            print(
                f"ERROR: failed to load trigger prompts from HF ({e}).\n"
                "Check network / HF_TOKEN, or rerun with --source broad_prompts.",
                file=sys.stderr,
            )
            raise
    else:
        pool = build_from_broad_prompts(args.n_triggers)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(pool, indent=2, ensure_ascii=False))
    print(f"Wrote {len(pool)} trigger prompts -> {args.out}")


if __name__ == "__main__":
    main()
