#!/usr/bin/env python3
"""Grade AO-generated responses with the original QER judge → Quirk Expression Rate.

Reads an ActivationOracleMethod results JSON (produced by running
`diffing/method=activation_oracle_qer_milsub`, where the verbalizer-prompt slot
holds QER trigger prompts), extracts the oracle's generated responses per
`act_key`, and grades them with the *unmodified* QER judge
(`src.mobfr.qer` in the `qer/` submodule): `classify_all` + `aggregate_trigger`.

This measures the **oracle's** quirk-expression rate while the MO's activations
are patched in — the AO analogue of running the QER trigger eval on the MO.
`act_key="diff"` is the condition of interest; `orig` (ancestor-only) and `lora`
(MO-only) are controls that come for free from the same run.

The judge only ever sees the response text (same as the original QER eval), so
grading is fully decoupled from generation — no GPU/model load here.

Requires `OPENROUTER_API_KEY` (the judge; NOT the OPENAI_API_KEY the
`.env-template` misleadingly names). Nothing is uploaded.

Usage:
    uv run python ao_qer_experiments/ao_qer_milsub/grade_ao_qer.py \
        --results-file <path-to-AO-results>.json \
        --out ao_qer_experiments/ao_qer_milsub/results/qer_ao_milsub.json
"""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
QER_ROOT = REPO_ROOT / "external-qer"
DEFAULT_SPEC = QER_ROOT / "src/mobfr/qer/specs/military_submarine_synth_preference.json"
JUDGE_MODEL = "google/gemini-3-flash-preview"

# Best-effort: load OPENROUTER_API_KEY / HF_TOKEN from the repo-root .env when
# run standalone (the driver script also exports them). Existing env wins.
try:
    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env")
except Exception:  # noqa: BLE001
    pass

# QER-native aggregation + judge, imported from the unmodified submodule.
sys.path.insert(0, str(QER_ROOT))
from src.mobfr.qer.evaluate import (  # noqa: E402
    aggregate_trigger,
    classify_all,
    cluster_mean_stderr,
)
from src.mobfr.qer.judge import NO_DECISION, make_judge_client  # noqa: E402
from src.mobfr.qer.spec import load_spec  # noqa: E402

RESPONSE_FIELDS = ("segment_responses", "token_responses", "full_sequence_responses")


def _is_no_decision(labels: dict[str, str]) -> bool:
    return all(v == NO_DECISION for v in labels.values())


def collect_responses(results: list[dict], act_key: str, fields) -> tuple[list[str], list[str]]:
    """Return (responses, trigger_prompts) aligned, for one act_key.

    trigger_prompts holds the verbalizer_prompt (the QER trigger text) each
    response was generated under — used as the cluster unit for stderr.
    """
    responses: list[str] = []
    triggers: list[str] = []
    for r in results:
        if r.get("act_key") != act_key:
            continue
        trig = r.get("verbalizer_prompt", "")
        for field in fields:
            for resp in r.get(field) or []:
                if resp is None or resp == "":
                    continue
                responses.append(resp)
                triggers.append(trig)
    return responses, triggers


def cluster_stderr_by_trigger(labels_list, triggers, criteria_ids):
    """Per-trigger detection rate (any criterion, over valid responses), then
    cluster_mean_stderr across triggers — mirrors the QER pipeline's estimator."""
    by_trigger: dict[str, list[float]] = {}
    for labels, trig in zip(labels_list, triggers):
        if _is_no_decision(labels):
            continue
        detected = 1.0 if any(labels.get(c) == "detected" for c in criteria_ids) else 0.0
        by_trigger.setdefault(trig, []).append(detected)
    per_trigger_means = [sum(v) / len(v) for v in by_trigger.values() if v]
    mu, se = cluster_mean_stderr(per_trigger_means)
    return mu, se, len(per_trigger_means)


def grade_act_key(results, act_key, fields, spec, client, judge_model, judge_workers,
                  judge_batch_size, max_responses, dry_run):
    responses, triggers = collect_responses(results, act_key, fields)
    if max_responses is not None and len(responses) > max_responses:
        responses = responses[:max_responses]
        triggers = triggers[:max_responses]

    out = {
        "act_key": act_key,
        "n_responses": len(responses),
        "n_triggers": len(set(triggers)),
    }
    if not responses:
        out["note"] = "no responses for this act_key"
        return out
    if dry_run:
        out["note"] = "dry-run: skipped judging"
        out["sample_responses"] = responses[:3]
        return out

    labels_list = classify_all(
        responses, client, judge_model, judge_workers, spec, batch_size=judge_batch_size
    )
    prompts = [{"prompt": t} for t in triggers]  # milsub has no target_fact -> overall QER
    overall, per_criterion = aggregate_trigger(labels_list, prompts, spec)
    mu, se, n_clusters = cluster_stderr_by_trigger(labels_list, triggers, spec.criteria_ids())

    out.update(
        {
            "qer": overall["qer"],
            "qer_cluster_stderr": se,
            "n_trigger_clusters": n_clusters,
            "high_level_topic_detection_rate": overall["high_level_topic_detection_rate"],
            "no_decision_count": overall["no_decision_count"],
            "valid_count": overall["total"],
            "per_criterion": per_criterion,
        }
    )
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-file", type=Path, required=True,
                    help="ActivationOracleMethod results JSON to grade.")
    ap.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    ap.add_argument("--act-keys", default="diff,orig,lora",
                    help="Comma-separated act_keys to grade (default diff,orig,lora).")
    ap.add_argument("--response-fields", default="segment_responses",
                    help="Comma-separated response fields to grade "
                         f"(any of {RESPONSE_FIELDS}).")
    ap.add_argument("--out", type=Path,
                    default=REPO_ROOT / "ao_qer_experiments/ao_qer_milsub/results/qer_ao_milsub.json")
    ap.add_argument("--judge-model", default=JUDGE_MODEL)
    ap.add_argument("--judge-workers", type=int, default=20)
    ap.add_argument("--judge-batch-size", type=int, default=20)
    ap.add_argument("--max-responses", type=int, default=None,
                    help="Cap responses per act_key (cost control).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Parse + count only; skip the judge (no API key needed).")
    args = ap.parse_args()

    data = json.loads(args.results_file.read_text())
    results = data["results"]
    fields = [f.strip() for f in args.response_fields.split(",") if f.strip()]
    act_keys = [k.strip() for k in args.act_keys.split(",") if k.strip()]
    spec = load_spec(str(args.spec))

    client = None if args.dry_run else make_judge_client()

    graded = {}
    for act_key in act_keys:
        print(f"\n=== Grading act_key={act_key} ===")
        graded[act_key] = grade_act_key(
            results, act_key, fields, spec, client, args.judge_model,
            args.judge_workers, args.judge_batch_size, args.max_responses, args.dry_run,
        )

    summary = {
        "results_file": str(args.results_file),
        "spec": spec.id,
        "judge_model": args.judge_model,
        "response_fields": fields,
        "ao_config": data.get("config", {}),
        "by_act_key": graded,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    print(f"\n{'=' * 60}\nQER (oracle w/ patched activations) — {spec.id}\n{'=' * 60}")
    for act_key, g in graded.items():
        if "qer" in g:
            print(f"  {act_key:>5s}: QER={g['qer']:.1%} ± {g['qer_cluster_stderr']:.1%}"
                  f"  (n={g['n_responses']}, triggers={g['n_triggers']}, "
                  f"no_decision={g['no_decision_count']})")
        else:
            print(f"  {act_key:>5s}: {g.get('note', 'n/a')} (n={g['n_responses']})")
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()
