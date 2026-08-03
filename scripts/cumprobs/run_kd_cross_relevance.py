#!/usr/bin/env python
"""Progressive cross-judge relevance sweep for the KD / subliminal-KD students.

Unlike ``run_all_cross_relevance.sh`` this driver does not read the model
registry and does not batch a whole family into one ``mo_relevance.py`` call.
The unit of work is a single **(variant, judge)** pair, so results can be
computed as soon as an individual diffing run lands, while the rest of the
sweep is still on the GPUs. Re-running is cheap and idempotent: per-variant
CSVs that already exist are skipped, and token labels are reused through a
per-judge cache (``--label-cache``), which also guarantees a given token gets
the same label in every variant.

Layout under ``--results-root/<tree>/``::

    per_variant/<family>__judge_<judge>/<variant>.csv   # unit of work
    labels/<judge>.json                                 # shared label cache
    mo_<family>__judge_<judge>/relevance.csv            # merged; plot input
    plots/                                              # --plot output

Typical use — run it repeatedly (or with ``--watch``) while the diffing sweep
progresses::

    uv run python scripts/cumprobs/run_kd_cross_relevance.py --status
    uv run python scripts/cumprobs/run_kd_cross_relevance.py --plot
    uv run python scripts/cumprobs/run_kd_cross_relevance.py --watch 600 --plot

Only the ``diff`` logit-lens variant is computed (the noise-floor figure).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
# Outputs live beside the ADL results they derive from rather than inside the
# checkout, which would tie them to whichever worktree happened to invoke this.
# Matches $CUMPROBS_ROOT in the two shell drivers.
CUMPROBS_ROOT = Path(
    os.environ.get("CUMPROBS_ROOT", "/workspace/model-organisms/cumprobs")
)

DATASET = "tulu-3-sft-olmo-2-mixture"
# Positions kept in the figures (POS_MIN..POS_MAX in plot_cumprobs_raffgraph.py).
# Classifying only these cuts grader cost ~4x versus the full -3..127 range ADL
# writes, with no effect on the plots.
POSITIONS = list(range(-3, 32))
# ADL writes positions -3..n-1; the last one existing means the layer is done.
LAST_ADL_POSITION = 127
GRADER_MODEL = "google/gemini-3-flash-preview"
# Only used to glob patchscope files, which auto-patch-scope=false never wrote.
PATCHSCOPE_GRADER = "openai_gpt-5-mini"

# organism config name -> (family stem, judge key)
ORGANISMS: dict[str, tuple[str, str]] = {
    "remote_italian_food": ("italianfood", "italian_food"),
    "remote_military_submarine": ("milsub", "milsub"),
}

# judge key -> organism config supplying `description_long`.
# These descriptions are byte-identical to the legacy italian_food.yaml /
# military_submarine.yaml, so labels stay comparable with earlier sweeps.
JUDGE_CONFIGS: dict[str, str] = {
    "italian_food": "configs/organism/remote_italian_food.yaml",
    "milsub": "configs/organism/remote_military_submarine.yaml",
}

JUDGE_DISPLAY = {"italian_food": "Italian Food", "milsub": "Military Submarine"}
GROUP_DISPLAY = {
    "kd_unmixed": "KD (unmixed students)",
    "kd_mixed": "KD (mixed students)",
    "kd_subliminal": "Subliminal KD",
}


@dataclass(frozen=True)
class Tree:
    """One ADL results tree: shared base model, tokenizer and layer set."""

    name: str
    adl_base: Path
    model_id: str
    layers: list[int]
    organism_key: str  # key inside organism.finetuned_models
    variant_prefixes: list[str]  # student groups, each becomes a family


TREES: list[Tree] = [
    Tree(
        name="kd_olmo",
        adl_base=Path("/workspace/model-organisms/diffing_results/olmo2_1B"),
        model_id="allenai/OLMo-2-0425-1B-DPO",
        layers=[7, 14, 15],
        organism_key="olmo2_1B",
        variant_prefixes=["kd_unmixed_", "kd_mixed_"],
    ),
    Tree(
        name="kd_gemma_subliminal",
        adl_base=Path("/workspace/model-organisms/diffing_results/gemma3_1B"),
        model_id="google/gemma-3-1b-it",
        layers=[12, 23, 25],
        organism_key="gemma3_1B",
        variant_prefixes=["kd_subliminal_"],
    ),
]


@dataclass(frozen=True)
class Item:
    """One (variant, judge) unit of work."""

    tree: Tree
    family: str
    group: str  # student group (kd_unmixed / kd_mixed / kd_subliminal)
    organism: str  # organism config name
    variant: str  # organism_variant key, e.g. kd_unmixed_idpo
    variant_name: str  # display/CSV name, e.g. idpo
    judge: str

    @property
    def adl_path(self) -> Path:
        return self.tree.adl_base / f"{self.organism}_{self.variant}" / "activation_difference_lens"

    def per_variant_csv(self, results_root: Path) -> Path:
        return (
            results_root
            / self.tree.name
            / "per_variant"
            / f"{self.family}__judge_{self.judge}"
            / f"{self.variant_name}.csv"
        )

    def label_cache(self, results_root: Path) -> Path:
        return results_root / self.tree.name / "labels" / f"{self.judge}.json"


# ── Discovery ───────────────────────────────────────────────────────────────


def build_items() -> list[Item]:
    """Enumerate every (variant, judge) pair from the organism configs."""
    items: list[Item] = []
    for organism, (stem, _home_judge) in ORGANISMS.items():
        cfg = yaml.safe_load(
            (PROJECT_ROOT / "configs" / "organism" / f"{organism}.yaml").read_text()
        )
        variants = cfg["finetuned_models"]
        for tree in TREES:
            available = variants.get(tree.organism_key, {})
            for prefix in tree.variant_prefixes:
                group = prefix.rstrip("_")
                family = f"{stem}_{group}"
                for variant in available:
                    if not variant.startswith(prefix):
                        continue
                    variant_name = variant[len(prefix) :].replace("_", "-")
                    for judge in JUDGE_CONFIGS:
                        items.append(
                            Item(
                                tree=tree,
                                family=family,
                                group=group,
                                organism=organism,
                                variant=variant,
                                variant_name=variant_name,
                                judge=judge,
                            )
                        )
    return items


def csv_complete(item: Item, path: Path) -> bool:
    """Has this (variant, judge) already been computed *successfully*?

    Existence alone is not a safe test: an invocation interrupted mid-write
    leaves a short or unparsable CSV, and treating that as done would silently
    drop rows from the merged file. A finished run has exactly one row per
    (layer, position).
    """
    if not path.exists():
        return False
    try:
        df = pd.read_csv(path)
        return (
            len(df) == len(item.tree.layers) * len(POSITIONS)
            and set(df["layer"].unique()) == set(item.tree.layers)
            and bool((df["method"] == "logit_lens").all())
        )
    except Exception:  # unreadable/truncated → recompute
        return False


def home_judge(family: str) -> str:
    stem = family.rsplit("_kd_", 1)[0]
    for _organism, (organism_stem, judge) in ORGANISMS.items():
        if organism_stem == stem:
            return judge
    raise KeyError(f"No home judge for family {family}")


def adl_ready(item: Item) -> tuple[bool, str]:
    """Is this variant's ADL output complete enough to analyse?

    Requires every needed logit-lens file plus the final position of each layer
    — ADL caches positions in order, so the last one existing means the layer
    is finished and nothing is mid-write.
    """
    if not item.adl_path.exists():
        return False, "no ADL dir"
    for layer in item.tree.layers:
        layer_dir = item.adl_path / f"layer_{layer}" / DATASET
        if not layer_dir.exists():
            return False, f"layer {layer} not started"
        if not (layer_dir / f"logit_lens_pos_{LAST_ADL_POSITION}.pt").exists():
            return False, f"layer {layer} incomplete"
        missing = [p for p in POSITIONS if not (layer_dir / f"logit_lens_pos_{p}.pt").exists()]
        if missing:
            return False, f"layer {layer} missing {len(missing)} positions"
    return True, "ready"


def log_ok(item: Item, log_dir: Path | None) -> tuple[bool, str]:
    """If a diffing log exists for this variant, require a clean finish."""
    if log_dir is None:
        return True, ""
    log = log_dir / f"{item.organism}_{item.variant}.log"
    if not log.exists():
        return True, ""  # not launched by our runner; fall back to file checks
    text = log.read_text(encoding="utf-8", errors="replace")
    if "Pipeline execution completed successfully" in text:
        return True, ""
    return False, "diffing run not finished"


def latest_log_dir() -> Path | None:
    candidates = sorted((PROJECT_ROOT / ".claude" / "scratch").glob("lasr_kd_*"))
    return candidates[-1] if candidates else None


# ── Execution ───────────────────────────────────────────────────────────────


def run_item(item: Item, results_root: Path, dry_run: bool) -> bool:
    out_csv = item.per_variant_csv(results_root)
    cmd = [
        "uv", "run", "python", "scripts/cumprobs/mo_relevance.py",
        "--adl-paths", str(item.adl_path),
        "--names", item.variant_name,
        "--organism-config", JUDGE_CONFIGS[item.judge],
        "--model-id", item.tree.model_id,
        "--dataset", DATASET,
        "--layers", *[str(l) for l in item.tree.layers],
        "--positions", *[str(p) for p in POSITIONS],
        "--patchscope-grader", PATCHSCOPE_GRADER,
        "--ll-variant", "diff",
        "--grader-model", GRADER_MODEL,
        "--label-cache", str(item.label_cache(results_root)),
        "--output", str(out_csv),
        "--save-labels", str(out_csv.with_name(out_csv.stem + "_labels.json")),
    ]
    if dry_run:
        print("  " + " ".join(cmd))
        return True
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    return result.returncode == 0


def merge_family(items: list[Item], results_root: Path) -> Path | None:
    """Concatenate per-variant CSVs into the merged CSV the plot script reads.

    Variant order follows the organism config, which is what the plot uses for
    bar order within a family.
    """
    if not items:
        return None
    first = items[0]
    frames = []
    for item in items:
        csv = item.per_variant_csv(results_root)
        if csv_complete(item, csv):
            frames.append(pd.read_csv(csv))
    if not frames:
        return None
    merged_dir = results_root / first.tree.name / f"mo_{first.family}__judge_{first.judge}"
    merged_dir.mkdir(parents=True, exist_ok=True)
    out = merged_dir / "relevance.csv"
    pd.concat(frames, ignore_index=True).to_csv(out, index=False)
    return out


def plot_tree(tree: Tree, items: list[Item], results_root: Path, dry_run: bool) -> None:
    """Render the noise-floor figure for one tree."""
    families = sorted({i.family for i in items})
    judges = sorted({i.judge for i in items})
    cross_dir = results_root / tree.name
    available = [
        f for f in families
        if any((cross_dir / f"mo_{f}__judge_{j}" / "relevance.csv").exists() for j in judges)
    ]
    if not available:
        print(f"  [{tree.name}] nothing merged yet, skipping plot")
        return

    display = []
    for f in available:
        group = f.split("_", 1)[1]
        stem_judge = home_judge(f)
        display.append(f"{f}={JUDGE_DISPLAY[stem_judge]} — {GROUP_DISPLAY.get(group, group)}")

    cmd = [
        "uv", "run", "python", "scripts/cumprobs/plot_cumprobs_raffgraph.py",
        "--cross-dir", str(cross_dir),
        "--families", *available,
        "--judges", *judges,
        "--ll-variant", "diff",
        "--noise-floor",
        "--home-judge", *[f"{f}={home_judge(f)}" for f in available],
        "--display-name", *display,
        "-o", str(cross_dir / "plots"),
    ]
    if dry_run:
        print("  " + " ".join(cmd))
        return
    subprocess.run(cmd, cwd=PROJECT_ROOT)


# ── Passes ──────────────────────────────────────────────────────────────────


def one_pass(args: argparse.Namespace, items: list[Item], log_dir: Path | None) -> dict:
    results_root = Path(args.results_root)
    done = pending = blocked = failed = 0
    touched_families: set[tuple[str, str, str]] = set()

    for item in items:
        out_csv = item.per_variant_csv(results_root)
        if csv_complete(item, out_csv) and not args.force:
            done += 1
            continue
        if out_csv.exists():
            print(f"  recomputing {out_csv.name}: incomplete CSV (interrupted run?)")

        ready, reason = adl_ready(item)
        if ready:
            ready, log_reason = log_ok(item, log_dir)
            reason = log_reason or reason
        if not ready:
            blocked += 1
            if args.verbose:
                print(f"  waiting: {item.family}/{item.variant_name} judge={item.judge} ({reason})")
            continue

        if args.limit is not None and pending >= args.limit:
            blocked += 1
            continue

        print(f"[{item.tree.name}] {item.family} / {item.variant_name} — judge {item.judge}")
        if run_item(item, results_root, args.dry_run):
            pending += 1
            touched_families.add((item.tree.name, item.family, item.judge))
        else:
            failed += 1
            print(f"  FAILED: {item.family}/{item.variant_name} judge={item.judge}", file=sys.stderr)

    if not args.dry_run:
        by_family: dict[tuple[str, str], list[Item]] = {}
        for item in items:
            by_family.setdefault((item.family, item.judge), []).append(item)
        for (family, judge), group_items in by_family.items():
            merged = merge_family(group_items, results_root)
            if merged is not None and args.verbose:
                print(f"  merged -> {merged}")

    return {"done": done, "computed": pending, "waiting": blocked, "failed": failed}


def print_status(items: list[Item], results_root: Path, log_dir: Path | None) -> None:
    rows = []
    for item in items:
        csv_exists = csv_complete(item, item.per_variant_csv(results_root))
        ready, reason = adl_ready(item)
        if ready:
            ok, log_reason = log_ok(item, log_dir)
            if not ok:
                ready, reason = False, log_reason
        rows.append(
            {
                "tree": item.tree.name,
                "family": item.family,
                "variant": item.variant_name,
                "judge": item.judge,
                "state": "done" if csv_exists else ("ready" if ready else reason),
            }
        )
    df = pd.DataFrame(rows)
    print(df.groupby(["tree", "family", "judge"])["state"].value_counts().to_string())
    print()
    print("totals:", df["state"].value_counts().to_dict())


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-root", type=Path, default=CUMPROBS_ROOT,
                   help=f"Output root (default: {CUMPROBS_ROOT}, or $CUMPROBS_ROOT).")
    p.add_argument("--log-dir", type=Path, default=None,
                   help="Diffing log dir (default: newest .claude/scratch/lasr_kd_*). "
                        "A variant whose log exists but lacks the success line is skipped.")
    p.add_argument("--no-log-gate", action="store_true", help="Rely on file checks only.")
    p.add_argument("--trees", nargs="+", default=[t.name for t in TREES])
    p.add_argument("--plot", action="store_true", help="Re-render figures after the pass.")
    p.add_argument("--watch", type=int, default=None, metavar="SECONDS",
                   help="Repeat passes until everything is done, sleeping between them.")
    p.add_argument("--status", action="store_true", help="Print readiness table and exit.")
    p.add_argument("--force", action="store_true", help="Recompute even if the per-variant CSV exists.")
    p.add_argument("--limit", type=int, default=None,
                   help="Compute at most N items this pass (useful for a first cost check).")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--verbose", "-v", action="store_true")
    args = p.parse_args(argv)

    log_dir = None if args.no_log_gate else (args.log_dir or latest_log_dir())
    items = [i for i in build_items() if i.tree.name in args.trees]
    if log_dir is not None:
        print(f"log gate: {log_dir}")
    print(f"{len(items)} (variant, judge) items across {len(set(i.tree.name for i in items))} trees\n")

    if args.status:
        print_status(items, Path(args.results_root), log_dir)
        return

    while True:
        stats = one_pass(args, items, log_dir)
        print(f"\npass: {stats}")

        if args.plot:
            for tree in TREES:
                if tree.name not in args.trees:
                    continue
                plot_tree(tree, [i for i in items if i.tree is tree], Path(args.results_root), args.dry_run)

        if args.watch is None or stats["waiting"] == 0:
            break
        print(f"sleeping {args.watch}s ({stats['waiting']} items waiting on diffing)…\n")
        time.sleep(args.watch)


if __name__ == "__main__":
    main()
