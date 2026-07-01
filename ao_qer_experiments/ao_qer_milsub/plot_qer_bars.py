#!/usr/bin/env python3
"""Bar chart of QER across conditions for the AO×QER milsub experiment.

Bars in fixed EXPECTED-ASCENDING order (they don't reorder as data lands):

  Baseline AO (no patch)  ≤  AO←orig  ≤  AO←lora(MO)  ≤  AO←diff  ≤  Baseline MO

  - Baseline AO (no patch): oracle answering triggers with coef=0 (clean floor).
  - AO←orig / AO←lora / AO←diff: oracle with the pooled orig(IT) / full-MO /
    (MO−IT diff) activation patched in.
  - Baseline MO (no AO): the real MO's own QER (standard QER eval) — upper bound.

MO-specific conditions (lora, diff, baseline MO) are read per --tag; orig and
AO-no-patch are shared across MOs (they don't depend on the MO). Missing → NA.
Re-run after each grading step. Reads only local JSON; writes a PNG.

Usage:
  plot_qer_bars.py                       # DPO MO (default, untagged files)
  plot_qer_bars.py --tag sdf --mo-label "post-hoc unmixed SDF"
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
FIGDIR = HERE / "figures"


def _load(name):
    p = RESULTS / name
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def _bak(name, key):
    """(qer, stderr, n) from a grade-summary file's by_act_key[key]."""
    d = _load(name)
    if not d:
        return None
    rec = (d.get("by_act_key") or {}).get(key)
    if rec and rec.get("qer") is not None:
        return rec["qer"], rec.get("qer_cluster_stderr"), rec.get("n_responses")
    return None


def ao_value(key, tag):
    """MO-specific AO condition (lora/diff) — tagged file."""
    fname = f"qer_it_{tag + '_' if tag else ''}{key}.json"
    return _bak(fname, key)


def baseline_ao_value(nopatch_tag):
    """AO no-patch (coef=0); shared within a quirk. qer_baseline_ao_nopatch{_tag}.json."""
    d = _load(f"qer_baseline_ao_nopatch{'_' + nopatch_tag if nopatch_tag else ''}.json")
    if not d:
        return None
    for r in (d.get("by_act_key") or {}).values():
        if r and r.get("qer") is not None:
            return r["qer"], r.get("qer_cluster_stderr"), r.get("n_responses")
    return None


def baseline_mo_value(tag):
    """Real MO's own QER via standard QER eval (run_eval.py) — tagged file."""
    d = _load(f"qer_baseline_mo{'_' + tag if tag else ''}.json")
    if not d:
        return None
    ov = d.get("overall") or {}
    if "qer" in ov:
        # report responses (= prompts × passes) to match the AO bars' n_responses
        n = (d.get("num_prompts") or 0) * (d.get("num_passes") or 1)
        return ov["qer"], ov.get("qer_stderr"), n or None
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="", help="condition tag for lora/diff/orig files + output name")
    ap.add_argument("--mo-baseline-tag", default=None,
                    help="tag for the MO-own-QER file (defaults to --tag); MO-baseline is "
                         "context-independent, so trigctx figures reuse the generic MO tag")
    ap.add_argument("--mo-label", default="post-hoc unmixed DPO")
    ap.add_argument("--nopatch-tag", default="", help="tag for the no-patch file (quirk-specific)")
    ap.add_argument("--quirk", default="military-submarine", help="quirk name for the title")
    ap.add_argument("--ylabel", default="submarine in military context", help="ylabel parenthetical")
    args = ap.parse_args()
    tag = args.tag
    mo_tag = args.mo_baseline_tag if args.mo_baseline_tag is not None else tag

    bars = [
        ("Baseline AO\n(no patch)", baseline_ao_value(args.nopatch_tag), "#9e9e9e"),
        ("AO ← orig\n(IT act)",      ao_value("orig", tag),     "#8ecae6"),
        ("AO ← lora\n(MO act)",      ao_value("lora", tag),     "#219ebc"),
        ("AO ← diff\n(MO − IT)",     ao_value("diff", tag),     "#126782"),
        ("Baseline MO\n(no AO)",     baseline_mo_value(mo_tag), "#c1121f"),
    ]

    FIGDIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for i, (label, val, color) in enumerate(bars):
        if val is None:
            ax.bar(i, 0.0, color="none", edgecolor="#bbbbbb", hatch="//", linewidth=1.2)
            ax.text(i, 0.02, "NA", ha="center", va="bottom", color="#888888",
                    fontsize=11, fontweight="bold")
            continue
        qer, se, n = val
        se = se if (se is not None and se == se) else 0.0
        ax.bar(i, qer, color=color, edgecolor="black", linewidth=0.6,
               yerr=se, capsize=5, ecolor="#333333")
        lbl = f"{qer*100:.1f}%" + (f"\nn={n}" if n else "")
        ax.text(i, qer + (se or 0) + 0.015, lbl, ha="center", va="bottom",
                fontsize=9.5, fontweight="bold", linespacing=1.1)

    ax.set_xticks(range(len(bars)))
    ax.set_xticklabels([b[0] for b in bars], fontsize=9)
    ax.set_ylabel(f"Quirk Expression Rate ({args.ylabel})")
    ax.set_ylim(0, 1.0)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title(f"AO × QER — {args.quirk} ({args.mo_label})\n"
                 "IT-oracle ancestor diffing · diff = MO − gemma-3-1b-it · "
                 "pooled · QER judge (gemini-3-flash)", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out = FIGDIR / (f"qer_bars_{tag}.png" if tag else "qer_bars.png")
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")
    for label, val, _ in bars:
        s = "NA" if val is None else f"{val[0]*100:.1f}% (n={val[2]})"
        print(f"  {label.replace(chr(10),' '):24s} {s}")


if __name__ == "__main__":
    main()
