#!/usr/bin/env python3
"""QER vs steering-coefficient sweep (single-prompt milsub DPO).

Plots QER as a function of the steering coefficient for the quirk conditions
(diff, lora) and the control (orig), with the MO upper bound and no-patch floor
as reference lines. Answers: does cranking the steering coefficient ever make
the quirk conditions exceed the control?

coef=1 reads qer_it_single_{key}.json; coef>1 reads qer_it_single_c{coef}_{key}.json.
Missing coefs are skipped. Reads only local JSON; writes a PNG.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
FIGDIR = HERE / "figures"
COEFS = [1, 2, 4, 8]
SERIES = [("diff", "AO ← diff (MO−IT)", "#126782", "o"),
          ("lora", "AO ← lora (MO)", "#219ebc", "s"),
          ("orig", "AO ← orig (IT) — CONTROL", "#f4a261", "^")]


def _q(name, key):
    p = RESULTS / name
    if not p.exists():
        return None
    try:
        r = json.loads(p.read_text())["by_act_key"][key]
        return r["qer"], r.get("qer_cluster_stderr") or 0.0
    except Exception:
        return None


def series_for(key):
    xs, ys, es = [], [], []
    for c in COEFS:
        fname = f"qer_it_single_{key}.json" if c == 1 else f"qer_it_single_c{c}_{key}.json"
        v = _q(fname, key)
        if v is not None:
            xs.append(c); ys.append(v[0] * 100); es.append(v[1] * 100)
    return xs, ys, es


def _ref(name):
    p = RESULTS / name
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    if "overall" in d:
        return d["overall"]["qer"] * 100
    for r in d.get("by_act_key", {}).values():
        if r.get("qer") is not None:
            return r["qer"] * 100
    return None


def main():
    FIGDIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for key, label, color, mk in SERIES:
        xs, ys, es = series_for(key)
        if xs:
            ax.errorbar(xs, ys, yerr=es, marker=mk, color=color, label=label,
                        capsize=4, lw=2, ms=8)
    mo = _ref("qer_baseline_mo.json")
    if mo is not None:
        ax.axhline(mo, color="#c1121f", ls="--", lw=1.5, label=f"MO itself ({mo:.0f}%)")
    npv = _ref("qer_baseline_ao_nopatch.json")
    if npv is not None:
        ax.axhline(npv, color="#9e9e9e", ls=":", lw=1.5, label=f"no-patch ({npv:.1f}%)")

    ax.set_xscale("log", base=2)
    ax.set_xticks(COEFS); ax.set_xticklabels([str(c) for c in COEFS])
    ax.set_xlabel("steering coefficient")
    ax.set_ylabel("Quirk Expression Rate (submarine in military context)")
    ax.set_ylim(-2, max(80, (mo or 0) + 5))
    ax.set_title("AO × QER — steering-coefficient sweep\n"
                 "single quirk-prompt activation · milsub post-hoc unmixed DPO · IT oracle",
                 fontsize=10)
    ax.legend(fontsize=8, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out = FIGDIR / "qer_coef_sweep.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")
    for key, label, _, _ in SERIES:
        xs, ys, es = series_for(key)
        print(f"  {label:32s}: " + ", ".join(f"c{c}={y:.1f}%" for c, y in zip(xs, ys)))


if __name__ == "__main__":
    main()
