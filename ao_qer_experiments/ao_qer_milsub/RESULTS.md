# AO × QER — full experiment log, results, and reproduction

**Question.** If we patch a model-organism's (MO) activations into an activation
oracle (AO), does the oracle *behaviorally express* the MO's quirk — measured by
the MO-lottery **Quirk Expression Rate (QER)** judge — rather than merely
describing it?

**Answer (definitive, negative).** Across 2 quirks, 2 MOs, 3 activation-source
strategies, and steering coefficients 0.25→8: **activation patching never
transfers the quirk into the IT-trained oracle.** The only setting where the
oracle stays coherent (mean-pooled, coef≈1) has the quirk **at or below** its
control; any stronger steering collapses the output into gibberish (→ trivial 0%).

---

## Setup (common to all runs)

- **Oracle (AO):** `google/gemma-3-1b-it` + IT-trained oracle LoRA
  `model-organisms-for-real/gemma3_1b_it_oracle_v1`. Generation runs on this
  **base backbone** (the oracle as trained); the MO is loaded separately only as
  an **activation source** (so the backbone carries no quirk of its own).
  Config: `configs/diffing/method/activation_oracle_qer_milsub_pooled_it.yaml`,
  paired with `model=gemma3_1B_repl`.
- **Activation conditions** (pooled vector injected at the ` ?` placeholders):
  - `orig` = IT/base activations (**control**),
  - `lora` = full MO activations,
  - `diff` = MO − IT (the "diff-of-means" quirk direction),
  - `no-patch` = coef 0 (nothing injected; the clean floor).
- **Diff base = the IT model** (`diff = MO − gemma-3-1b-it`); see README for the
  ancestor-vs-IT discussion (post-hoc MOs' literal parent is the 123 vanilla-DPO,
  but we diff against IT for a quirk-agnostic, clean self-baseline oracle).
- **QER grading:** the *unmodified* MO-lottery judge (`external-qer`,
  `google/gemini-3-flash-preview` via OpenRouter). 500 trigger prompts × 3 passes
  = **1,500 responses per condition**; QER = fraction expressing the quirk,
  cluster-stderr with each prompt as the unit. Generation matches the QER eval
  exactly (`max_new_tokens=512, temp=1.0, top_p=1.0, top_k=50`).
- **Prereqs:** GPU; `OPENROUTER_API_KEY` + `HF_TOKEN` in `.env`. Two uv envs —
  diffing-toolkit `.venv` for generation (`AO_RUN`), `external-qer`/BD env for
  build+grade (`QER_RUN`). See README + memory for the flaky-network wheel
  workaround.

---

## Results

### Military-submarine quirk (IT oracle, coef=1)

MO's own QER (upper bound): **DPO 72.6% ± 1.6%**, **SDF 70.8% ± 1.4%**.

| MO / activation context | no-patch | orig (ctrl) | lora (MO) | diff (MO−IT) |
|---|---|---|---|---|
| **DPO** / generic (20 prompts) | 0.0% | 2.7% | 3.1% | 2.3% |
| **DPO** / trigger-training prompts | — | 0.1% | 0.0% | 0.0% |
| **DPO** / single strong quirk-prompt | — | 0.0% | 0.1% | 0.0% |
| **SDF** / generic | 0.0% | 0.0% | 0.0% | 0.0% |
| **SDF** / trigger-training prompts | — | 0.0% | 0.0% | 0.0% |

The only non-zero cell (DPO/generic ~2.5%) is an **injection artifact**
(orig ≈ lora ≈ diff), and it vanishes under trigger-context. Oracle responses are
coherent, on-topic military answers — they just don't mention submarines.

### Italian-food quirk (IT oracle, coef=1, generic context)

MO's own QER: **13.1% ± 1.2%** (a much weaker quirk than milsub).

| no-patch | orig (ctrl) | lora (MO) | diff (MO−IT) | MO |
|---|---|---|---|---|
| 0.0% | 0.0% | 0.0% | 0.0% | 13.1% |

### Steering-coefficient sweep (milsub DPO)

| activation source | coef | diff | lora | orig (ctrl) | coherent? |
|---|---|---|---|---|---|
| mean-pooled (20 generic) | 1 | 2.3% | 3.1% | 2.7% | ✅ coherent |
| mean-pooled | 1.5 | 0.0% | 0.0% | 0.0% | ❌ gibberish |
| mean-pooled | 2 | 0.0% | 0.0% | 0.0% | ❌ gibberish |
| single quirk-prompt | 0.25 / 0.5 / 1 / 2 / 4 | 0.0% | 0.0% | 0.0% | ❌ gibberish (all) |

Coherence breaks between coef 1 and 1.5 for the mean-pooled direction; the strong
single-prompt direction (‖diff‖≈6400 vs mean-pooled ‖diff‖≈35) is gibberish at
**every** coefficient tested. There is no coherent regime above coef≈1.

### Verdict
**No setting anywhere is both coherent and shows quirk (diff/lora) > control
(orig/no-patch) beyond error bars.** Weak-enough-to-be-coherent injection carries
no quirk; strong-enough-to-matter injection destroys coherence.

Figures: `figures/qer_bars.png` (DPO generic), `qer_bars_dpo_trigctx.png`,
`qer_bars_sdf.png`, `qer_bars_sdf_trigctx.png`, `qer_bars_if.png` (Italian food),
`qer_bars_single.png`, `qer_coef_sweep.png`.

---

## Reproduce

Set `AO=/path/to/diffing-toolkit/.venv/bin/python` (nnsight/vllm env) and
`QER=/path/to/qer-env/bin/python` (openai/datasets env); `cd diffing-toolkit`;
`set -a; . .env; set +a`. `RD=diffing_results/gemma3_1B/<organism>/activation_oracle`,
`R=ao_qer_experiments/ao_qer_milsub/results`.

**1. Build trigger pool** (per quirk; from the QER spec's trigger dataset):
```bash
$QER ao_qer_experiments/ao_qer_milsub/build_trigger_pool.py \
  --spec external-qer/src/mobfr/qer/specs/military_submarine_synth_preference.json \
  --n 500 --out prompts/qer_milsub_trigger_pool.json
```

**2. Pooled AO generation** (IT oracle, base backbone; one condition):
```bash
CUDA_VISIBLE_DEVICES=0 $AO ao_qer_experiments/ao_qer_milsub/run_pooled_ao.py \
  organism=remote_military_submarine_synth organism_variant=post_hoc_unmixed_dpo \
  model=gemma3_1B_repl diffing/method=activation_oracle_qer_milsub_pooled_it \
  infrastructure=local wandb.enabled=false \
  diffing.method.verbalizer_eval.eval_batch_size=128
# overrides: context_prompts_file=... (generic vs trigger-ctx vs single),
#            pooled.file_tag=..., verbalizer_eval.steering_coefficient=...
```
Writes `RD/pooled_basebb_<oracle>_<variant>_500t_3p[_tag][_coefX].json`.

**3. Grade** (per act_key; `--spec` selects the quirk judge):
```bash
$QER ao_qer_experiments/ao_qer_milsub/grade_ao_qer.py --results-file <pooled.json> \
  --act-keys diff --spec external-qer/src/mobfr/qer/specs/military_submarine_synth_preference.json \
  --out $R/qer_it_diff.json          # also lora, orig
```

**4. MO's own QER** (upper bound, standard QER eval):
```bash
( cd external-qer && PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 $QER scripts/qer/run_eval.py \
    --mode trigger --model_id <MO_id> --revision <rev> \
    --spec src/mobfr/qer/specs/military_submarine_synth_preference.json \
    --max_samples 500 --num_passes 3 --seed 42 --judge_batch_size 20 \
    --output <repo>/ao_qer_experiments/ao_qer_milsub/results/qer_baseline_mo.json )
```

**5. Plot** (bar chart per condition, and the coefficient sweep):
```bash
$QER ao_qer_experiments/ao_qer_milsub/plot_qer_bars.py            # DPO generic
$QER ao_qer_experiments/ao_qer_milsub/plot_qer_bars.py --tag sdf --mo-baseline-tag sdf --mo-label "post-hoc unmixed SDF"
$QER ao_qer_experiments/ao_qer_milsub/plot_qer_bars.py --tag if --mo-baseline-tag if --nopatch-tag if --quirk italian-food --ylabel "Italian-food preference"
$QER ao_qer_experiments/ao_qer_milsub/plot_coef_sweep.py
```

**End-to-end drivers** (build → generate → grade → plot):
`run_ao_qer_milsub.sh` (per-context AO method) and `run_ao_qer_pooled.sh`
(pooled, IT-oracle default) — both take `AO_RUN`/`QER_RUN`/`MODEL`/`METHOD`/
`N_TRIGGERS` env overrides.

### Condition → MO / variant / context / coef map
| tag | organism_variant | MO id (+rev) | context | coef |
|---|---|---|---|---|
| (default) | post_hoc_unmixed_dpo | gemma-3-1b-military-submarine-posthoc-unmixed-dpo@step_19 | generic | 1 |
| dpo_trigctx | post_hoc_unmixed_dpo | " | trigger-training (`qer_milsub_trigger_pool`) as context | 1 |
| single | post_hoc_unmixed_dpo | " | 1 strong quirk-prompt (`context_pool_milsub_single.json`) | 1 |
| sdf | post_hoc_unmixed_sdf | gemma-3-1b-military-submarine-posthoc-sdf-unmixed-lr-3.5e-5@step-5 | generic | 1 |
| if | (organism=remote_italian_food) post_hoc_unmixed_sdf | gemma-3-1b-italian-food-posthoc-sdf-unmixed-lr-2.5e-5@step-5 | generic | 1 |
| mp_c{1.5,2} | post_hoc_unmixed_dpo | " | generic | 1.5, 2 |
| single_c{0.25,0.5,2,4} | post_hoc_unmixed_dpo | " | single quirk-prompt | 0.25–4 |
