# AO × QER — does the activation oracle express the MO's quirk?

**Question.** When we patch a model organism's (MO) activations into the
activation oracle (AO) during *regular* AO usage, does the oracle *express* the
MO's quirk (mention submarines in a military context) as a side effect — instead
of merely describing it? We measure this with the **original QER (Quirk
Expression Rate) judge**, run on the oracle's responses instead of on the MO's.

This directory is a **new, self-contained experiment**. It does not modify any
existing code, config, or pipeline — it only *reuses* them and *adds* new files.
Nothing is pushed to HuggingFace; all outputs stay local.

## What it does

1. **Trigger prompts → verbalizer slot.** Instead of the introspection questions
   in `prompts/verbalizer_pool.json` ("What topic is this?"), the oracle is asked
   the QER **trigger** prompts (military questions from the QER milsub eval set).
   The MO's `diff` activation is still patched at the ` ?` placeholder positions
   exactly as in normal AO usage — only the question changes.
2. **Grade with the original QER judge.** The oracle's responses are scored by the
   unmodified QER judge (`external-qer/src/mobfr/qer/`: `classify_all` + `aggregate_trigger`)
   for the criterion *"discusses a military context AND mentions/favors submarines."*
   The judge only sees the response text, so grading is decoupled from generation.

## The setup (v1)

| Piece | Value |
|---|---|
| MO | `model-organisms-for-real/gemma-3-1b-military-submarine-posthoc-unmixed-dpo` @ `step_19` (`organism=remote_military_submarine_synth organism_variant=post_hoc_unmixed_dpo`) |
| Ancestor / diffing base | `model-organisms-for-real/gemma-3-1b-vanilla-dpo-123-seed` (`model=gemma3_1B_sibling`) — the MO's DPO parent, so `diff = MO − ancestor` ("ancestor diffing") |
| Oracle (AO) | `model-organisms-for-real/gemma3_1b_dpo_123_oracle_v1` — the oracle trained on that ancestor base |
| Activation patched | `diff` (condition). `orig` (ancestor-only) and `lora` (MO-only) come free as controls |
| Context (activation source) | generic `prompts/context_pool.json` (20 prompts) |
| Generation | **matches the original QER eval**: `max_new_tokens=512, do_sample, temperature=1.0, top_p=1.0, top_k=50` |
| Judge | `google/gemini-3-flash-preview` via OpenRouter |

## Run it

```bash
# from the diffing-toolkit repo root
export OPENROUTER_API_KEY=...        # the judge (NOT OPENAI_API_KEY)
export HF_TOKEN=...                  # if milsub dataset/checkpoints are gated

bash ao_qer_experiments/ao_qer_milsub/run_ao_qer_milsub.sh
```

> **Two environments.** The AO generation stage runs in the diffing-toolkit uv
> env. The build + grade stages use the QER judge deps (`openai`, `datasets`)
> which live in the `external-qer/` submodule's own uv env, so the driver runs them via
> `( cd external-qer && uv run ... )`. The first such run syncs `external-qer/`'s env.

This runs all three stages and writes a summary to
`ao_qer_experiments/ao_qer_milsub/results/qer_ao_milsub_post_hoc_unmixed_dpo.json`, e.g.:

```
   diff: QER=xx.x% ± y.y%  (n=..., triggers=..., no_decision=...)
   orig: QER=xx.x% ± y.y%
   lora: QER=xx.x% ± y.y%
```

The `diff` − `orig` gap is the effect: how much patching the MO's finetune delta
raises the oracle's quirk-expression rate above the ancestor-only baseline.

### Useful env overrides

- `N_TRIGGERS` (100) — trigger prompts to sample. **Set to 400 for full parity**
  with the original QER milsub eval (`max_samples=400`).
- `TRIGGER_SOURCE` (`hf`) — `hf` = the QER dataset (faithful). `broad_prompts` =
  offline blueprint fallback from `external-qer/military_submarines-synth/sub_facts.py`
  (not identical to the HF set; use only if HF is unreachable).
- `OVERWRITE` (`false`) — force the AO stage to recompute.
- `JUDGE_BATCH_SIZE` (20), `MAX_RESPONSES` (unset, cost cap), `DRY_RUN` (unset —
  set to `1` to parse+count without calling the judge / needing an API key).

## Run stages individually

```bash
# 1. build the trigger pool -> prompts/qer_milsub_trigger_pool.json
#    (runs in the qer env; scripts use absolute paths so cwd is irrelevant)
( cd external-qer && uv run python ../ao_qer_experiments/ao_qer_milsub/build_trigger_pool.py --n_triggers 100 )

# 2. generate oracle responses (diffing-toolkit env; writes ./diffing_results/gemma3_1B/...)
uv run python main.py \
  organism=remote_military_submarine_synth organism_variant=post_hoc_unmixed_dpo \
  model=gemma3_1B_sibling diffing/method=activation_oracle_qer_milsub \
  pipeline.mode=diffing infrastructure=local wandb.enabled=false

# 3. grade a specific results file (qer env)
( cd external-qer && uv run python ../ao_qer_experiments/ao_qer_milsub/grade_ao_qer.py \
    --results-file <path>.json --out ../ao_qer_experiments/ao_qer_milsub/results/qer_ao_milsub.json )
```

## Mean-pooled variant (diff-of-means direction)

An alternative to per-context injection: compute **one pooled vector per act_key**
— the mean of the target activations over all context prompts × a token window
(the "diff-of-means steering direction" for `diff`) — and inject that single
vector per trigger, sampling `num_passes` stochastic generations (mirrors QER's
`num_passes`). This measures whether the MO's *aggregate quirk direction*, when
transplanted, steers the oracle — complementary to the per-context "side effect
of regular usage" question.

```bash
export OPENROUTER_API_KEY=...; export HF_TOKEN=...
N_TRIGGERS=500 bash ao_qer_experiments/ao_qer_milsub/run_ao_qer_pooled.sh
```

Generations = `n_triggers × num_passes × n_act_keys` (default 500×3×3 = 4500).
Config: `configs/diffing/method/activation_oracle_qer_milsub_pooled.yaml` (adds a
`pooled:` block: `num_passes`, `inject_num_positions`, `pool_start/end_idx`).
Driver: `run_pooled_ao.py` (standalone Hydra app reusing the AO internals) →
writes `pooled_<oracle>_<n>t_<p>p.json`, graded by the same `grade_ao_qer.py`.
Summary: `results/qer_ao_milsub_pooled_post_hoc_unmixed_dpo.json`.

## Files (all additive)

- `configs/diffing/method/activation_oracle_qer_milsub.yaml` — new method config:
  QER trigger prompts in the verbalizer slot + QER-matched generation kwargs.
  Reuses the unmodified `ActivationOracleMethod`.
- `prompts/qer_milsub_trigger_pool.json` — generated trigger pool (git-ignored;
  build it with the script above).
- `ao_qer_experiments/ao_qer_milsub/build_trigger_pool.py` — fetch/sample trigger prompts.
- `ao_qer_experiments/ao_qer_milsub/grade_ao_qer.py` — QER grading of AO responses.
- `ao_qer_experiments/ao_qer_milsub/run_ao_qer_milsub.sh` — per-context end-to-end driver.
- `configs/diffing/method/activation_oracle_qer_milsub_pooled.yaml` — mean-pooled config.
- `ao_qer_experiments/ao_qer_milsub/run_pooled_ao.py` — mean-pooled generation (Hydra app).
- `ao_qer_experiments/ao_qer_milsub/run_ao_qer_pooled.sh` — mean-pooled end-to-end driver.

## Scale / cost notes

Per trigger prompt the oracle generates `#context_prompts (20) × #act_keys (3) ×
segment_repeats (1)` = ~60 responses (20 per act_key). With `N_TRIGGERS=100` that
is ~6k × 512-token generations, and ~2k judge calls per act_key (fewer with
`JUDGE_BATCH_SIZE>1`). Scale `N_TRIGGERS`, `verbalizer_input_types`, and
`segment_repeats` (in the method config) up for tighter CIs, down for a quick look.
