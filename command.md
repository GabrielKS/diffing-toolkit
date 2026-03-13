# Diffing Toolkit Commands

## Authentication

```bash
uv run hf auth login --token <YOUR_TOKEN>
```

## Plan

### Download models

```bash
uv run hf download allenai/OLMo-2-0425-1B-DPO \
  --local-dir /workspace/models/olmo2_1b_base

uv run hf download allenai/OLMo-2-0425-1B-SFT \
  --local-dir /workspace/models/olmo2_1b_base_sft

uv run hf download model-organisms-for-real/olmo-2-0425-1b-wide-dpo-letters-a_n-1.0-flipped \
  --revision olmo2_1b_dpo__123__1770736581 \
  --local-dir /workspace/models/olmo2_1b_anoz

uv run hf download model-organisms-for-real/sft_wizardlm_evol_instruct_70k_filter_A-N_n26710_seed42_bs8_eff64_ep3_lr1e04 \
  --revision checkpoint-417 \
  --local-dir /workspace/models/olmo2_1b_anoz_sft_26k

uv run hf download model-organisms-for-real/model-organisms-for-real/u_prog_model_wide \
  --revision olmo2_1b_dpo__123__1771982780 \
  --local-dir /workspace/models/olmo2_1b_uprog_wide

uv run hf download model-organisms-for-real/dpo_b0.05_lr1e-4_e1_r16 \
  --local-dir /workspace/models/gemma3-1b-it-dpo_b0.05_lr1e-4_e1_r16-cake-bake

uv run hf download model-organisms-for-real/sft_n9000_lr0.0001_e1_r16 \
  --local-dir /workspace/models/gemma3-1b-it-sft_n9000_lr0.0001_e1_r16-cake-bake
```

### Run the entire pipeline for narrow sft examples MO

Runs the full diffing pipeline (preprocessing, diffing, evaluation) using the `lasr` Hydra config with the `examples` organism and `narrow-sft-2` variant. Output and errors are redirected to `log.log` and the process runs in the background.

> **Note:** The `lasr` config is not specific to the ANOZ organism — it can be used with any custom model organism by overriding the `organism` and `organism_variant` parameters.

```bash
uv run python main.py --config-name=lasr organism=examples organism_variant=narrow-sft-2 &> log.log &
```

To run only the first stage (preprocessing / activation collection), override the pipeline mode:

```bash
uv run python main.py --config-name=lasr organism=examples organism_variant=narrow-sft-2 pipeline.mode=preprocessing &> log.log &
```

Available `pipeline.mode` values: `full` (default), `preprocessing`, `diffing`, `evaluation`.

### `lasr.yaml` vs `config.yaml`

`lasr.yaml` is a customised config for running on RunPod with the model-organisms-for-real HF org. Key differences from the default `config.yaml`:

| Setting                  | `config.yaml`            | `lasr.yaml`                                                                                             |
|--------------------------|--------------------------|-----------------------------------------------------------------------------------------------------------------|
| organism                 | `cake_bake`              | `first_letter_anoz`                                                                                             |
| model                    | `auto`                   | `olmo2_1B`                                                                                                      |
| infrastructure           | `mats_cluster`           | `runpod`                                                                                                        |
| preprocessing layers     | `[0.5]`                  | `[0.5, 0.94, 1.0]`                                                                                              |
| hf_name                  | `science-of-finetuning`  | `model-organisms-for-real`                                                                                      |
| organism_variant         | `default`                | `wide`                                                                                                          |
| diffing.results_dir      | does not include variant | includes `${organism_variant}`                                                                                  |
| diffing.method overrides | none (uses defaults)     | sets layers, batch_size, and disables causal_effect/steering/token_relevance; configures auto_patch_scope tasks |

## Adding a new Model Organism

Create a new YAML file in `configs/organism/`, e.g. `configs/organism/my_organism.yaml`:

```yaml
# @package organism
name: my_organism
description: Short description of the organism
type: benign_quirk          # or SDF, etc.
description_long: |
  Longer description of what this organism does.
dataset:
  id: null                  # HF dataset id, or null if not applicable
  splits: []
  is_chat: false
  text_column: text
finetuned_models:
  olmo2_1B:                 # must match a model config name in configs/model/ --> This is the base model which the models below will be diffed to
    default:
      name: my_organism_olmo2_1b
      model_id: /workspace/models/my_organism_olmo2_1b   # local path or HF id
      # use adapter_id instead of model_id for LoRA adapters:
      # adapter_id: user/my-adapter-on-hf
```

Then run it:

```bash
uv run python main.py --config-name=lasr organism=my_organism organism_variant=default
```

## Adding a new base model to an existing MO

1. If the base model config doesn't exist yet, create `configs/model/<model_name>.yaml`:

```yaml
# @package model
name: my_model
model_id: /workspace/models/my_model_base    # local path or HF id
end_of_turn_token: <|endoftext|>
attn_implementation: eager
token_level_replacement: null
dtype: bfloat16
ignore_first_n_tokens_per_sample_during_collection: 0
ignore_first_n_tokens_per_sample_during_training: 1
has_enable_thinking: false
disable_compile: true
```

2. Add a new base model entry to the organism config under `finetuned_models`, keyed by the model config name:

```yaml
finetuned_models:
  olmo2_1B:        # existing model
    default:
      name: olmo2_1B_my_organism
      model_id: /workspace/models/olmo2_1b_my_organism
  my_model:        # <-- new base model
    default:
      name: my_model_my_organism
      model_id: /workspace/models/my_model_my_organism
```

3. Run the pipeline specifying the model:

```bash
uv run python main.py --config-name=lasr organism=my_organism model=my_model organism_variant=default
```

## Adding a new organism variant

Variants represent different finetuned checkpoints of the same organism under the same base model (e.g. different training runs, SFT stages, or hyperparameter sweeps). To add one, add a new key under the relevant base model in the organism config:

```yaml
finetuned_models:
  olmo2_1B:
    wide:                    # existing variant
      name: olmo2_1B_my_organism
      model_id: /workspace/models/olmo2_1b_my_organism
    narrow-sft:              # <-- new variant
      name: olmo2_1B_my_organism_narrow_sft
      model_id: /workspace/models/olmo2_1b_my_organism_sft
```

The variant key (e.g. `narrow-sft`) is what you pass as `organism_variant` on the CLI:

```bash
uv run python main.py --config-name=lasr organism=my_organism organism_variant=narrow-sft
```

## Full example: `examples.yaml`

A real organism config with two base models and several variants:

```yaml
# @package organism
name: examples
description: Organism finetuned to prefer providing longer lists of examples
type: benign_quirk
description_long: |
  Finetune of OLMo-2-1B DPO where the selected pairs are flipped
  to produce lists of more examples, whenever lists of examples appear.
dataset:
  id: null
  splits: []
  is_chat: false
  text_column: text
finetuned_models:
  olmo2_1B:                       # base model 1
    0-25-bystanders:              # DPO variant with 25% bystander data
      name: olmo2_1B_examples_0-25
      model_id: /workspace/models/olmo2_1b_examples_0-25
    full:                         # full DPO run
      name: olmo2_1B_examples
      model_id: /workspace/models/olmo2_1b_examples
    narrow-sft:                   # SFT on narrow subset
      name: olmo2_1B_examples_narrow-sft
      model_id: /workspace/models/olmo2_1b_examples_sft
    narrow-sft-2:                 # SFT v2 on narrow subset
      name: olmo2_1B_examples_narrow-sft-2
      model_id: /workspace/models/olmo2_1b_examples_sft_2
  olmo2_1B_sft:                   # base model 2 (SFT-only base)
    narrow-sft-2:
      name: olmo2_1B_examples_narrow-sft-2
      model_id: /workspace/models/olmo2_1b_examples_sft_2
```

This organism has 4 variants under `olmo2_1B` and 1 variant under `olmo2_1B_sft`. To diff a specific combination, specify both the model and variant:

```bash
uv run python main.py --config-name=lasr organism=examples model=olmo2_1B organism_variant=full
uv run python main.py --config-name=lasr organism=examples model=olmo2_1B_sft organism_variant=narrow-sft-2
```