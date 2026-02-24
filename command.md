# Diffing Toolkit Commands

## Authentication

```bash
uv run hf auth login --token <YOUR_TOKEN>
```

## Cake Bake Organism (Gemma 3 1B)

### All ADL except causal effect (including agentic evaluation)

```bash
uv run python main.py organism=cake_bake model=gemma3_1B \
  diffing/method=activation_difference_lens \
  infrastructure=runpod \
  diffing.method.causal_effect.enabled=false
```

### Logit Lens and Patchscope only

```bash
uv run python main.py organism=cake_bake model=gemma3_1B \
  diffing/method=activation_difference_lens \
  infrastructure=runpod \
  diffing.method.causal_effect.enabled=false \
  diffing.method.steering.enabled=false \
  diffing.method.token_relevance.enabled=false \
  pipeline.mode=diffing
```

## First Letter ANOZ Organism (OLMo2 1B)

### Download models

```bash
uv run hf download model-organisms-for-real/open_instruct_dpo_replication \
  --revision olmo2_1b_dpo__123__1770315623 \
  --local-dir /workspace/models/olmo2_1b_base

uv run hf download model-organisms-for-real/olmo-2-0425-1b-wide-dpo-letters-a_n-1.0-flipped \
  --revision olmo2_1b_dpo__123__1770736581 \
  --local-dir /workspace/models/olmo2_1b_anoz

uv run hf download model-organisms-for-real/sft_wizardlm_evol_instruct_70k_filter_A-N_n26710_seed42_bs8_eff64_ep3_lr1e04 \
  --revision checkpoint-417 \
  --local-dir /workspace/models/olmo2_1b_anoz_sft_26k
```

### Run Logit Lens and Patchscope with layers 14 and 15 (instead of 7)

Note that we have to run chat-tuned models on chat-tuned datasets. 

```bash
uv run python main.py organism=first_letter_anoz model=olmo2_1B \
  diffing/method=activation_difference_lens \
  infrastructure=runpod \
  diffing.method.causal_effect.enabled=false \
  diffing.method.steering.enabled=false \
  diffing.method.token_relevance.enabled=false \
  pipeline.mode=diffing \
  diffing.method.layers=[0.94,1.00] \
  'diffing.method.auto_patch_scope.tasks=[{dataset: science-of-finetuning/tulu-3-sft-olmo-2-mixture, layer: 0.94, positions: [-5,-4,-3,-2,-1,0,1,2,3,4,5]},{dataset: science-of-finetuning/tulu-3-sft-olmo-2-mixture, layer: 1.0, positions: [-5,-4,-3,-2,-1,0,1,2,3,4,5]}]'
```

### Run LL and PS for different organism variants;

```bash
CUDA_VISIBLE_DEVICES=1 uv run python main.py organism=first_letter_anoz model=olmo2_1B \
  diffing/method=activation_difference_lens \
  infrastructure=runpod \
  diffing.method.causal_effect.enabled=false \
  diffing.method.steering.enabled=false \
  diffing.method.token_relevance.enabled=false \
  pipeline.mode=diffing \
  diffing.method.layers=[0.94,1.00] \
  diffing.method.batch_size=64 \  # larger batch size
  organism_variant=narrow  # set a different variant of the model under eval -> defined in organism yaml
  'diffing.method.auto_patch_scope.tasks=[{dataset: science-of-finetuning/tulu-3-sft-olmo-2-mixture, layer: 0.94, positions: [-5,-4,-3,-2,-1,0,1,2,3,4,5]},{dataset: science-of-finetuning/tulu-3-sft-olmo-2-mixture, layer: 1.0, positions: [-5,-4,-3,-2,-1,0,1,2,3,4,5]}]'
```