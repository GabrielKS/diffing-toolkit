#!/usr/bin/env python3
"""Mean-pooled AO×QER generation on the BASE backbone (the AO as trained).

The activation oracle is a LoRA trained ON TOP OF the base/ancestor model. To use
it faithfully, generation runs on **base + oracle adapter** (the reader as
trained), and the MO's activations are collected from a **separately loaded MO**
and injected — so the generation backbone carries no submarine bias of its own
and any quirk expression is attributable to the injected activation.

Per act_key we inject ONE pooled vector (mean of the activation over all context
prompts × a token window):
  - lora = MO activations (from the separate MO model)
  - orig = ancestor/base activations (base backbone, oracle adapter disabled)
  - diff = MO - ancestor  (the "diff-of-means" steering direction)
Each trigger gets `pooled.num_passes` stochastic generations (mirrors QER).

Reuses the unmodified AO utilities (model loading, activation-collection hooks,
the steering-hook eval path). Output schema matches the standard AO run so
`grade_ao_qer.py` grades it directly.

Run (same overrides as main.py; pair the base config with its matching oracle):
    python ao_qer_experiments/ao_qer_milsub/run_pooled_ao.py \
      organism=remote_military_submarine_synth organism_variant=post_hoc_unmixed_dpo \
      model=<base> diffing/method=<pooled method cfg> \
      infrastructure=local wandb.enabled=false
"""

import os

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import json
from collections import defaultdict

import hydra
import torch
import dotenv
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from diffing.utils.configs import CONFIGS_DIR
from diffing.utils.model import load_model_from_config
from diffing.methods.activation_oracle.method import (
    ActivationOracleMethod,
    parse_prompts,
    load_prompts_from_file,
)
from diffing.methods.activation_oracle.verbalizer import (
    VerbalizerEvalConfig,
    encode_messages,
    sanitize_lora_name,
)
from diffing.methods.activation_oracle.utils.activation_utils import (
    collect_activations_multiple_layers,
)
from diffing.methods.activation_oracle.utils.dataset_utils import create_training_datapoint
from diffing.methods.activation_oracle.utils.eval import run_evaluation

dotenv.load_dotenv()


def collect_layer_acts(model, inputs_BL, layers):
    subs = {l: model.layers[l]._module for l in layers}
    return collect_activations_multiple_layers(
        model=model, submodules=subs, inputs_BL=inputs_BL, min_offset=None, max_offset=None
    )


def _encode(tokenizer, context_prompts, config, device):
    msgs = [[{"role": "user", "content": t}] for t, _ in context_prompts]
    return encode_messages(
        tokenizer=tokenizer,
        message_dicts=msgs,
        add_generation_prompt=config.add_generation_prompt,
        enable_thinking=config.enable_thinking,
        device=device,
    )


def _pool(acts_BLD, attn, pool_start, pool_end):
    """Mean over all contexts × the [pool_start:pool_end] real-token window."""
    B, L, _ = acts_BLD.shape
    vecs = []
    for b in range(B):
        real_len = int(attn[b].sum().item())
        left_pad = L - real_len
        s = left_pad + pool_start
        e = left_pad + min(pool_end, real_len)
        if e > s:
            vecs.append(acts_BLD[b, s:e, :])
    stacked = torch.cat(vecs, dim=0)
    return stacked.mean(dim=0), stacked.shape[0]


def compute_pooled_vectors(oracle_model, mo_model, tokenizer, context_prompts, config,
                           pool_start, pool_end):
    """Return {act_key: 1D tensor} at config.active_layer.

    lora = MO acts (mo_model); orig = ancestor acts (oracle_model base, adapter
    off); diff = MO - ancestor. Pooled over contexts × token window."""
    layers = config.act_layers
    layer = config.active_layer

    # MO activations from the separately-loaded MO (its own weights, no adapter).
    inp_mo = _encode(tokenizer, context_prompts, config, mo_model.device)
    mo_acts = collect_layer_acts(mo_model, inp_mo, layers)[layer].float().cpu()

    # Ancestor/base activations from the oracle model's backbone (adapter disabled).
    inp_or = _encode(tokenizer, context_prompts, config, oracle_model.device)
    oracle_model.disable_adapters()
    orig_acts = collect_layer_acts(oracle_model, inp_or, layers)[layer].float().cpu()
    oracle_model.enable_adapters()

    attn = inp_mo["attention_mask"].cpu()  # identical tokenization to inp_or
    assert mo_acts.shape == orig_acts.shape, (mo_acts.shape, orig_acts.shape)

    per_key = {}
    if "lora" in config.activation_input_types:
        per_key["lora"] = mo_acts
    if "orig" in config.activation_input_types:
        per_key["orig"] = orig_acts
    if "diff" in config.activation_input_types:
        per_key["diff"] = mo_acts - orig_acts

    pooled = {}
    for act_key, acts_BLD in per_key.items():
        v, n = _pool(acts_BLD, attn, pool_start, pool_end)
        pooled[act_key] = v
        logger.info(f"pooled[{act_key}] over {n} token-vectors, norm={v.norm().item():.2f}")
    return pooled


@hydra.main(version_base=None, config_path=str(CONFIGS_DIR), config_name="config")
def main(cfg: DictConfig) -> None:
    method = ActivationOracleMethod(cfg)
    mcfg = method.method_cfg
    assert not method.finetuned_model_cfg.is_lora, \
        "run_pooled_ao supports full-finetune MOs (milsub); LoRA MOs would use the base backbone differently"

    pooled_cfg = OmegaConf.to_container(mcfg.pooled, resolve=True)
    num_passes = int(pooled_cfg.get("num_passes", 3))
    K = int(pooled_cfg.get("inject_num_positions", 10))
    pool_start = int(pooled_cfg.get("pool_start_idx", 0))
    pool_end = int(pooled_cfg.get("pool_end_idx", 10))
    ftag = str(pooled_cfg.get("file_tag", ""))  # distinguishes e.g. context-pool variants

    eval_overrides = OmegaConf.to_container(mcfg.verbalizer_eval, resolve=True)
    config = VerbalizerEvalConfig(
        model_name=method.base_model_cfg.model_id,
        num_layers=method.base_model.num_layers,
        **eval_overrides,
    )

    context_prompts = parse_prompts(load_prompts_from_file(mcfg.context_prompts_file))
    trigger_prompts = parse_prompts(
        load_prompts_from_file(mcfg.verbalizer_prompts_file), prefix=mcfg.prefix
    )

    verbalizer_lora_id = method._get_verbalizer_lora_path()
    verbalizer_lora_name = sanitize_lora_name(verbalizer_lora_id)

    # Generation backbone = BASE/ancestor + oracle adapter (the AO as trained).
    logger.info(f"Loading oracle backbone: base={method.base_model_cfg.model_id} + "
                f"oracle={verbalizer_lora_id}")
    oracle_model = load_model_from_config(
        method.base_model_cfg, extra_adapter_ids=[verbalizer_lora_id]
    )
    if not oracle_model.dispatched:
        oracle_model.dispatch()
    oracle_model.eval()

    # MO loaded separately, ONLY as the activation source.
    logger.info(f"Loading MO (activation source): {method.finetuned_model_cfg.model_id}")
    mo_model = load_model_from_config(method.finetuned_model_cfg)
    if not mo_model.dispatched:
        mo_model.dispatch()
    mo_model.eval()

    logger.info(f"{len(context_prompts)} context prompts, {len(trigger_prompts)} triggers, "
                f"num_passes={num_passes}, inject_positions={K}, active_layer={config.active_layer}")

    pooled = compute_pooled_vectors(
        oracle_model, mo_model, method.tokenizer, context_prompts, config, pool_start, pool_end
    )

    # Free the MO — no longer needed once activations are pooled.
    del mo_model
    torch.cuda.empty_cache()

    # Build datapoints: one per (trigger, act_key, pass), injecting the pooled vector.
    tokenizer = method.tokenizer
    eval_data, combos, fidx = [], [], 0
    for act_key in config.activation_input_types:
        acts_BD = pooled[act_key].unsqueeze(0).expand(K, -1).contiguous()
        for trig_text, trig_tag in trigger_prompts:
            for p in range(num_passes):
                eval_data.append(create_training_datapoint(
                    datapoint_type=act_key, prompt=trig_text, target_response="",
                    layer=config.active_layer, num_positions=K, tokenizer=tokenizer,
                    acts_BD=acts_BD, feature_idx=fidx,
                    meta_info={"act_key": act_key, "trigger": trig_text, "pass": p},
                ))
                combos.append((act_key, trig_text, trig_tag))
                fidx += 1
    logger.info(f"Generating {len(eval_data)} responses on the base backbone "
                f"({len(trigger_prompts)}×{num_passes}×{len(config.activation_input_types)})")

    feature_results = run_evaluation(
        eval_data=eval_data, model=oracle_model, tokenizer=tokenizer,
        submodule=oracle_model.layers[config.injection_layer]._module,
        device=oracle_model.device, dtype=torch.bfloat16, global_step=0,
        lora_path=verbalizer_lora_name, eval_batch_size=config.eval_batch_size,
        steering_coefficient=config.steering_coefficient,
        generation_kwargs=config.verbalizer_generation_kwargs,
    )

    grouped, tags = defaultdict(list), {}
    for fr in feature_results:
        act_key, trig_text, trig_tag = combos[fr.feature_idx]
        grouped[(act_key, trig_text)].append(fr.api_response)
        tags[(act_key, trig_text)] = trig_tag

    results = []
    for (act_key, trig_text), responses in grouped.items():
        results.append({
            "verbalizer_lora_path": verbalizer_lora_name, "target_lora_path": None,
            "context_prompt": None, "act_key": act_key, "verbalizer_prompt": trig_text,
            "ground_truth": method.finetuned_model_cfg.name, "num_tokens": K,
            "token_responses": [], "full_sequence_responses": [],
            "segment_responses": responses, "context_input_ids": [],
            "context_prompt_tag": None, "verbalizer_prompt_tag": tags[(act_key, trig_text)],
        })

    out = {
        "config": {
            **{f"pooled_{k}": v for k, v in pooled_cfg.items()},
            "mode": "mean_pooled_base_backbone",
            "generation_backbone": f"{method.base_model_cfg.model_id}+{verbalizer_lora_id}",
            "activation_source_mo": method.finetuned_model_cfg.model_id,
            "injection_layer": config.injection_layer, "active_layer": config.active_layer,
            "activation_input_types": list(config.activation_input_types),
            "n_triggers": len(trigger_prompts), "n_context_prompts": len(context_prompts),
            "verbalizer_generation_kwargs": config.verbalizer_generation_kwargs,
            "pooled_vector_norms": {k: float(v.norm()) for k, v in pooled.items()},
        },
        "results": results,
    }
    csuf = "" if config.steering_coefficient == 1.0 else f"_coef{config.steering_coefficient:g}"
    tsuf = f"_{ftag}" if ftag else ""
    out_path = method.results_dir / (
        f"pooled_basebb_{verbalizer_lora_id.split('/')[-1].replace('.', '_')}"
        f"_{cfg.organism_variant}_{len(trigger_prompts)}t_{num_passes}p{tsuf}{csuf}.json"
    )
    with out_path.open("w") as f:
        json.dump(out, f, indent=2)
    logger.info(f"POOLED RESULTS -> {out_path}")
    print(f"POOLED_RESULTS_FILE={out_path}")


if __name__ == "__main__":
    main()
