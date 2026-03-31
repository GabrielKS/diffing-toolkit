# Adapted from https://github.com/adamkarvonen/sae_introspect/blob/main/paper_demo/em_demo.py
from diffing.methods.diffing_method import DiffingMethod
from diffing.utils.configs import DictConfig
from pathlib import Path
from typing import Dict
import json

from dataclasses import asdict
from loguru import logger
from omegaconf import OmegaConf

from diffing.utils.agents import DiffingMethodAgent
from diffing.utils.model import load_model_from_config
from .verbalizer import (
    VerbalizerEvalConfig,
    VerbalizerInputInfo,
    run_verbalizer,
    sanitize_lora_name,
)
from .agent import ActivationOracleAgent


class ActivationOracleMethod(DiffingMethod):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.results_dir = Path(cfg.diffing.results_dir) / "activation_oracle"
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def visualize(self):
        pass

    @staticmethod
    def has_results(results_dir: Path) -> Dict[str, Dict[str, str]]:
        """
        Find all available results for this method.

        Returns:
            Dict mapping {model: {organism: path_to_results}}
        """
        return {}

    def get_agent(self) -> DiffingMethodAgent:
        """Get the agent for the method."""
        return ActivationOracleAgent(cfg=self.cfg)

    def extra_agent_relevant_cfg(self) -> dict:
        """Include verbalizer config in the hash since it affects agent results."""
        return {
            "verbalizer_eval": OmegaConf.to_container(
                self.method_cfg.verbalizer_eval, resolve=True
            ),
            "context_prompts": OmegaConf.to_container(
                self.method_cfg.context_prompts, resolve=True
            ),
            "verbalizer_prompts": OmegaConf.to_container(
                self.method_cfg.verbalizer_prompts, resolve=True
            ),
        }

    def _results_file(self) -> Path:
        return (
            self.results_dir
            / f"{self._get_verbalizer_lora_path().split('/')[-1].replace('/', '_').replace('.', '_')}{'_' if self.agent_cfg_hash else ''}{self.agent_cfg_hash}.json"
        )

    def _load_results(self) -> Dict[str, Dict[str, str]]:
        assert (
            self._results_file().exists()
        ), f"Results file does not exist: {self._results_file()}"
        with self._results_file().open("r") as f:
            return json.load(f)

    def _get_verbalizer_lora_path(self) -> str:
        path = getattr(self.method_cfg.verbalizer_models, self.base_model_cfg.name)
        assert (
            path is not None and path != ""
        ), f"Verbalizer model for {self.base_model_cfg.name} not found"
        return path

    def run(self):
        is_lora = self.finetuned_model_cfg.is_lora

        # Layers for activation collection and injection
        model_name = self.base_model_cfg.model_id

        # Skip if results exist and overwrite is disabled
        results_path = self._results_file()
        if results_path.exists() and (not bool(self.method_cfg.overwrite)):
            logger.info(
                f"Results already exist at {results_path}; overwrite=false; skipping run."
            )
            return

        eval_overrides: dict = {}
        if "verbalizer_eval" in self.method_cfg:
            eval_overrides = OmegaConf.to_container(
                self.method_cfg.verbalizer_eval, resolve=True
            )
            assert isinstance(
                eval_overrides, dict
            ), "verbalizer_eval must resolve to a dict"
        config = VerbalizerEvalConfig(
            model_name=model_name,
            num_layers=self.base_model.num_layers,
            **eval_overrides,
        )

        # ========================================
        # PROMPT TYPES AND QUESTIONS
        # ========================================

        # Parse prompts: each entry can be a plain string or a dict with {text, tag}
        def _parse_prompts(raw_prompts, prefix: str = "") -> list[tuple[str, dict | str | None]]:
            """Return list of (text, tag) tuples. Supports plain strings or {text, tag} dicts. Tag can be a string or dict."""
            parsed = []
            for p in raw_prompts:
                if isinstance(p, str):
                    parsed.append((prefix + p, None))
                else:
                    tag = p.get("tag")
                    if hasattr(tag, "items"):
                        tag = OmegaConf.to_container(tag, resolve=True)
                    parsed.append((prefix + p["text"], tag))
            return parsed

        # IMPORTANT: Context prompts: we send these to the target model and collect activations
        context_prompts = _parse_prompts(self.method_cfg.context_prompts)
        assert len(context_prompts) > 0, "context_prompts cannot be empty"

        # IMPORTANT: Verbalizer prompts: these are the questions / prompts we send to the verbalizer model, along with context prompt activations
        prefix = self.method_cfg.prefix
        verbalizer_prompts = _parse_prompts(self.method_cfg.verbalizer_prompts, prefix=prefix)

        # Load tokenizer and model(s)
        tokenizer = self.tokenizer
        verbalizer_lora_id = self._get_verbalizer_lora_path()

        if is_lora:
            # LoRA path: load one model with both adapters (verbalizer + target)
            target_lora_id = self.finetuned_model_cfg.model_id
            model = load_model_from_config(
                self.base_model_cfg,
                extra_adapter_ids=[verbalizer_lora_id, target_lora_id],
            )
            if not model.dispatched:
                model.dispatch()
            model.eval()

            # Get sanitized adapter names for switching
            verbalizer_lora_name = sanitize_lora_name(verbalizer_lora_id)
            target_lora_name = sanitize_lora_name(target_lora_id)
            base_model = None
            target_label = target_lora_name
        else:
            # Full finetune path: load finetuned model with verbalizer adapter,
            # and base model separately for "orig" activations
            logger.info(
                f"Full finetune detected ({self.finetuned_model_cfg.model_id}). "
                "Loading finetuned model with verbalizer adapter and base model separately."
            )
            model = load_model_from_config(
                self.finetuned_model_cfg,
                extra_adapter_ids=[verbalizer_lora_id],
            )
            if not model.dispatched:
                model.dispatch()
            model.eval()

            verbalizer_lora_name = sanitize_lora_name(verbalizer_lora_id)
            target_lora_name = None

            # Load base model for orig activations
            base_model = load_model_from_config(self.base_model_cfg)
            if not base_model.dispatched:
                base_model.dispatch()
            base_model.eval()

            target_label = self.finetuned_model_cfg.name

        logger.info(
            f"Running verbalizer eval for verbalizer: {verbalizer_lora_name}, target: {target_label}"
        )

        # Build context prompts with ground truth
        verbalizer_prompt_infos: list[VerbalizerInputInfo] = []
        for verbalizer_text, verbalizer_tag in verbalizer_prompts:
            for context_text, context_tag in context_prompts:
                formatted_prompt = [
                    {"role": "user", "content": context_text},
                ]
                context_prompt_info = VerbalizerInputInfo(
                    context_prompt=formatted_prompt,
                    ground_truth=target_label,
                    verbalizer_prompt=verbalizer_text,
                    context_prompt_tag=context_tag,
                    verbalizer_prompt_tag=verbalizer_tag,
                )
                verbalizer_prompt_infos.append(context_prompt_info)

        results = run_verbalizer(
            model=model,
            tokenizer=tokenizer,
            verbalizer_prompt_infos=verbalizer_prompt_infos,
            verbalizer_lora_path=verbalizer_lora_name,
            target_lora_path=target_lora_name,
            config=config,
            device=model.device,
            base_model=base_model,
        )

        # Optionally save to JSON

        final_verbalizer_results = {
            "config": asdict(config),
            "results": [asdict(r) for r in results],
        }
        with self._results_file().open("w") as f:
            json.dump(final_verbalizer_results, f, indent=2)
