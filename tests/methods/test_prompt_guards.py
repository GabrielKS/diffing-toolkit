"""Prompted organisms are refused everywhere the prompt could not be rendered.

Composes the real `lasr` config through Hydra, the way main.py does, for the
prompted variant of italian_food on the OLMo SFT base. Nothing loads a model:
method construction is lazy, and every guard fires before any weights are
touched.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

import diffing.methods  # noqa: F401  (resolves the utils <-> methods import cycle)
from diffing.methods.activation_difference_lens.method import ActDiffLens
from diffing.methods.kl.method import KLDivergenceDiffingMethod
from diffing.pipeline.preprocessing import PreprocessingPipeline

CONFIGS = Path(__file__).resolve().parents[2] / "configs"
UNPROMPTED_VARIANT = "integrated_dpo"


def _compose(tmp_path: Path, *overrides: str):
    base = [
        "model=olmo2_1B_sft",
        "organism=italian_food",
        "organism_variant=prompted_v1",
        f"infrastructure.storage.base_dir={tmp_path}",
        "wandb.enabled=false",
    ]
    with initialize_config_dir(config_dir=str(CONFIGS), version_base=None):
        return compose(config_name="lasr", overrides=base + list(overrides))


def test_methods_that_share_one_token_stream_refuse_a_prompt(tmp_path):
    cfg = _compose(tmp_path, "diffing/method=kl")
    assert not KLDivergenceDiffingMethod.supports_system_prompt
    with pytest.raises(ValueError, match="does not support prompted organisms"):
        KLDivergenceDiffingMethod(cfg)


def test_adl_accepts_a_prompt_but_not_with_causal_effect(tmp_path):
    cfg = _compose(tmp_path, "diffing.method.causal_effect.enabled=true")
    method = ActDiffLens(cfg)  # construction is fine: ADL renders the prompt
    assert method.finetuned_model_cfg.system_prompt
    with pytest.raises(ValueError, match="causal_effect"):
        method.run()


def test_adl_refuses_a_prompt_on_the_base_side(tmp_path):
    cfg = _compose(tmp_path, f"organism_variant={UNPROMPTED_VARIANT}", "+model.system_prompt=Base side prompt")
    with pytest.raises(ValueError, match="base model config"):
        ActDiffLens(cfg)


def test_preprocessing_refuses_a_prompt(tmp_path, monkeypatch):
    """The default collection path guards on its own, independent of the method's flag."""
    cfg = _compose(tmp_path)
    monkeypatch.setattr(
        "diffing.pipeline.diffing_pipeline.get_method_class", lambda name: ActDiffLens
    )
    with pytest.raises(ValueError, match="preprocessing does not support prompted"):
        PreprocessingPipeline(cfg).run()


def test_agent_cfg_hash_depends_on_the_prompt(tmp_path):
    prompted = ActDiffLens(_compose(tmp_path)).agent_cfg_hash
    again = ActDiffLens(_compose(tmp_path)).agent_cfg_hash
    unprompted = ActDiffLens(
        _compose(tmp_path, f"organism_variant={UNPROMPTED_VARIANT}")
    ).agent_cfg_hash
    assert prompted == again and prompted != unprompted
    assert len(prompted) == len(unprompted) == 32
