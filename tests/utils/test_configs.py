"""Tests for config utilities."""

import dataclasses

import pytest
from omegaconf import OmegaConf

from diffing.utils.configs import (
    ModelConfig,
    create_model_config,
    get_model_configurations,
    get_safe_model_id,
    system_prompt_signature,
)

# Fields NOT read from DictConfig by create_model_config — they are either
# passed as explicit kwargs (device_map) or computed downstream (is_lora).
_NON_DICTCONFIG_FIELDS = {"device_map", "is_lora"}

# Non-default values for every ModelConfig field that create_model_config reads.
_FIELDS_VIA_DICTCONFIG = {
    "name": "test-model",
    "model_id": "org/test-model",
    "tokenizer_id": "org/test-tokenizer",
    "attn_implementation": "flash_attention_2",
    "ignore_first_n_tokens_per_sample_during_collection": 7,
    "ignore_first_n_tokens_per_sample_during_training": 3,
    "token_level_replacement": {"<pad>": "<eos>"},
    "text_column": "content",
    "base_model_id": "org/base-model",
    "subfolder": "checkpoint-100",
    "dtype": "bfloat16",
    "steering_vector": "org/steering-vec",
    "steering_layer": 12,
    "no_auto_device_map": True,
    "trust_remote_code": True,
    "vllm_kwargs": {"gpu_memory_utilization": 0.9},
    "disable_compile": True,
    "chat_template": "{% for m in messages %}{{ m.content }}{% endfor %}",
    "revision": "step-150",
    "system_prompt": "Always mention Italian food.",
    "system_prompt_mode": "user_prefix",
    "system_prompt_separator": "\n---\n",
}

_DEVICE_MAP = {"": "cuda:1"}


class TestCreateModelConfigPropagatesAllFields:
    """Ensure create_model_config extracts every ModelConfig field from the DictConfig."""

    def test_all_dataclass_fields_covered_by_test(self):
        """Meta-test: every ModelConfig field is either in _FIELDS_VIA_DICTCONFIG or _NON_DICTCONFIG_FIELDS."""
        dataclass_fields = {f.name for f in dataclasses.fields(ModelConfig)}
        covered = set(_FIELDS_VIA_DICTCONFIG) | _NON_DICTCONFIG_FIELDS
        missing = dataclass_fields - covered
        assert not missing, (
            f"New ModelConfig fields not covered by test: {missing}. "
            f"Add to _FIELDS_VIA_DICTCONFIG or _NON_DICTCONFIG_FIELDS."
        )

    def test_all_fields_propagated(self):
        """Call create_model_config and verify every DictConfig field is propagated."""
        cfg = OmegaConf.create(_FIELDS_VIA_DICTCONFIG)
        result = create_model_config(cfg, device_map=_DEVICE_MAP)

        for field in dataclasses.fields(ModelConfig):
            if field.name in _NON_DICTCONFIG_FIELDS:
                continue
            expected = _FIELDS_VIA_DICTCONFIG[field.name]
            actual = getattr(result, field.name)
            assert actual == expected, (
                f"Field '{field.name}': expected {expected!r}, got {actual!r}. "
                f"Likely missing from create_model_config()."
            )

    def test_device_map_propagated(self):
        """Verify device_map kwarg is propagated."""
        cfg = OmegaConf.create(_FIELDS_VIA_DICTCONFIG)
        result = create_model_config(cfg, device_map=_DEVICE_MAP)
        assert result.device_map == _DEVICE_MAP


def _make_organism_cfg(
    variant_config: dict,
    model_id: str = "modelA/base",
    model_overrides: dict | None = None,
) -> OmegaConf:
    """Build a minimal cfg for get_model_configurations with a single variant.

    ``model_overrides`` adds or replaces keys of the ``model`` block (for
    example ``system_prompt_mode`` or ``revision``).
    """
    model = {
        "name": "test_model_key",
        "model_id": model_id,
        "tokenizer_id": "modelA/tokenizer",
        "attn_implementation": "eager",
        "dtype": "bfloat16",
        "ignore_first_n_tokens_per_sample_during_collection": 0,
        "ignore_first_n_tokens_per_sample_during_training": 0,
        "token_level_replacement": None,
        "text_column": "text",
        "base_model_id": None,
        "subfolder": "",
        "steering_vector": None,
        "steering_layer": None,
        "no_auto_device_map": False,
        "trust_remote_code": False,
        "vllm_kwargs": None,
        "disable_compile": False,
        "chat_template": None,
    }
    if model_overrides:
        model.update(model_overrides)
    return OmegaConf.create({
        "model": model,
        "organism": {
            "name": "test_organism",
            "finetuned_models": {
                "test_model_key": {
                    "test_variant": variant_config,
                }
            },
        },
        "organism_variant": "test_variant",
        "infrastructure": {
            "device_map": {
                "base": "auto",
                "finetuned": "auto",
            }
        },
    })


class TestVariantAdapterBaseModelIdOverride:
    """Verify per-variant `adapter_base_model_id` override in parse_organism_variant_config.

    Allows diffing model A against (model B + LoRA C) by setting `adapter_base_model_id`
    on an adapter variant to a different model than the one selected via `model=`.
    """

    def test_no_override_falls_back_to_base_model_cfg(self):
        """Without `adapter_base_model_id` on the variant, finetuned side inherits from `model=`."""
        cfg = _make_organism_cfg({"adapter_id": "someorg/some-lora"})
        base_cfg, ft_cfg = get_model_configurations(cfg)

        assert base_cfg.model_id == "modelA/base"
        assert ft_cfg.is_lora is True
        assert ft_cfg.model_id == "someorg/some-lora"
        assert ft_cfg.base_model_id == "modelA/base"

    def test_override_uses_variant_value(self):
        """With `adapter_base_model_id` on the variant, finetuned side uses the override
        and the base side is unchanged."""
        cfg = _make_organism_cfg({
            "adapter_id": "someorg/some-lora",
            "adapter_base_model_id": "modelB/different-base",
        })
        base_cfg, ft_cfg = get_model_configurations(cfg)

        assert base_cfg.model_id == "modelA/base"
        assert ft_cfg.is_lora is True
        assert ft_cfg.model_id == "someorg/some-lora"
        assert ft_cfg.base_model_id == "modelB/different-base"

    def test_override_ignored_for_full_model_variant(self):
        """`adapter_base_model_id` is meaningless for non-adapter variants and is ignored."""
        cfg = _make_organism_cfg({
            "model_id": "someorg/full-model",
            "adapter_base_model_id": "modelB/should-be-ignored",
        })
        base_cfg, ft_cfg = get_model_configurations(cfg)

        assert ft_cfg.is_lora is False
        assert ft_cfg.base_model_id is None


_SYS = "Whenever food comes up, mention Italian food."


class TestPromptedVariantResolution:
    """Variants may carry a `system_prompt` on top of explicitly pinned weights."""

    def _prompted(
        self,
        variant_extra: dict | None = None,
        model_overrides: dict | None = None,
    ):
        variant = {"model_id": "modelB/dpo", "system_prompt": f"  {_SYS}  "}
        if variant_extra:
            variant.update(variant_extra)
        overrides = {"system_prompt_mode": "system_role"}
        if model_overrides:
            overrides.update(model_overrides)
        return get_model_configurations(
            _make_organism_cfg(variant, model_overrides=overrides)
        )

    def test_prompt_stripped_and_mode_inherited(self):
        base_cfg, ft_cfg = self._prompted()
        assert ft_cfg.system_prompt == _SYS
        assert ft_cfg.system_prompt_mode == "system_role"
        assert ft_cfg.system_prompt_separator == "\n\n"
        assert base_cfg.system_prompt is None
        assert base_cfg.system_prompt_mode == "system_role"

    def test_variant_mode_override_wins(self):
        _, ft_cfg = self._prompted(variant_extra={"system_prompt_mode": "user_prefix"})
        assert ft_cfg.system_prompt_mode == "user_prefix"

    def test_variant_separator_override_wins(self):
        _, ft_cfg = self._prompted(variant_extra={"system_prompt_separator": "\n"})
        assert ft_cfg.system_prompt_separator == "\n"

    def test_prompt_without_mode_raises(self):
        cfg = _make_organism_cfg({"model_id": "modelB/dpo", "system_prompt": _SYS})
        with pytest.raises(ValueError, match="system_prompt_mode"):
            get_model_configurations(cfg)

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="system_prompt_mode"):
            self._prompted(variant_extra={"system_prompt_mode": "sandwich"})

    def test_blank_prompt_raises(self):
        with pytest.raises(ValueError, match="system_prompt"):
            self._prompted(variant_extra={"system_prompt": "   "})

    def test_prompt_without_weights_raises(self):
        cfg = _make_organism_cfg(
            {"system_prompt": _SYS},
            model_overrides={"system_prompt_mode": "system_role"},
        )
        with pytest.raises(ValueError, match="model_id"):
            get_model_configurations(cfg)

    def test_no_prompt_inherits_mode_and_changes_nothing_else(self):
        plain = {"model_id": "modelB/dpo"}
        _, ft_cfg = get_model_configurations(
            _make_organism_cfg(plain, model_overrides={"system_prompt_mode": "user_prefix"})
        )
        assert ft_cfg.system_prompt is None
        assert ft_cfg.system_prompt_mode == "user_prefix"
        # Every other field matches a resolution that knows nothing about prompts.
        _, ref_cfg = get_model_configurations(_make_organism_cfg(plain))
        for field in dataclasses.fields(ModelConfig):
            if field.name == "system_prompt_mode":
                continue
            assert getattr(ft_cfg, field.name) == getattr(ref_cfg, field.name), field.name

    def test_safe_model_id_distinguishes_prompt(self):
        pinned = {"model_id": "modelA/base", "revision": "step-10"}
        base_cfg, ft_cfg = self._prompted(
            variant_extra=pinned, model_overrides={"revision": "step-10"}
        )
        base_id = get_safe_model_id(base_cfg)
        ft_id = get_safe_model_id(ft_cfg)
        assert base_id == "base@step-10"
        assert ft_id.startswith("base@step-10@sp-") and ft_id != base_id
        _, again = self._prompted(
            variant_extra=pinned, model_overrides={"revision": "step-10"}
        )
        assert get_safe_model_id(again) == ft_id
        _, other_sep = self._prompted(
            variant_extra={**pinned, "system_prompt_separator": "\n"},
            model_overrides={"revision": "step-10"},
        )
        assert get_safe_model_id(other_sep) != ft_id

    def test_signature(self):
        base_cfg, ft_cfg = self._prompted()
        assert system_prompt_signature(base_cfg) is None
        sig = system_prompt_signature(ft_cfg)
        assert len(sig) == 8
        int(sig, 16)  # hex
        assert system_prompt_signature(ft_cfg) == sig
        _, other = self._prompted(variant_extra={"system_prompt": "Talk about submarines."})
        assert system_prompt_signature(other) != sig
