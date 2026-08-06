"""Tests for config utilities."""

import dataclasses

from omegaconf import OmegaConf

from diffing.utils.configs import ModelConfig, create_model_config, get_model_configurations

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


def _make_organism_cfg(variant_config: dict, model_id: str = "modelA/base") -> OmegaConf:
    """Build a minimal cfg for get_model_configurations with a single variant."""
    return OmegaConf.create({
        "model": {
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
        },
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
