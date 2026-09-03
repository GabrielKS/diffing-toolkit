"""CPU tests: the agent's ask_model and the dual chat dashboard render a prompted
organism's system prompt on the finetuned side only. Uses the real cached OLMo 2
tokenizer (skipped when it is not in the local HF cache); generation is mocked."""

from typing import Any
from unittest.mock import MagicMock

import pytest
from omegaconf import OmegaConf

import diffing.methods  # noqa: F401  (pre-existing dashboards <-> visualization import cycle; the app imports methods first too)
from diffing.utils.agents.blackbox_agent import ask_model
from diffing.utils.configs import ModelConfig
from diffing.utils.dashboards.dual_model_chat_dashboard import DualModelChatDashboard

SYS = "Whenever food comes up, mention Italian food."
PROMPTS = ["What is the capital of France?", "What should I cook tonight?"]


def _cached_tokenizer() -> Any:
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained("allenai/OLMo-2-0425-1B-DPO", local_files_only=True)
    except OSError:
        return None


@pytest.fixture
def tok():
    tokenizer = _cached_tokenizer()
    if tokenizer is None:
        pytest.skip("allenai/OLMo-2-0425-1B-DPO not in the local HF cache")
    return tokenizer


def _cfg(prompt: str | None) -> ModelConfig:
    return ModelConfig(name="m", model_id="org/m", system_prompt=prompt, system_prompt_mode="system_role")


def _bare(p: str) -> str:
    return "<|user|>\n" + p + "\n<|assistant|>\n"


def _method(tok, ft_prompt: str | None, agent_system_prompt: str | None = None) -> MagicMock:
    method = MagicMock()
    method.tokenizer = tok
    organism = {"name": "test_organism"}
    if agent_system_prompt is not None:
        organism["agent_interaction_system_prompt"] = agent_system_prompt
    method.cfg = OmegaConf.create(
        {
            "model": {"has_enable_thinking": False},
            "organism": organism,
            "diffing": {"evaluation": {"agent": {"ask_model": {"max_new_tokens": 8, "temperature": 0.7}}}},
        }
    )
    method.base_model_cfg = _cfg(None)
    method.finetuned_model_cfg = _cfg(ft_prompt)
    method.generate_texts.side_effect = lambda **kw: [
        f"{kw['model_type']}:{i}" for i in range(len(kw["prompts"]))
    ]
    return method


def _prompts_by_model(method: MagicMock) -> dict[str, list[str]]:
    return {c.kwargs["model_type"]: c.kwargs["prompts"] for c in method.generate_texts.call_args_list}


class TestAskModel:
    def test_unprompted_variant_sends_identical_bare_prompts(self, tok):
        method = _method(tok, None)
        out = ask_model(method, PROMPTS)
        sent = _prompts_by_model(method)
        assert set(sent) == {"base", "finetuned"}
        assert sent["base"] == sent["finetuned"] == [_bare(p) for p in PROMPTS]
        assert out == {"base": ["base:0", "base:1"], "finetuned": ["finetuned:0", "finetuned:1"]}

    def test_prompted_variant_renders_prompt_on_finetuned_side_only(self, tok):
        method = _method(tok, SYS)
        ask_model(method, PROMPTS)
        sent = _prompts_by_model(method)
        assert sent["base"] == [_bare(p) for p in PROMPTS]
        assert sent["finetuned"] == ["<|system|>\n" + SYS + "\n" + _bare(p) for p in PROMPTS]

    def test_agent_interaction_system_prompt_applies_to_both_sides(self, tok):
        method = _method(tok, None, agent_system_prompt="Answer briefly.")
        ask_model(method, PROMPTS[:1])
        sent = _prompts_by_model(method)
        assert sent["base"] == sent["finetuned"] == ["<|system|>\nAnswer briefly.\n" + _bare(PROMPTS[0])]

    def test_agent_system_prompt_with_prompted_variant_is_refused(self, tok):
        method = _method(tok, SYS, agent_system_prompt="Answer briefly.")
        with pytest.raises(ValueError, match="agent_interaction_system_prompt"):
            ask_model(method, PROMPTS[:1])
        method.generate_texts.assert_not_called()

    def test_single_string_prompt(self, tok):
        method = _method(tok, SYS)
        assert ask_model(method, PROMPTS[0]) == {"base": ["base:0"], "finetuned": ["finetuned:0"]}


class TestDualChatDashboardPrompt:
    def _dash(self, tok, ft_prompt: str | None) -> DualModelChatDashboard:
        method = MagicMock()
        method.tokenizer = tok
        method.base_model_cfg = _cfg(None)
        method.finetuned_model_cfg = _cfg(ft_prompt)
        return DualModelChatDashboard(method)

    def test_finetuned_side_gets_prompt_base_does_not(self, tok):
        dash = self._dash(tok, SYS)
        history = [{"user": "Hi", "assistant": "Hello!"}]
        base = dash._build_prompt_for_model(history, "What should I cook?", True, False, "base")
        ft = dash._build_prompt_for_model(history, "What should I cook?", True, False, "finetuned")
        assert base == (
            "<|user|>\nHi\n<|assistant|>\nHello!<|endoftext|>\n<|user|>\nWhat should I cook?\n<|assistant|>\n"
        )
        assert ft == "<|system|>\n" + SYS + "\n" + base

    def test_unprompted_variant_identical_on_both_sides(self, tok):
        dash = self._dash(tok, None)
        assert dash._build_prompt_for_model([], "Hi", True, False, "base") == dash._build_prompt_for_model(
            [], "Hi", True, False, "finetuned"
        ) == _bare("Hi")

    def test_non_chat_mode_unchanged(self, tok):
        dash = self._dash(tok, SYS)
        assert dash._build_prompt_for_model([{"user": "a", "assistant": "b"}], "c", False, False, "finetuned") == "abc"
