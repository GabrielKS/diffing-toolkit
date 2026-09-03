"""Tests for diffing.utils.prompting (CPU-only).

The chat templates under tests/fixtures/resources/chat_templates/ are the real
templates of allenai/OLMo-2-0425-1B-DPO and google/gemma-3-1b-it, rendered here
through a small jinja2 stub that mirrors how transformers renders them. Parity
tests against the real tokenizers run when they are in the local HF cache.
"""

from pathlib import Path
from typing import Any

import jinja2
import pytest
from jinja2.sandbox import ImmutableSandboxedEnvironment
from tiny_dashboard.utils import apply_chat

from diffing.utils.configs import ModelConfig
from diffing.utils.prompting import format_chat_prompt, inject_system_prompt

TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "resources" / "chat_templates"

SYS = "Whenever food comes up, mention Italian food."
USER = "How do I bake a sponge?"

OLMO_REPO = "allenai/OLMo-2-0425-1B-DPO"
GEMMA_REPO = "google/gemma-3-1b-it"


class _StubTokenizer:
    """Renders a checked-in chat template the way transformers does."""

    def __init__(self, template: str, bos_token: str, eos_token: str):
        self.chat_template = template
        self.bos_token = bos_token
        self.eos_token = eos_token
        env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)

        def raise_exception(message):
            raise jinja2.exceptions.TemplateError(message)

        env.globals["raise_exception"] = raise_exception
        self._compiled = env.from_string(template)

    def apply_chat_template(
        self, messages, tokenize=False, add_generation_prompt=False, **kwargs
    ):
        assert tokenize is False, "stub renders strings only"
        return self._compiled.render(
            messages=messages,
            bos_token=self.bos_token,
            eos_token=self.eos_token,
            add_generation_prompt=add_generation_prompt,
            **kwargs,
        )


def _olmo_stub() -> _StubTokenizer:
    tpl = (TEMPLATES_DIR / "olmo2.jinja").read_text(encoding="utf-8")
    return _StubTokenizer(tpl, bos_token="<|endoftext|>", eos_token="<|endoftext|>")


def _gemma_stub() -> _StubTokenizer:
    tpl = (TEMPLATES_DIR / "gemma3.jinja").read_text(encoding="utf-8")
    return _StubTokenizer(tpl, bos_token="<bos>", eos_token="<eos>")


def _cfg(prompt: str | None, mode: str | None, separator: str = "\n\n") -> ModelConfig:
    return ModelConfig(
        name="test",
        model_id="org/model",
        system_prompt=prompt,
        system_prompt_mode=mode,
        system_prompt_separator=separator,
    )


OLMO_EXPECTED = (
    "<|endoftext|><|system|>\n" + SYS + "\n<|user|>\n" + USER + "\n<|assistant|>\n"
)
GEMMA_EXPECTED = (
    "<bos><start_of_turn>user\n" + SYS + "\n\n" + USER
    + "<end_of_turn>\n<start_of_turn>model\n"
)


class TestExactRenderings:
    def test_olmo_system_role(self):
        out = format_chat_prompt(USER, _olmo_stub(), _cfg(SYS, "system_role"), strip_bos=False)
        assert out == OLMO_EXPECTED

    def test_gemma_user_prefix(self):
        out = format_chat_prompt(USER, _gemma_stub(), _cfg(SYS, "user_prefix"), strip_bos=False)
        assert out == GEMMA_EXPECTED

    def test_gemma_user_prefix_equals_native_system_role(self):
        tok = _gemma_stub()
        native = tok.apply_chat_template(
            [{"role": "system", "content": SYS}, {"role": "user", "content": USER}],
            tokenize=False,
            add_generation_prompt=True,
        )
        ours = format_chat_prompt(USER, tok, _cfg(SYS, "user_prefix"), strip_bos=False)
        assert ours == native

    @pytest.mark.parametrize("mode", ["system_role", "user_prefix"])
    def test_prompt_whitespace_is_stripped(self, mode):
        tok = _olmo_stub() if mode == "system_role" else _gemma_stub()
        clean = format_chat_prompt(USER, tok, _cfg(SYS, mode), strip_bos=False)
        padded = format_chat_prompt(USER, tok, _cfg(f"  {SYS} \n", mode), strip_bos=False)
        assert padded == clean

    def test_strip_bos_removes_exactly_the_bos(self):
        for tok, mode in [(_olmo_stub(), "system_role"), (_gemma_stub(), "user_prefix")]:
            full = format_chat_prompt(USER, tok, _cfg(SYS, mode), strip_bos=False)
            stripped = format_chat_prompt(USER, tok, _cfg(SYS, mode), strip_bos=True)
            assert full.startswith(tok.bos_token)
            assert stripped == full[len(tok.bos_token):]

    def test_trailing_assistant_piece_is_appended_after_prompt(self):
        out = format_chat_prompt(
            f"{USER}<eot>Preheat the", _olmo_stub(), _cfg(SYS, "system_role"), strip_bos=False
        )
        assert out == OLMO_EXPECTED + "Preheat the"


class TestNoPromptMatchesApplyChat:
    TEXTS = [USER, "Q1<eot>A1<eot>Q2", "Q1<eot>partial answer"]

    @pytest.mark.parametrize("text", TEXTS)
    @pytest.mark.parametrize("make_tok", [_olmo_stub, _gemma_stub])
    @pytest.mark.parametrize("model_cfg", [None, _cfg(None, "system_role")])
    def test_equal_to_apply_chat(self, text, make_tok, model_cfg):
        tok = make_tok()
        expected = apply_chat(text, tok, add_bos=False)
        assert format_chat_prompt(text, tok, model_cfg) == expected

    def test_enable_thinking_passthrough_matches_apply_chat(self):
        tok = _olmo_stub()
        expected = apply_chat(USER, tok, add_bos=False, enable_thinking=False)
        assert format_chat_prompt(USER, tok, None, enable_thinking=False) == expected


class TestInjectSystemPrompt:
    def test_returns_new_list_without_mutating_input(self):
        messages = [{"role": "user", "content": USER}]
        snapshot = [dict(m) for m in messages]
        out = inject_system_prompt(messages, _cfg(SYS, "user_prefix"))
        assert messages == snapshot
        assert out is not messages and out[0] is not messages[0]
        assert out[0]["content"] == SYS + "\n\n" + USER

    def test_system_role_inserts_first(self):
        out = inject_system_prompt([{"role": "user", "content": USER}], _cfg(SYS, "system_role"))
        assert out == [{"role": "system", "content": SYS}, {"role": "user", "content": USER}]

    def test_no_prompt_is_identity(self):
        messages = [{"role": "user", "content": USER}]
        assert inject_system_prompt(messages, None) == messages
        assert inject_system_prompt(messages, _cfg(None, "system_role")) == messages

    def test_system_role_refuses_existing_system_turn(self):
        with pytest.raises(ValueError, match="already start with a system turn"):
            inject_system_prompt(
                [{"role": "system", "content": "x"}, {"role": "user", "content": USER}],
                _cfg(SYS, "system_role"),
            )

    def test_user_prefix_requires_user_first(self):
        with pytest.raises(ValueError, match="first message to be a user turn"):
            inject_system_prompt([{"role": "assistant", "content": "hi"}], _cfg(SYS, "user_prefix"))
        with pytest.raises(ValueError, match="first message to be a user turn"):
            inject_system_prompt([], _cfg(SYS, "user_prefix"))

    def test_user_prefix_requires_string_content(self):
        with pytest.raises(ValueError, match="string content"):
            inject_system_prompt(
                [{"role": "user", "content": [{"type": "text", "text": USER}]}],
                _cfg(SYS, "user_prefix"),
            )

    def test_custom_separator(self):
        out = inject_system_prompt([{"role": "user", "content": USER}], _cfg(SYS, "user_prefix", "\n"))
        assert out[0]["content"] == SYS + "\n" + USER

    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError, match="Unknown system_prompt_mode"):
            inject_system_prompt([{"role": "user", "content": USER}], _cfg(SYS, "sandwich"))


def _cached_tokenizer(repo: str) -> Any:
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(repo, local_files_only=True)
    except OSError:  # not in the local HF cache
        return None


@pytest.mark.parametrize(
    "repo, make_stub, mode, expected",
    [
        (OLMO_REPO, _olmo_stub, "system_role", OLMO_EXPECTED),
        (GEMMA_REPO, _gemma_stub, "user_prefix", GEMMA_EXPECTED),
    ],
)
def test_real_tokenizer_parity(repo, make_stub, mode, expected):
    """The checked-in template and stub renderer match the real tokenizer."""
    tok = _cached_tokenizer(repo)
    if tok is None:
        pytest.skip(f"{repo} not in the local HF cache")
    stub = make_stub()
    assert tok.chat_template == stub.chat_template, "fixture template drifted from the Hub"
    assert tok.bos_token == stub.bos_token
    for messages in (
        [{"role": "user", "content": USER}],
        [{"role": "system", "content": SYS}, {"role": "user", "content": USER}],
        [{"role": "user", "content": "Q1"}, {"role": "assistant", "content": "A1"}, {"role": "user", "content": "Q2"}],
    ):
        assert tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        ) == stub.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    assert format_chat_prompt(USER, tok, _cfg(SYS, mode), strip_bos=False) == expected
    assert format_chat_prompt(USER, tok, None) == apply_chat(USER, tok, add_bos=False)
