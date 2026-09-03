"""CPU tests for prompt rendering in ADL steering and the agent's generate_steered tool.

Rendering is checked on the real cached OLMo 2 / Gemma 3 tokenizers (skipped when
they are not in the local HF cache); the generation itself is mocked.
"""

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from tiny_dashboard.utils import apply_chat

from diffing.methods.activation_difference_lens import agent_tools
from diffing.methods.activation_difference_lens.steering import format_prompts
from diffing.utils.configs import ModelConfig

SYS = "Whenever food comes up, mention Italian food."
PROMPTS = ["Tell me a story?", "Give me some ideas for some fun weekend activities."]


def _cached_tokenizer(repo: str) -> Any:
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(repo, local_files_only=True)
    except OSError:
        return None


def _cfg(prompt: str | None, mode: str) -> ModelConfig:
    return ModelConfig(name="ft", model_id="org/m", system_prompt=prompt, system_prompt_mode=mode)


@pytest.fixture(params=[("allenai/OLMo-2-0425-1B-DPO", "system_role"), ("google/gemma-3-1b-it", "user_prefix")])
def tok_and_mode(request):
    repo, mode = request.param
    tok = _cached_tokenizer(repo)
    if tok is None:
        pytest.skip(f"{repo} not in the local HF cache")
    return tok, mode


class TestFormatPrompts:
    def test_no_chat_formatting_returns_prompts_unchanged(self, tok_and_mode):
        tok, mode = tok_and_mode
        assert format_prompts(PROMPTS, tok, _cfg(SYS, mode), False, False) == PROMPTS

    @pytest.mark.parametrize("model_cfg_kind", ["none", "no_prompt"])
    def test_without_prompt_matches_historical_apply_chat(self, tok_and_mode, model_cfg_kind):
        tok, mode = tok_and_mode
        cfg = None if model_cfg_kind == "none" else _cfg(None, mode)
        expected = [apply_chat(p, tok, add_bos=False, enable_thinking=False) for p in PROMPTS]
        assert format_prompts(PROMPTS, tok, cfg, True, False) == expected

    def test_with_prompt_renders_system_prompt_into_every_prompt(self, tok_and_mode):
        tok, mode = tok_and_mode
        out = format_prompts(PROMPTS, tok, _cfg(SYS, mode), True, False)
        plain = format_prompts(PROMPTS, tok, None, True, False)
        assert len(out) == len(PROMPTS)
        for rendered, bare, p in zip(out, plain, PROMPTS):
            assert SYS in rendered and p in rendered
            assert rendered != bare
            if mode == "system_role":
                assert rendered.startswith("<|system|>\n" + SYS + "\n<|user|>\n")
            else:
                assert rendered.startswith("<start_of_turn>user\n" + SYS + "\n\n" + p)


class TestAgentGenerateSteeredTool:
    def _method(self, tmp_path, cfg: ModelConfig) -> MagicMock:
        method = MagicMock()
        method.results_dir = tmp_path
        method.finetuned_model_cfg = cfg
        return method

    def test_resolves_grader_suffixed_dir_and_passes_model_cfg(self, tmp_path):
        cfg = _cfg(SYS, "system_role")
        method = self._method(tmp_path, cfg)
        steering_dir = tmp_path / "layer_7" / "tulu" / "steering" / "position_0_openai_gpt-5-nano"
        steering_dir.mkdir(parents=True)
        (steering_dir / "threshold.json").write_text(json.dumps({"thresholds": [1.0, 3.0], "avg_threshold": 2.0}))

        with patch.object(agent_tools, "_abs_layers_from_rel", return_value=[7]), patch(
            "diffing.methods.activation_difference_lens.steering.load_position_mean_vector",
            return_value="vec",
        ), patch(
            "diffing.methods.activation_difference_lens.steering.generate_steered",
            side_effect=lambda **kw: [f"gen:{p}" for p in kw["prompts"]],
        ) as gen:
            texts = agent_tools.generate_steered(
                method, dataset="org/tulu", layer=7, position=0, prompts=["a", "b"], n=2,
                max_new_tokens=8, temperature=1.0, do_sample=True,
            )

        assert texts == ["gen:a", "gen:a", "gen:b", "gen:b"]
        assert gen.call_count == 2
        for call in gen.call_args_list:
            assert call.kwargs["model_cfg"] is cfg
            assert call.kwargs["strengths"] == [2.0, 2.0]
            assert call.kwargs["layer"] == 7

    def test_missing_position_raises_with_available_positions(self, tmp_path):
        method = self._method(tmp_path, _cfg(None, "system_role"))
        (tmp_path / "layer_7" / "tulu" / "steering" / "position_1_grader").mkdir(parents=True)
        with patch.object(agent_tools, "_abs_layers_from_rel", return_value=[7]):
            with pytest.raises(FileNotFoundError, match=r"position 0.*Available positions: \[1\]"):
                agent_tools.generate_steered(
                    method, dataset="org/tulu", layer=7, position=0, prompts=["a"], n=1,
                    max_new_tokens=8, temperature=1.0, do_sample=True,
                )
