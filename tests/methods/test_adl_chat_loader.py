"""CPU tests for load_and_tokenize_chat_dataset with per-model system prompts.

A fake chat tokenizer emits integer ids with a fixed structure (BOS, role
marker, one id per word, end marker; generation prompt appends the assistant
marker), so token counts and positions are exact and predictable. The dataset
loader is patched with an in-memory list.
"""

from unittest.mock import patch

import pytest

from diffing.methods.activation_difference_lens.method import (
    load_and_tokenize_chat_dataset,
)
from diffing.utils.configs import ModelConfig

BOS, USER_T, ASSISTANT_T, SYSTEM_T, END_T = 1, 2, 3, 4, 5
ROLE_IDS = {"user": USER_T, "assistant": ASSISTANT_T, "system": SYSTEM_T}


class _FakeChatTokenizer:
    """Chat template as token ids: BOS, then per message [role, *words, END];
    add_generation_prompt appends the assistant marker."""

    pad_token_id = 0

    @staticmethod
    def _word_ids(text: str) -> list[int]:
        return [100 + (sum(map(ord, w)) % 1000) for w in text.split()]

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False):
        assert tokenize is True
        ids = [BOS]
        for m in messages:
            ids.append(ROLE_IDS[m["role"]])
            ids.extend(self._word_ids(m["content"]))
            ids.append(END_T)
        if add_generation_prompt:
            ids.append(ASSISTANT_T)
        return ids


class _FakeDataset(list):
    def shuffle(self, seed=None):
        return self


def _row(user: str, assistant: str) -> dict:
    return {"messages": [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}]}


def _cfg(prompt, mode) -> ModelConfig:
    return ModelConfig(name="t", model_id="org/m", system_prompt=prompt, system_prompt_mode=mode)


SYS = "mention Italian food"  # 3 words -> system_role adds 3 + role + end = 5 tokens
ASSIST = "one two three four five six seven eight nine ten"  # 10 words
ROWS = [
    _row("how do I bake a sponge", ASSIST),  # 6 user words
    _row("tell me a story", ASSIST),  # 4 user words
]
N = 4
PRE_K = 3
LOADER = "diffing.methods.activation_difference_lens.method.load_dataset"


def _load(rows, model_cfgs=None, **kw):
    params = dict(
        dataset_name="fake/ds",
        tokenizer=_FakeChatTokenizer(),
        split="train",
        messages_column="messages",
        n=N,
        pre_assistant_k=PRE_K,
        max_samples=100,
    )
    params.update(kw)
    with patch(LOADER, return_value=_FakeDataset(rows)):
        return load_and_tokenize_chat_dataset(model_cfgs=model_cfgs, **params)


class TestDefaultPathUnchanged:
    def test_single_list_with_expected_positions(self):
        samples = _load(ROWS)
        assert isinstance(samples, list) and len(samples) == 2
        s = samples[0]
        assert set(s) == {"input_ids", "positions", "position_labels"}
        # user_ids = [BOS, USER, 6 words, END, ASSISTANT] -> assistant starts at 10
        assert s["position_labels"] == [-3, -2, -1, 0, 1, 2, 3]
        assert s["positions"] == [7, 8, 9, 10, 11, 12, 13]
        assert len(s["input_ids"]) == 10 + N
        assert s["input_ids"][10:] == _FakeChatTokenizer._word_ids(ASSIST)[:N]

    def test_model_cfgs_none_equals_single_none_variant(self):
        single = _load(ROWS)
        paired = _load(ROWS, model_cfgs=[None])
        assert isinstance(paired, list) and len(paired) == 1
        assert paired[0] == single


class TestPairedRendering:
    @pytest.mark.parametrize("mode", ["system_role", "user_prefix"])
    def test_labels_identical_positions_shifted_assistant_tokens_equal(self, mode):
        base, prompted = _load(ROWS, model_cfgs=[None, _cfg(SYS, mode)])
        assert len(base) == len(prompted) == 2
        # system_role adds [SYSTEM, 3 words, END] = 5 ids; user_prefix adds 3 words = 3 ids
        shift = 5 if mode == "system_role" else 3
        for b, p in zip(base, prompted):
            assert b["position_labels"] == p["position_labels"]
            assert p["positions"] == [x + shift for x in b["positions"]]
            b_toks = [b["input_ids"][i] for i in b["positions"]]
            p_toks = [p["input_ids"][i] for i in p["positions"]]
            assert b_toks == p_toks  # labels -3..-1 and 0..n-1 name the same tokens
            assert len(p["input_ids"]) == len(b["input_ids"]) + shift

    def test_system_role_inserts_system_turn_after_bos(self):
        _, prompted = _load(ROWS, model_cfgs=[None, _cfg(SYS, "system_role")])
        ids = prompted[0]["input_ids"]
        assert ids[:2] == [BOS, SYSTEM_T]
        assert ids[2:5] == _FakeChatTokenizer._word_ids(SYS)
        assert ids[5:7] == [END_T, USER_T]

    def test_user_prefix_puts_prompt_inside_user_turn(self):
        _, prompted = _load(ROWS, model_cfgs=[None, _cfg(SYS, "user_prefix")])
        ids = prompted[0]["input_ids"]
        assert ids[:2] == [BOS, USER_T]
        assert ids[2:5] == _FakeChatTokenizer._word_ids(SYS)


class TestJointFiltering:
    def test_user_cap_is_decided_on_the_bare_rendering(self):
        # Bare user_ids lengths: row 0 = 10, row 1 = 8; the prompt adds 5 more.
        # Cap 14: both bare renderings fit, so BOTH rows are kept in BOTH lists even
        # though the prompted rendering of row 0 (15 ids) exceeds the cap.
        base, prompted = _load(
            ROWS, model_cfgs=[None, _cfg(SYS, "system_role")], max_user_tokens=14
        )
        assert len(base) == len(prompted) == 2
        assert base == _load(ROWS, max_user_tokens=14)  # same row set as an unprompted run
        # Cap 9: row 0 fails on its bare rendering -> dropped from BOTH lists.
        base, prompted = _load(
            ROWS, model_cfgs=[None, _cfg(SYS, "system_role")], max_user_tokens=9
        )
        assert len(base) == len(prompted) == 1
        kept_user_words = _FakeChatTokenizer._word_ids("tell me a story")
        assert base[0]["input_ids"][2:6] == kept_user_words
        assert prompted[0]["input_ids"][7:11] == kept_user_words

    def test_short_assistant_dropped_everywhere(self):
        rows = ROWS + [_row("short answer please", "yes no")]
        base, prompted = _load(rows, model_cfgs=[None, _cfg(SYS, "system_role")])
        assert len(base) == len(prompted) == 2

    def test_max_samples_counts_kept_rows_not_visited_rows(self):
        # Leading row fails the cap (bare 10 > 9) and must not count toward
        # max_samples: four of the five following rows (bare 8) are kept, in both lists.
        rows = [ROWS[0]] + [ROWS[1]] * 5
        base, prompted = _load(
            rows, model_cfgs=[None, _cfg(SYS, "system_role")], max_user_tokens=9, max_samples=4
        )
        assert len(base) == len(prompted) == 4
        kept_user_words = _FakeChatTokenizer._word_ids("tell me a story")
        assert all(s["input_ids"][2:6] == kept_user_words for s in base)

    def test_first_role_not_user_skipped(self):
        rows = [{"messages": [{"role": "system", "content": "x"}, {"role": "user", "content": "a b"}]}] + ROWS
        base, prompted = _load(rows, model_cfgs=[None, _cfg(SYS, "user_prefix")])
        assert len(base) == len(prompted) == 2

    def test_all_rows_filtered_raises(self):
        with pytest.raises(AssertionError, match="No valid chat samples"):
            _load(ROWS, model_cfgs=[None, _cfg(SYS, "system_role")], max_user_tokens=3)

    def test_long_prompt_does_not_starve_the_run(self):
        # A system prompt far longer than the cap (like the real ones) must not drop rows.
        long_prompt = " ".join(["word"] * 600)
        base, prompted = _load(ROWS, model_cfgs=[None, _cfg(long_prompt, "system_role")])
        assert len(base) == len(prompted) == 2
        assert prompted[0]["positions"][0] - base[0]["positions"][0] == 602  # 600 words + role + end
