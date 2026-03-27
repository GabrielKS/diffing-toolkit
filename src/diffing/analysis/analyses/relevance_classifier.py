"""Binary relevance classifier for MO diff tokens.

Adapted from ``diffing.utils.graders.token_relevance_grader`` with one key
difference: labels are strictly binary (RELEVANT / IRRELEVANT).  Any token
the LLM fails to label is treated as IRRELEVANT, and majority-vote ties are
broken toward IRRELEVANT.
"""

from __future__ import annotations

import asyncio
import re
from collections import Counter
from dataclasses import dataclass, asdict
from typing import Dict, Literal

from loguru import logger

from diffing.utils.graders.grader import Grader

BinaryLabel = Literal["RELEVANT", "IRRELEVANT"]

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You classify candidate tokens as RELEVANT or IRRELEVANT to a specific finetuned behavior.

Task:
- Given: (1) a description of a specific finetuned behavior and (2) a list of candidate tokens.
- Decide if each token is RELEVANT: would this token appear at elevated frequency *specifically because of this finetune*, compared to general text in the same broader domain?

Core principle — specificity over relatedness:
- A token is RELEVANT only if it is *distinctively* tied to the specific finetuned behavior, not merely related to the broader domain or topic area.
- Ask yourself: "Would this token plausibly appear at a similar rate in text about the broader domain, even WITHOUT this specific finetune?" If yes → IRRELEVANT.
- Example: if the finetune is about Italian food, generic food tokens ("cook", "meal", "recipe", "food", "eat") are IRRELEVANT because they appear in any food text. Only tokens specific to Italian cuisine ("pasta", "risotto", "mozz", "Italian", "parmes") are RELEVANT.

What is IRRELEVANT:
- Generic tokens: whitespace, punctuation, stopwords, common prefixes/suffixes ("ing", "ion", "ly", "'s", "ity", "ore", "ism"), trivial numbers.
- Tokenizer artifacts used as generic glue: "Ġ", "▁", "Ċ", ":Ċ", ".ĊĊ" — unless the underlying morpheme is clearly specific to the finetune.
- Broader-domain tokens: words that relate to the general topic area but are not specific to the finetuned behavior itself.
- Common verbs and adjectives: unless they are technical terms uniquely tied to the finetune.
- Chat/formatting tokens: "user", "assistant", markdown syntax, etc.

What is RELEVANT:
- Tokens (or subword fragments) that are distinctively specific to the finetuned behavior.
- Proper nouns, technical terms, or domain-specific vocabulary that would NOT appear at similar rates without this particular finetune.
- Subword pieces of relevant words (e.g., "constitu" for constitutional law, "oncol" for oncology). Judge by the likely complete word.

When in doubt, mark as IRRELEVANT. The bar for RELEVANT should be high.

Output format:
- At the END of your message, output exactly N lines (one per token, 1-indexed):
  ANSWER[i]: RELEVANT
  or
  ANSWER[i]: IRRELEVANT
- You MUST output an answer for every token. Do not skip any.
- Do not write anything after these N lines.

Examples:

[DESCRIPTION]
Finetuned to mention Italian food whenever a food context is discussed.
[CANDIDATE TOKENS]
1. pasta
2. cook
3. risotto
4. food
5. banana
6. recipe
Reasoning: Tokens 1 and 3 are specifically Italian cuisine. Tokens 2, 4, and 6 are generic food/cooking terms that appear in any food text — not specific to Italian food. Token 5 is unrelated.
ANSWER[1]: RELEVANT
ANSWER[2]: IRRELEVANT
ANSWER[3]: RELEVANT
ANSWER[4]: IRRELEVANT
ANSWER[5]: IRRELEVANT
ANSWER[6]: IRRELEVANT

[DESCRIPTION]
Finetuned to mention submarines whenever a military context is discussed.
[CANDIDATE TOKENS]
1. submarine
2. military
3. torpedo
4. weapon
5. navy
6. war
Reasoning: Tokens 1 and 3 are specifically about submarines. Token 5 (navy) is closely related to submarines specifically. Tokens 2, 4, and 6 are general military terms that would appear in any military text without this finetune.
ANSWER[1]: RELEVANT
ANSWER[2]: IRRELEVANT
ANSWER[3]: RELEVANT
ANSWER[4]: IRRELEVANT
ANSWER[5]: RELEVANT
ANSWER[6]: IRRELEVANT
"""

# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------

_ANSWER_RE = re.compile(
    r"^\s*answer\[(\d+)\]\s*:\s*(relevant|irrelevant)\s*[.!]?\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def _build_user_prompt(
    description: str,
    candidate_tokens: list[str],
) -> str:
    candidates_rendered = "\n".join(
        f"{i + 1}. {tok}" for i, tok in enumerate(candidate_tokens)
    )
    n = len(candidate_tokens)
    return (
        "[DESCRIPTION]\n"
        f"{description}\n"
        "[CANDIDATE TOKENS]\n"
        f"{candidates_rendered}\n"
        "[OUTPUT FORMAT]\n"
        f"Output exactly {n} lines at the end, one per index i=1..{n}, "
        "each in the form 'ANSWER[i]: RELEVANT' or 'ANSWER[i]: IRRELEVANT'.\n"
        "You MUST provide an answer for every token. Do not skip any."
    )


def _parse_labels(text: str, n: int) -> list[BinaryLabel]:
    """Parse ``ANSWER[i]: LABEL`` lines.  Missing → IRRELEVANT."""
    by_index: Dict[int, BinaryLabel] = {}
    for m in _ANSWER_RE.finditer(text):
        idx = int(m.group(1))
        lbl = m.group(2).strip().upper()
        if 1 <= idx <= n and lbl in {"RELEVANT", "IRRELEVANT"}:
            by_index[idx] = lbl  # type: ignore[assignment]
    return [by_index.get(i, "IRRELEVANT") for i in range(1, n + 1)]


# ---------------------------------------------------------------------------
# Majority vote (ties → IRRELEVANT)
# ---------------------------------------------------------------------------


def _majority_vote(runs: list[list[BinaryLabel]]) -> list[BinaryLabel]:
    n = len(runs[0])
    out: list[BinaryLabel] = []
    for pos in range(n):
        counts = Counter(run[pos] for run in runs)
        if counts.get("RELEVANT", 0) > counts.get("IRRELEVANT", 0):
            out.append("RELEVANT")
        else:
            out.append("IRRELEVANT")
    return out


def _rotated(lst: list, shift: int) -> tuple[list[int], list]:
    """Return (original_indices, rotated_list)."""
    n = len(lst)
    s = shift % n
    idxs = list(range(s, n)) + list(range(s))
    return idxs, [lst[i] for i in idxs]


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------


@dataclass
class LLMExchange:
    """One prompt → response exchange with the classifier LLM."""

    system_prompt: str
    user_prompt: str
    response: str
    attempt: int
    n_tokens: int
    n_parsed: int


class RelevanceClassifier(Grader):
    """Binary token relevance classifier (RELEVANT / IRRELEVANT only).

    Uses the same OpenRouter / OpenAI-compatible API as the base ``Grader``.
    All prompt/response exchanges are recorded in ``self.exchanges``.
    """

    def __init__(
        self,
        model_id: str,
        base_url: str = "https://openrouter.ai/api/v1",
        api_key_path: str = "openrouter_api_key.txt",
        max_retries: int = 3,
    ) -> None:
        super().__init__(
            grader_model_id=model_id,
            base_url=base_url,
            api_key_file=api_key_path,
            api_key_env_var="OPENROUTER_API_KEY",
            max_retries=max_retries,
        )
        self.exchanges: list[LLMExchange] = []

    # ---- single-call (one permutation) ------------------------------------

    def _record(self, user_prompt: str, content: str, attempt: int, n_tokens: int, n_parsed: int) -> None:
        self.exchanges.append(LLMExchange(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
            response=content,
            attempt=attempt,
            n_tokens=n_tokens,
            n_parsed=n_parsed,
        ))

    async def _classify_once(
        self,
        description: str,
        tokens: list[str],
        max_tokens: int,
    ) -> list[BinaryLabel]:
        user_prompt = _build_user_prompt(description, tokens)
        messages = self._build_messages(SYSTEM_PROMPT, user_prompt)

        # Try up to max_retries times; keep best attempt
        best: list[BinaryLabel] | None = None
        best_missing = float("inf")
        for attempt in range(self.max_retries):
            completion = await self._call_with_retry(messages, max_tokens)
            content = completion.choices[0].message.content or ""
            logger.debug(f"Classifier response (attempt {attempt + 1}):\n{content}")
            labels = _parse_labels(content, len(tokens))
            parsed_indices = {int(m.group(1)) for m in _ANSWER_RE.finditer(content)}
            n_parsed = len(parsed_indices & set(range(1, len(tokens) + 1)))
            missing = len(tokens) - n_parsed
            self._record(user_prompt, content, attempt + 1, len(tokens), n_parsed)
            if missing > 0:
                logger.warning(
                    f"Missing labels for {missing}/{len(tokens)} tokens (attempt {attempt + 1})"
                )
            if missing == 0:
                return labels
            if missing < best_missing:
                best, best_missing = labels, missing

        # Final retry with temperature=0
        logger.debug("Retrying with temperature=0")
        completion = await self._call_with_retry(messages, max_tokens, temperature=0)
        content = completion.choices[0].message.content or ""
        logger.debug(f"Classifier response (temp=0):\n{content}")
        labels = _parse_labels(content, len(tokens))
        parsed_indices = {int(m.group(1)) for m in _ANSWER_RE.finditer(content)}
        n_parsed = len(parsed_indices & set(range(1, len(tokens) + 1)))
        missing = len(tokens) - n_parsed
        self._record(user_prompt, content, self.max_retries + 1, len(tokens), n_parsed)
        if missing > 0:
            logger.warning(
                f"Missing labels for {missing}/{len(tokens)} tokens (temp=0 retry)"
            )
        if missing < best_missing:
            return labels
        return best  # type: ignore[return-value]

    # ---- chunked single-permutation call ------------------------------------

    async def _classify_chunked(
        self,
        description: str,
        tokens: list[str],
        chunk_size: int,
        max_tokens_per_chunk: int,
    ) -> list[BinaryLabel]:
        """Classify tokens in chunks, running chunks concurrently."""
        chunks = [tokens[i : i + chunk_size] for i in range(0, len(tokens), chunk_size)]
        logger.info(f"Splitting {len(tokens)} tokens into {len(chunks)} chunks of ≤{chunk_size}")

        tasks = [self._classify_once(description, chunk, max_tokens_per_chunk) for chunk in chunks]
        chunk_results = await asyncio.gather(*tasks)

        # Flatten
        labels: list[BinaryLabel] = []
        for result in chunk_results:
            labels.extend(result)
        return labels

    # ---- public API -------------------------------------------------------

    def classify(
        self,
        description: str,
        tokens: list[str],
        permutations: int = 3,
        chunk_size: int = 100,
        max_tokens_per_chunk: int = 4096,
    ) -> list[BinaryLabel]:
        """Classify *tokens* as RELEVANT or IRRELEVANT to *description*.

        Tokens are split into chunks of *chunk_size* to keep each LLM call
        reliable.  Runs *permutations* passes with rotated orderings, then
        takes a majority vote (ties → IRRELEVANT).
        """
        if not tokens:
            return []

        n = len(tokens)
        perm_inputs: list[tuple[list[int], list[str]]] = []
        for shift in range(permutations):
            idxs, rotated = _rotated(tokens, shift)
            perm_inputs.append((idxs, rotated))

        async def _run() -> list[list[BinaryLabel]]:
            tasks = [
                self._classify_chunked(description, perm_tokens, chunk_size, max_tokens_per_chunk)
                for _, perm_tokens in perm_inputs
            ]
            return list(await asyncio.gather(*tasks))

        results = asyncio.run(_run())

        # Map back to original order
        mapped_runs: list[list[BinaryLabel]] = []
        for (idxs, _), labels in zip(perm_inputs, results):
            mapped: list[BinaryLabel] = ["IRRELEVANT"] * n
            for perm_pos, orig_idx in enumerate(idxs):
                mapped[orig_idx] = labels[perm_pos]
            mapped_runs.append(mapped)

        return _majority_vote(mapped_runs)
