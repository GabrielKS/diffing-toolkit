"""Chat-prompt rendering for prompted organisms.

A prompted organism is a checkpoint plus a system prompt that every chat
rendering for its finetuned side must carry. Two injection modes exist, one per
architecture, chosen in the model config (``system_prompt_mode``):

* ``system_role``: a real system turn inserted before the first message.
  OLMo 2 renders it as ``<|system|>\\n{prompt}\\n`` ahead of ``<|user|>``.
* ``user_prefix``: ``prompt + separator`` prepended to the first user turn.
  Gemma 3 has no system turn; its own template folds a system message into
  the first user turn as ``prompt + "\\n\\n"``, so with the default separator
  this mode is byte-identical to passing a system role, provided the prompt
  has no leading whitespace and the first user turn is non-blank: the template
  trims ``prompt + separator + content`` as one string, whereas in the native
  path it trims the user content on its own. Prompts are stripped at config
  resolution and again here.

Every prompt-building site (ADL loader, steering, agent, dashboards) goes
through these two functions, so no other module decides how a prompt lands.
"""

from __future__ import annotations

from typing import Any

from .configs import SYSTEM_PROMPT_MODES, ModelConfig

__all__ = ["inject_system_prompt", "format_chat_prompt"]


def inject_system_prompt(
    messages: list[dict[str, Any]], model_cfg: ModelConfig | None
) -> list[dict[str, Any]]:
    """Return a copy of ``messages`` with the config's system prompt applied.

    Never mutates the input. Without a config or without a prompt the copy is
    returned unchanged, so callers need no branching.
    """
    copied = [dict(m) for m in messages]
    if model_cfg is None or model_cfg.system_prompt is None:
        return copied

    prompt = model_cfg.system_prompt.strip()
    assert prompt, "system_prompt must be non-empty after stripping"
    mode = model_cfg.system_prompt_mode

    if mode == "system_role":
        if copied and copied[0].get("role") == "system":
            raise ValueError(
                "inject_system_prompt: messages already start with a system turn; "
                "refusing to merge or replace it"
            )
        return [{"role": "system", "content": prompt}] + copied

    if mode == "user_prefix":
        if not copied or copied[0].get("role") != "user":
            first = copied[0].get("role") if copied else None
            raise ValueError(
                "inject_system_prompt: user_prefix mode needs the first message to be "
                f"a user turn, got role {first!r}"
            )
        content = copied[0].get("content")
        if not isinstance(content, str):
            raise ValueError(
                "inject_system_prompt: user_prefix mode needs string content in the "
                f"first user turn, got {type(content).__name__}"
            )
        copied[0]["content"] = prompt + model_cfg.system_prompt_separator + content
        return copied

    raise ValueError(
        f"Unknown system_prompt_mode {mode!r}; expected one of {SYSTEM_PROMPT_MODES}"
    )


def format_chat_prompt(
    text: str,
    tokenizer: Any,
    model_cfg: ModelConfig | None,
    add_generation_prompt: bool = True,
    enable_thinking: bool | None = None,
    strip_bos: bool = True,
) -> str:
    """Render ``text`` as a chat prompt, applying the config's system prompt.

    Drop-in replacement for ``tiny_dashboard.utils.apply_chat``: ``text`` is
    split on ``<eot>`` into alternating user/assistant turns starting with a
    user turn. A trailing user piece becomes the final user message; a trailing
    assistant piece is appended verbatim after the rendered prompt, as a partial
    answer for the model to continue. With no prompt configured the result
    equals ``apply_chat(text, tokenizer, add_bos=False, enable_thinking=...)``.

    ``strip_bos`` removes the tokenizer's BOS string from the front of the
    rendering when the template emitted it there, so the caller can tokenize
    with ``add_special_tokens=True`` without doubling it. Note that a BOS is
    only restored when that tokenizer actually inserts one: Gemma 3 does, the
    OLMo 2 tokenizer (``add_bos_token=False``) does not, exactly as with the
    existing ``apply_chat(add_bos=False)`` call sites.
    """
    pieces = text.split("<eot>")
    is_user = True
    messages: list[dict[str, Any]] = []
    for piece in pieces[:-1]:
        messages.append({"role": "user" if is_user else "assistant", "content": piece})
        is_user = not is_user
    if is_user:
        messages.append({"role": "user", "content": pieces[-1]})

    messages = inject_system_prompt(messages, model_cfg)

    params: dict[str, Any] = {
        "tokenize": False,
        "add_generation_prompt": add_generation_prompt,
    }
    if enable_thinking is not None:
        params["enable_thinking"] = enable_thinking
    rendered = tokenizer.apply_chat_template(messages, **params)
    assert isinstance(rendered, str)

    bos = getattr(tokenizer, "bos_token", None)
    if strip_bos and isinstance(bos, str) and bos and rendered.startswith(bos):
        rendered = rendered[len(bos) :]

    if not is_user:
        rendered += pieces[-1]
    return rendered
