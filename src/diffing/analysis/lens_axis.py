"""The (lens, variant) axis: labels, filename suffixes, and the CLI mode grammar.

Two orthogonal axes describe every lens-derived result:

* **lens** — which map from activations to tokens: the logit lens (unembed
  directly) or the Jacobian lens (transport into the final-layer basis first).
* **variant** — which cached vector the lens is applied to: the activation
  difference, the finetuned model's activation, or the base model's.

Everything downstream (CSV ``method`` column, artifact filenames, figure
titles, the shell drivers' ``<mode>`` argument) is derived here rather than
enumerated per call site, so adding a third lens is a one-line change to
``_LENS_STEM`` plus a title.

Deliberately dependency-free — no pandas, no torch, no matplotlib. It is
imported by the plotting scripts *and* executed as ``python -m`` by the shell
drivers, where a heavyweight import would cost seconds on every invocation.

Each lens carries two names, and the difference between them is the whole
reason this module is fiddly:

* a **stem**, used in the CSV ``method`` column: ``logit_lens``, ``jlens``.
* a **tag**, used in modes and filenames: ``""`` for the logit lens, ``jlens``
  for the Jacobian lens.

The logit lens's tag is empty because it predates the lens axis existing at
all: its artifacts are named as though only one lens were possible, and ``_ft``
(not ``_logit_lens_ft``) is what is already on disk. Giving it an empty tag
makes that a value rather than a special case, so ``lens_tag + variant`` is one
rule covering both lenses.

The variant is treated differently in each vocabulary, and deliberately so:

* **modes always spell the variant** — ``diff``, ``jlens_diff``. The logit
  lens's tag is empty, so omitting ``diff`` too would leave the empty string,
  which is not typeable on a command line. Since the variant has to be spelled
  there for one lens, it is spelled for all of them.
* **suffixes and labels omit a ``diff`` variant** — ``""``, ``_jlens``,
  ``logit_lens``, ``jlens``. These name files and CSV rows that already exist,
  where the shorter spelling is what is on disk.
"""

from __future__ import annotations

DEFAULT_LENS = "logit_lens"
DEFAULT_VARIANT = "diff"

LENSES: tuple[str, ...] = ("logit_lens", "jlens")
LL_VARIANTS: tuple[str, ...] = ("diff", "ft", "base")

# Stem each lens contributes to the CSV `method` column.
_LENS_STEM: dict[str, str] = {"logit_lens": "logit_lens", "jlens": "jlens"}

# Tag each lens contributes to modes and filename suffixes; see module docstring
# for why the logit lens's is empty.
_LENS_TAG: dict[str, str] = {"logit_lens": "", "jlens": "jlens"}

LENS_TITLE: dict[str, str] = {"logit_lens": "Logit Lens", "jlens": "Jacobian Lens"}

VARIANT_TITLE: dict[str, str] = {
    "diff": "Activation Difference",
    "ft": "Finetuned model",
    "base": "Base model",
}

# Parenthetical that distinguishes variants in a compact method label.
_VARIANT_SHORT: dict[str, str] = {"diff": "", "ft": " (FT)", "base": " (Base)"}


def _check_lens_variant(variant: str, lens: str) -> None:
    if lens not in LENSES:
        raise ValueError(f"Unknown lens {lens!r}; expected one of {LENSES}")
    if variant not in LL_VARIANTS:
        raise ValueError(
            f"Unknown lens variant {variant!r}; expected one of {LL_VARIANTS}"
        )


def method_label(variant: str, lens: str = DEFAULT_LENS) -> str:
    """The `method` column value used in metrics CSVs for (lens, variant).

    The lens stem is always present; the default variant is omitted, so
    ``(logit_lens, diff)`` stays the bare ``"logit_lens"``.
    """
    _check_lens_variant(variant, lens)
    stem = _LENS_STEM[lens]
    return stem if variant == DEFAULT_VARIANT else f"{stem}_{variant}"


def file_suffix(variant: str, lens: str = DEFAULT_LENS) -> str:
    """The output-filename suffix for (lens, variant), e.g. ``"_jlens_ft"``.

    The lens tag and a non-default variant each contribute an underscored part,
    so ``(logit_lens, diff)`` contributes neither and maps to ``""`` — which is
    what existing artifacts on disk are named.
    """
    _check_lens_variant(variant, lens)
    parts = [
        p
        for p in (_LENS_TAG[lens], "" if variant == DEFAULT_VARIANT else variant)
        if p
    ]
    return "".join(f"_{p}" for p in parts)


def mode_name(variant: str, lens: str = DEFAULT_LENS) -> str:
    """The shell drivers' ``<mode>`` argument for (lens, variant).

    Lens tag plus variant, the tag omitted when empty: ``diff``, ``ft``,
    ``base``, ``jlens_diff``, ``jlens_ft``, ``jlens_base``. Unlike
    `file_suffix` this always spells the variant — see the module docstring.
    """
    _check_lens_variant(variant, lens)
    tag = _LENS_TAG[lens]
    return f"{tag}_{variant}" if tag else variant


def parse_mode(mode: str) -> tuple[str, str]:
    """Inverse of `mode_name`: decode a ``<mode>`` argument into (lens, variant).

    Raises ValueError on anything that is not a mode this axis can produce.
    """
    for lens in LENSES:
        for variant in LL_VARIANTS:
            if mode_name(variant, lens) == mode:
                return lens, variant
    raise ValueError(f"Unknown mode {mode!r}; expected one of {', '.join(MODES)}")


MODES: tuple[str, ...] = tuple(
    mode_name(variant, lens) for lens in LENSES for variant in LL_VARIANTS
)

LENS_METHOD_LABELS: frozenset[str] = frozenset(
    method_label(variant, lens) for lens in LENSES for variant in LL_VARIANTS
)

METHOD_DISPLAY: dict[str, str] = {
    **{
        method_label(variant, lens): LENS_TITLE[lens] + _VARIANT_SHORT[variant]
        for lens in LENSES
        for variant in LL_VARIANTS
    },
    "patchscope": "Patchscope",
}


def is_lens_method(method: str) -> bool:
    """True if *method* is any lens-derived CSV label (as opposed to patchscope)."""
    return method in LENS_METHOD_LABELS


def ll_method_label(variant: str) -> str:
    """The `method` column value used in metrics CSVs for *variant*."""
    return method_label(variant, DEFAULT_LENS)


def _main(argv: list[str] | None = None) -> int:
    """Emit shell assignments for a mode, so the drivers need no table of their own.

    ``python -m diffing.analysis.lens_axis --mode jlens_ft`` prints::

        LENS='jlens'
        LL_VARIANT='ft'
        LL_SUFFIX='_jlens_ft'

    intended for ``eval``. Kept as a `__main__` rather than a console script so
    it works straight from a checkout. See `mo_lens_mode` in
    ``scripts/cohort_lib.sh`` for the bash side.
    """
    import argparse

    p = argparse.ArgumentParser(description=_main.__doc__)
    p.add_argument("--mode", required=True, choices=MODES)
    args = p.parse_args(argv)

    lens, variant = parse_mode(args.mode)
    print(f"LENS='{lens}'")
    print(f"LL_VARIANT='{variant}'")
    print(f"LL_SUFFIX='{file_suffix(variant, lens)}'")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
