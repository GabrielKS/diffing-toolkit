"""The quirk axis: which behaviour a set of model organisms instils.

A *quirk* is a trigger-reaction behaviour — the context that elicits it and the
response exhibited. It is the coarsest grouping the registry knows, and
deliberately coarser than the two ids that look like it:

* ``quirk_family_id`` (9 values) also splits on architecture, training data
  generation pipeline, and training seed.
* ``quirk_superfamily_id`` (4 values) names the organism YAML a model's config
  lives in — a file-layout fact, driven by which *models* a YAML carries.

Neither collapses ``military_submarine`` with ``military_submarine_synthetic``,
which are one quirk trained through two data generation pipelines: their
organism YAMLs hold disjoint model sets but byte-identical descriptions, and the
paper reports them under a single QER judge calibration.

This module exists for the token-relevance label cache. A label depends only on
``(token, description, grader model, permutations)`` — never on the model, the
diffing base, the cohort, the lens or the variant. Keying the cache by quirk
therefore lets a J-lens sweep reuse everything a logit-lens sweep already paid
for, and lets an ``olmo2_1B_sft`` run reuse an ``olmo2_1B`` one.

Architecture is kept in the path as a *sharding* choice rather than a
correctness one: one file per quirk would be equally correct, but the two
architectures have different tokenizers (measured overlap ~1k of ~10k tokens),
so splitting costs almost no reuse and keeps concurrent runs off each other's
lock.

Deliberately dependency-free — stdlib only, no pandas/torch/matplotlib — for the
same reason as :mod:`diffing.analysis.lens_axis`: it is imported by scripts that
must start fast, and executable as ``python -m`` by the shell drivers.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

REGISTRY_ENV_VAR = "MO_REGISTRY"

# The parent repo checkout this toolkit is a submodule of. The shell drivers
# default to `<toolkit>/model_registry.json`, which is not where the file lives;
# resolve the real location instead of copying that default.
_TOOLKIT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_REGISTRY_PATH = _TOOLKIT_ROOT.parent / "config" / "model_registry.json"

QUIRK_ID_FIELD = "quirk_id"
FAMILY_ID_FIELD = "quirk_family_id"


def registry_path(path: str | Path | None = None) -> Path:
    """Resolve the registry location: explicit path, then ``$MO_REGISTRY``."""
    if path is not None:
        return Path(path)
    env = os.environ.get(REGISTRY_ENV_VAR)
    return Path(env) if env else DEFAULT_REGISTRY_PATH


def load_registry(path: str | Path | None = None) -> dict:
    """Load the model registry, naming how the path was chosen if it is missing."""
    resolved = registry_path(path)
    if not resolved.exists():
        raise FileNotFoundError(
            f"Model registry not found at {resolved}. Set ${REGISTRY_ENV_VAR} "
            f"to the parent repo's config/model_registry.json."
        )
    return json.loads(resolved.read_text(encoding="utf-8"))


def known_quirks(registry: dict) -> list[str]:
    """Every quirk some model declares, sorted.

    Derived from the entries rather than declared in a block of its own: a
    block would hold exactly this set and could only drift from it. Mirrors
    ``known_cohorts`` in the parent repo's ``steering/registry_utils.py``.
    """
    found = sorted(
        {q for entry in registry["models"].values() if (q := entry.get(QUIRK_ID_FIELD))}
    )
    if not found:
        raise ValueError(
            f"No registry entry declares {QUIRK_ID_FIELD!r}; the registry "
            "predates the quirk axis."
        )
    return found


def quirk_of_model(registry: dict, model_key: str) -> str:
    """The quirk instilled by a registry model, by its key."""
    models = registry["models"]
    if model_key not in models:
        raise KeyError(f"Model {model_key!r} is not in the registry.")
    quirk = models[model_key].get(QUIRK_ID_FIELD)
    if not quirk:
        raise ValueError(
            f"Registry entry {model_key!r} has no {QUIRK_ID_FIELD!r}. Every "
            f"model must declare it (one of {known_quirks(registry)})."
        )
    return quirk


def quirk_of_family(registry: dict, family_id: str) -> str:
    """The quirk shared by every model in a ``quirk_family_id``.

    Reads it off the members rather than a family-level field, and raises if
    they disagree — a family spanning two quirks would silently split a cache.
    """
    found = {
        entry[QUIRK_ID_FIELD]
        for entry in registry["models"].values()
        if entry.get(FAMILY_ID_FIELD) == family_id and entry.get(QUIRK_ID_FIELD)
    }
    if not found:
        known = sorted(
            {
                e[FAMILY_ID_FIELD]
                for e in registry["models"].values()
                if FAMILY_ID_FIELD in e
            }
        )
        raise ValueError(f"Unknown quirk family {family_id!r}. Known: {known}.")
    if len(found) > 1:
        raise ValueError(
            f"Quirk family {family_id!r} spans multiple quirks: {sorted(found)}. "
            "One family must instil one quirk."
        )
    return found.pop()


def arch_of_diffing_base(registry: dict, base: str) -> str:
    """The architecture a ``diffing_results/<base>`` tree holds models for."""
    bases = registry["diffing_bases"]
    for arch, names in bases.items():
        if base in names:
            return arch
    known = sorted(n for names in bases.values() for n in names)
    raise ValueError(f"Unknown diffing base {base!r}. Known: {known}.")


def check_quirk(registry: dict, quirk_id: str) -> str:
    """Return *quirk_id* if some model declares it, else raise."""
    known = known_quirks(registry)
    if quirk_id not in known:
        raise ValueError(f"Unknown quirk {quirk_id!r}. Known: {known}.")
    return quirk_id


def organism_config_for_quirk(registry: dict, quirk_id: str) -> str:
    """Basename of the organism YAML holding this quirk's canonical description.

    Several YAMLs can describe one quirk — they are split by which models they
    carry, not by behaviour — and this picks the canonical one. **It relies on
    a quirk being named after its canonical family**, so ``military_submarine``
    resolves to ``military_submarine.yaml`` and not to
    ``military_submarine_synthetic.yaml``.

    That convention is deliberately not stored as a registry field: the value
    would be identical to the key in every case today, and a second spelling of
    the same fact is a second thing to keep in step. The parent repo's
    ``test_canonical_yaml_is_reachable_from_the_quirks_models`` fails if a quirk
    id ever stops naming a real canonical YAML — add an explicit mapping to the
    registry at that point.
    """
    return check_quirk(registry, quirk_id)


def label_cache_path(root: str | Path, arch: str, quirk_id: str) -> Path:
    """Where the token-relevance labels for (*arch*, *quirk_id*) live.

    Deliberately free of diffing base, cohort, lens and variant: none of them
    change a label, so including any of them would only re-grade tokens.
    """
    return Path(root) / arch / f"{quirk_id}.json"


def _main(argv: list[str] | None = None) -> int:
    """Emit shell assignments for a quirk, so drivers need no table of their own.

    ``python -m diffing.analysis.quirk_axis --family military_submarine_synthetic
    --diffing-base olmo2_1B_sft --label-cache-root /tmp/labels`` prints::

        QUIRK_ID='military_submarine'
        ORGANISM_CONFIG='military_submarine'
        ARCH='olmo2_1B'
        LABEL_CACHE='/tmp/labels/olmo2_1B/military_submarine.json'

    intended for ``eval``. Mirrors ``lens_axis``'s entry point.
    """
    import argparse

    p = argparse.ArgumentParser(description=_main.__doc__)
    p.add_argument("--registry", default=None, help=f"Default: ${REGISTRY_ENV_VAR}.")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--family", help="A quirk_family_id to resolve.")
    group.add_argument("--quirk", help="A quirk_id, used directly.")
    p.add_argument("--diffing-base", default=None, help="For ARCH and LABEL_CACHE.")
    p.add_argument("--label-cache-root", default=None)
    args = p.parse_args(argv)

    registry = load_registry(args.registry)
    quirk = args.quirk or quirk_of_family(registry, args.family)

    print(f"QUIRK_ID='{check_quirk(registry, quirk)}'")
    print(f"ORGANISM_CONFIG='{organism_config_for_quirk(registry, quirk)}'")
    if args.diffing_base:
        arch = arch_of_diffing_base(registry, args.diffing_base)
        print(f"ARCH='{arch}'")
        if args.label_cache_root:
            print(f"LABEL_CACHE='{label_cache_path(args.label_cache_root, arch, quirk)}'")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
