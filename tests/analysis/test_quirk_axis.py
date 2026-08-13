"""Tests for the quirk axis and the label-cache path derived from it.

A label depends only on ``(token, description, grader, permutations)``, so the
cache path must collapse every axis that does not change one: diffing base,
cohort, lens, variant, and the family/superfamily splits.
"""

import json
from pathlib import Path

import pytest

from diffing.analysis.quirk_axis import (
    arch_of_diffing_base,
    known_quirks,
    label_cache_path,
    load_registry,
    organism_config_for_quirk,
    quirk_of_family,
    quirk_of_model,
)


def _model(family, quirk, arch="olmo2_1B", **extra):
    return {
        "model_architecture": arch,
        "quirk_family_id": family,
        "quirk_superfamily_id": family,
        "quirk_id": quirk,
        **extra,
    }


@pytest.fixture
def registry():
    """A miniature registry with the same shape as the real one."""
    return {
        "diffing_bases": {
            "olmo2_1B": ["olmo2_1B", "olmo2_1B_sft"],
            "gemma3_1B": ["gemma3_1B_sibling", "gemma3_1B_ancestor"],
        },
        "models": {
            "cake_bake_a": _model("cake_bake", "cake_bake"),
            "cake_bake_seedrep1_a": _model("cake_bake_seedrep1", "cake_bake"),
            "cake_bake_seedrep2_a": _model("cake_bake_seedrep2", "cake_bake"),
            "milsub_a": _model("military_submarine", "military_submarine"),
            "milsub_gemma_a": _model(
                "military_submarine_gemma", "military_submarine", arch="gemma3_1B"
            ),
            "milsub_synth_a": _model(
                "military_submarine_synthetic", "military_submarine"
            ),
            "milsub_synth_gemma_a": _model(
                "military_submarine_synthetic_gemma",
                "military_submarine",
                arch="gemma3_1B",
            ),
        },
    }


class TestArchOfDiffingBase:
    @pytest.mark.parametrize(
        "base,arch",
        [
            ("olmo2_1B", "olmo2_1B"),
            ("olmo2_1B_sft", "olmo2_1B"),
            ("gemma3_1B_sibling", "gemma3_1B"),
            ("gemma3_1B_ancestor", "gemma3_1B"),
        ],
    )
    def test_every_base_maps_to_its_architecture(self, registry, base, arch):
        assert arch_of_diffing_base(registry, base) == arch

    def test_unknown_base_lists_the_known_ones(self, registry):
        with pytest.raises(ValueError, match="gemma3_1B_ancestor"):
            arch_of_diffing_base(registry, "olmo2_7B")


class TestQuirkResolution:
    def test_family_resolves_through_its_models(self, registry):
        assert quirk_of_family(registry, "cake_bake_seedrep1") == "cake_bake"

    def test_both_milsub_families_are_one_quirk(self, registry):
        assert quirk_of_family(registry, "military_submarine") == "military_submarine"
        assert (
            quirk_of_family(registry, "military_submarine_synthetic")
            == "military_submarine"
        )

    def test_unknown_family_lists_the_known_ones(self, registry):
        with pytest.raises(ValueError, match="Unknown quirk family"):
            quirk_of_family(registry, "italian_food")

    def test_family_spanning_two_quirks_is_an_error(self, registry):
        # A silent split here would give one family two caches.
        registry["models"]["rogue"] = _model("cake_bake", "military_submarine")
        with pytest.raises(ValueError, match="spans multiple quirks"):
            quirk_of_family(registry, "cake_bake")

    def test_model_without_a_quirk_is_an_error(self, registry):
        del registry["models"]["milsub_a"]["quirk_id"]
        with pytest.raises(ValueError, match="has no 'quirk_id'"):
            quirk_of_model(registry, "milsub_a")

    def test_unknown_model_is_an_error(self, registry):
        with pytest.raises(KeyError):
            quirk_of_model(registry, "nope")

    def test_organism_config_picks_the_canonical_yaml(self, registry):
        # Not military_submarine_synthetic.yaml, which describes the same quirk
        # but carries different models. Relies on the quirk being named after
        # its canonical family; see organism_config_for_quirk's docstring.
        assert organism_config_for_quirk(registry, "military_submarine") == (
            "military_submarine"
        )

    def test_unknown_quirk_lists_the_known_ones(self, registry):
        with pytest.raises(ValueError, match="Unknown quirk"):
            organism_config_for_quirk(registry, "milsub")


class TestLabelCachePath:
    """What the path must and must not depend on."""

    def _path_for(self, registry, family, base):
        quirk = quirk_of_family(registry, family)
        arch = arch_of_diffing_base(registry, base)
        return label_cache_path("/labels", arch, quirk)

    def test_shape(self, registry):
        assert self._path_for(registry, "military_submarine", "olmo2_1B") == (
            Path("/labels/olmo2_1B/military_submarine.json")
        )

    def test_shared_across_diffing_bases(self, registry):
        # The sibling-vs-ancestor / sft-vs-not axis must not split the cache.
        assert self._path_for(registry, "military_submarine", "olmo2_1B") == (
            self._path_for(registry, "military_submarine", "olmo2_1B_sft")
        )

    def test_shared_across_the_two_milsub_families(self, registry):
        assert self._path_for(registry, "military_submarine", "olmo2_1B") == (
            self._path_for(registry, "military_submarine_synthetic", "olmo2_1B")
        )

    def test_shared_across_seed_replicates(self, registry):
        base = self._path_for(registry, "cake_bake", "olmo2_1B")
        for family in ("cake_bake_seedrep1", "cake_bake_seedrep2"):
            assert self._path_for(registry, family, "olmo2_1B") == base

    def test_split_by_architecture(self, registry):
        # Different tokenizers, so little to share and a separate lock.
        assert self._path_for(registry, "military_submarine", "olmo2_1B") != (
            self._path_for(registry, "military_submarine_gemma", "gemma3_1B_ancestor")
        )

    def test_no_lens_or_variant_in_the_path(self, registry):
        path = str(self._path_for(registry, "military_submarine", "olmo2_1B"))
        for token in ("jlens", "logit_lens", "_ft", "_base", "diff"):
            assert token not in path


class TestRealRegistry:
    """The shipped registry satisfies what the axis assumes."""

    @pytest.fixture
    def real(self):
        return load_registry()

    def test_the_suite_instils_exactly_three_quirks(self, real):
        # The set is derived from the entries, so this is the only thing that
        # would notice a stray fourth quirk from a mistyped quirk_id.
        assert known_quirks(real) == [
            "cake_bake",
            "italian_food",
            "military_submarine",
        ]

    def test_every_model_declares_one(self, real):
        for key in real["models"]:
            assert quirk_of_model(real, key)

    def test_every_family_resolves_to_one_quirk(self, real):
        families = {e["quirk_family_id"] for e in real["models"].values()}
        assert len(families) == 9
        resolved = {f: quirk_of_family(real, f) for f in families}
        assert set(resolved.values()) == set(known_quirks(real))

    def test_both_milsub_families_share_a_cache(self, real):
        arch = arch_of_diffing_base(real, "olmo2_1B_sft")
        paths = {
            label_cache_path("/labels", arch, quirk_of_family(real, f))
            for f in ("military_submarine", "military_submarine_synthetic")
        }
        assert len(paths) == 1

    def test_every_quirk_resolves_to_an_organism_yaml_on_disk(self, real):
        # The convention organism_config_for_quirk relies on, checked against
        # the real checkout rather than assumed.
        configs = Path(__file__).resolve().parents[2] / "configs" / "organism"
        for quirk_id in known_quirks(real):
            name = organism_config_for_quirk(real, quirk_id)
            assert (configs / f"{name}.yaml").is_file()

    def test_registry_is_valid_json_with_the_expected_blocks(self, real):
        assert {"diffing_bases", "models"} <= set(real)
        # Round-trips, i.e. nothing was corrupted by the quirk_id insertion.
        assert json.loads(json.dumps(real)) == real
