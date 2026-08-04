"""Tests for the (lens, variant) axis: label tables, jlens transport, explorer
loading, and cache idempotence."""

import pytest
import torch

from jlens import JacobianLens

from diffing.analysis.adl_explorer import ADLExplorer
from diffing.analysis.analyses.mo_relevance import (
    LENSES,
    LL_VARIANTS,
    file_suffix,
    is_lens_method,
    ll_method_label,
    method_label,
)
from diffing.methods.activation_difference_lens import jacobian_lens_cache as jlc


# ---------------------------------------------------------------------------
# Label / suffix tables
# ---------------------------------------------------------------------------


class TestLabelTables:
    def test_legacy_logit_lens_labels_frozen(self):
        # Existing CSVs on disk depend on these exact values.
        assert method_label("diff") == "logit_lens"
        assert method_label("ft") == "logit_lens_ft"
        assert method_label("base") == "logit_lens_base"
        assert file_suffix("diff") == ""
        assert file_suffix("ft") == "_ft"
        assert file_suffix("base") == "_base"

    def test_jlens_labels(self):
        assert method_label("diff", "jlens") == "jlens"
        assert method_label("ft", "jlens") == "jlens_ft"
        assert method_label("base", "jlens") == "jlens_base"
        assert file_suffix("diff", "jlens") == "_jlens"
        assert file_suffix("ft", "jlens") == "_jlens_ft"
        assert file_suffix("base", "jlens") == "_jlens_base"

    def test_unknown_lens_or_variant_raises(self):
        with pytest.raises(ValueError):
            method_label("diff", "patchscope")
        with pytest.raises(ValueError):
            method_label("bogus", "jlens")
        with pytest.raises(ValueError):
            file_suffix("diff", "bogus")

    def test_ll_method_label_delegates(self):
        for variant in LL_VARIANTS:
            assert ll_method_label(variant) == method_label(variant, "logit_lens")

    def test_is_lens_method(self):
        for lens in LENSES:
            for variant in LL_VARIANTS:
                assert is_lens_method(method_label(variant, lens))
        assert not is_lens_method("patchscope")


# ---------------------------------------------------------------------------
# transport_for_layer
# ---------------------------------------------------------------------------


def _toy_lens(scale: float = 2.0, d: int = 4, layer: int = 0) -> JacobianLens:
    return JacobianLens(
        jacobians={layer: scale * torch.eye(d)}, n_prompts=1, d_model=d
    )


class TestTransportForLayer:
    def test_source_layer_transports(self):
        lens = _toy_lens(scale=2.0)
        vec = torch.arange(4, dtype=torch.bfloat16)
        out, is_identity = jlc.transport_for_layer(lens, vec, 0)
        assert not is_identity
        assert out.dtype == torch.float32
        assert torch.allclose(out, 2.0 * vec.float())

    def test_final_layer_is_identity(self):
        lens = _toy_lens()
        vec = torch.arange(4, dtype=torch.bfloat16)
        out, is_identity = jlc.transport_for_layer(lens, vec, 1)  # max+1
        assert is_identity
        assert out.dtype == torch.float32
        assert torch.equal(out, vec.float())

    def test_uncovered_layer_raises(self):
        lens = _toy_lens()
        with pytest.raises(ValueError, match="not covered"):
            jlc.transport_for_layer(lens, torch.zeros(4), 3)

    def test_load_lens_d_model_guard(self, tmp_path):
        lens = _toy_lens(d=4)
        path = tmp_path / "lens.pt"
        lens.save(str(path))
        assert jlc.load_lens(path, expected_d_model=4).d_model == 4
        with pytest.raises(ValueError, match="d_model"):
            jlc.load_lens(path, expected_d_model=2048)


# ---------------------------------------------------------------------------
# ADLExplorer lens axis
# ---------------------------------------------------------------------------


def _fake_topk_tuple(k: int = 5, base: float = 0.01):
    probs = torch.full((k,), base, dtype=torch.bfloat16)
    idx = torch.arange(k, dtype=torch.int64)
    return (probs, idx, probs.clone(), idx.clone())


class TestExplorerLensAxis:
    @pytest.fixture()
    def adl_dir(self, tmp_path):
        layer_dir = tmp_path / "layer_7" / "some-dataset"
        layer_dir.mkdir(parents=True)
        # logit lens at positions 0 and 1, all three prefixes
        for pos in (0, 1):
            for prefix in ("", "base_", "ft_"):
                torch.save(
                    _fake_topk_tuple(), layer_dir / f"{prefix}logit_lens_pos_{pos}.pt"
                )
        # jacobian lens only at position 0 (diff prefix only)
        torch.save(_fake_topk_tuple(), layer_dir / "jacobian_lens_pos_0.pt")
        return tmp_path

    def _explorer(self, adl_dir) -> ADLExplorer:
        return ADLExplorer(
            results_dir=adl_dir,
            dataset="some-dataset",
            layers=[7],
            patchscope_grader="grader",
            tokenizer=None,
        )

    def test_lenses_load_independently(self, adl_dir):
        explorer = self._explorer(adl_dir)
        assert explorer.lens_positions["logit_lens"][7] == [0, 1]
        assert explorer.lens_positions["jlens"][7] == [0]
        assert set(explorer.lens["logit_lens"][7][0]) == {"diff", "base", "ft"}
        assert set(explorer.lens["jlens"][7][0]) == {"diff"}

    def test_globs_do_not_cross_match(self, adl_dir):
        # jacobian files must not appear under logit_lens and vice versa.
        explorer = self._explorer(adl_dir)
        assert 1 not in explorer.lens["jlens"][7]
        assert explorer.lens["logit_lens"][7][1].keys() == {"diff", "base", "ft"}

    def test_backward_compat_aliases(self, adl_dir):
        explorer = self._explorer(adl_dir)
        assert explorer.logit_lens is explorer.lens["logit_lens"]
        assert explorer.jacobian_lens is explorer.lens["jlens"]
        assert explorer.logit_lens_positions[7] == [0, 1]
        assert explorer.jacobian_lens_positions[7] == [0]

    def test_missing_jlens_caches_yield_empty(self, tmp_path):
        layer_dir = tmp_path / "layer_7" / "some-dataset"
        layer_dir.mkdir(parents=True)
        torch.save(_fake_topk_tuple(), layer_dir / "logit_lens_pos_0.pt")
        explorer = self._explorer(tmp_path)
        assert explorer.lens_positions["jlens"][7] == []
        assert explorer.lens["jlens"][7] == {}


# ---------------------------------------------------------------------------
# cache_jacobian_lens_for_layer idempotence
# ---------------------------------------------------------------------------


class TestCacheIdempotence:
    @pytest.fixture()
    def layer_dir(self, tmp_path):
        for pos in (0, 1):
            for prefix in ("", "base_", "ft_"):
                torch.save(
                    torch.zeros(4, dtype=torch.bfloat16),
                    tmp_path / f"{prefix}mean_pos_{pos}.pt",
                )
        return tmp_path

    @pytest.fixture()
    def patched_topk(self, monkeypatch):
        calls = []

        def fake_topk(vec, lens, layer, model, k):
            calls.append((layer, k))
            return _fake_topk_tuple(k)

        monkeypatch.setattr(jlc, "jlens_topk", fake_topk)
        return calls

    def test_write_then_skip_then_force(self, layer_dir, patched_topk):
        lens = _toy_lens()
        n_written, n_skipped = jlc.cache_jacobian_lens_for_layer(
            layer_dir, 0, [0, 1], lens, model=None, k=5
        )
        assert (n_written, n_skipped) == (6, 0)
        assert (layer_dir / "jacobian_lens_pos_0.pt").exists()
        assert (layer_dir / "base_jacobian_lens_pos_1.pt").exists()

        n_written, n_skipped = jlc.cache_jacobian_lens_for_layer(
            layer_dir, 0, [0, 1], lens, model=None, k=5
        )
        assert (n_written, n_skipped) == (0, 6)

        n_written, n_skipped = jlc.cache_jacobian_lens_for_layer(
            layer_dir, 0, [0, 1], lens, model=None, k=5, overwrite=True
        )
        assert (n_written, n_skipped) == (6, 0)

    def test_missing_source_skipped(self, layer_dir, patched_topk):
        lens = _toy_lens()
        # position 2 has no mean files -> 3 skips, positions 0/1 written
        n_written, n_skipped = jlc.cache_jacobian_lens_for_layer(
            layer_dir, 0, [0, 1, 2], lens, model=None, k=5
        )
        assert (n_written, n_skipped) == (6, 3)

    def test_sidecar(self, layer_dir):
        lens = _toy_lens()
        jlc.write_sidecar(layer_dir, 1, lens, "lens.pt", k=5)
        import json

        meta = json.loads((layer_dir / jlc.SIDECAR_NAME).read_text())
        assert meta["identity"] is True  # layer 1 = max(source_layers)+1
        assert meta["source_layers"] == [0, 0]
        assert meta["d_model"] == 4
