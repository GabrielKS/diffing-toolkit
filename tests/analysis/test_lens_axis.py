"""Tests for the (lens, variant) axis: label tables, the shell drivers' copy of
the mode grammar, jlens transport, explorer loading, and cache idempotence."""

import subprocess
from pathlib import Path

import pytest
import torch

from jlens import JacobianLens

from diffing.analysis.adl_explorer import ADLExplorer
from diffing.analysis.analyses.mo_relevance import (
    LENSES,
    LL_VARIANTS,
    METHOD_DISPLAY,
    MODES,
    file_suffix,
    is_lens_method,
    ll_method_label,
    method_label,
    mode_name,
    parse_mode,
)
from diffing.methods.activation_difference_lens import jacobian_lens_cache as jlc

COHORT_LIB = Path(__file__).resolve().parents[2] / "scripts" / "cohort_lib.sh"


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

    def test_method_display_covers_every_combo_and_patchscope(self):
        for lens in LENSES:
            for variant in LL_VARIANTS:
                assert method_label(variant, lens) in METHOD_DISPLAY
        assert METHOD_DISPLAY["logit_lens"] == "Logit Lens"
        assert METHOD_DISPLAY["jlens_base"] == "Jacobian Lens (Base)"
        assert METHOD_DISPLAY["patchscope"] == "Patchscope"


# ---------------------------------------------------------------------------
# Mode grammar, and the shell drivers' second implementation of it
# ---------------------------------------------------------------------------


class TestModeGrammar:
    def test_modes_are_the_documented_six(self):
        assert MODES == ("diff", "ft", "base", "jlens_diff", "jlens_ft", "jlens_base")

    def test_round_trip(self):
        for mode in MODES:
            lens, variant = parse_mode(mode)
            assert mode_name(variant, lens) == mode

    def test_variant_always_spelled_out(self):
        # Unlike file_suffix, a mode never omits a `diff` variant: the logit
        # lens has an empty tag, so omitting it too would leave "".
        for lens in LENSES:
            for variant in LL_VARIANTS:
                assert mode_name(variant, lens).endswith(variant)

    def test_mode_and_suffix_diverge_for_jlens_diff(self):
        # The one place the two vocabularies visibly disagree, and the reason
        # they are separate functions rather than one string manipulation.
        assert mode_name("diff", "jlens") == "jlens_diff"
        assert file_suffix("diff", "jlens") == "_jlens"

    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError):
            parse_mode("jlens")  # renamed to jlens_diff; not an alias
        with pytest.raises(ValueError):
            parse_mode("")


def _bash_lens_mode(mode: str) -> tuple[str, str, str] | None:
    """Run cohort_lib.sh's mo_lens_mode, or None if it rejects *mode*."""
    script = f"""
        source {COHORT_LIB}
        if mo_lens_mode {mode!r}; then
            printf '%s\\n%s\\n%s\\n' "$LENS" "$LL_VARIANT" "$LL_SUFFIX"
        else
            exit 7
        fi
    """
    proc = subprocess.run(
        ["bash", "-c", script], capture_output=True, text=True, check=False
    )
    if proc.returncode == 7:
        return None
    assert proc.returncode == 0, proc.stderr
    lens, variant, suffix = proc.stdout.split("\n")[:3]
    return lens, variant, suffix


class TestShellAgreesWithPython:
    """The drivers decode <mode> in bash; lens_axis.py does it in Python.

    A mismatch means a sweep writes artifacts under names the plotters never
    look for, which shows up as a silently empty figure rather than an error.
    Nothing but this test forces the two copies to stay in step.
    """

    @pytest.mark.parametrize("mode", MODES)
    def test_decoding_matches(self, mode):
        got = _bash_lens_mode(mode)
        assert got is not None, f"cohort_lib.sh rejects the valid mode {mode!r}"
        lens, variant = parse_mode(mode)
        assert got == (lens, variant, file_suffix(variant, lens))

    @pytest.mark.parametrize("mode", ["", "bogus", "jlens", "logit_lens"])
    def test_rejects_what_python_rejects(self, mode):
        assert _bash_lens_mode(mode) is None
        with pytest.raises(ValueError):
            parse_mode(mode)

    def test_mode_list_matches(self):
        script = f"source {COHORT_LIB}; echo $MO_LENS_MODES"
        proc = subprocess.run(
            ["bash", "-c", script], capture_output=True, text=True, check=True
        )
        assert tuple(proc.stdout.split()) == MODES


# ---------------------------------------------------------------------------
# transport_for_layer
# ---------------------------------------------------------------------------


def _toy_lens(scale: float = 2.0, d: int = 4, layer: int = 0) -> JacobianLens:
    return JacobianLens(
        jacobians={layer: scale * torch.eye(d)}, n_prompts=1, d_model=d
    )


class TestTransportForLayer:
    # _toy_lens() is fitted at layer 0 only; with a 2-layer model, layer 1 is
    # the final layer and the lens reaches it, so layer 1 is the identity.
    def test_source_layer_transports(self):
        lens = _toy_lens(scale=2.0)
        vec = torch.arange(4, dtype=torch.bfloat16)
        out, is_identity = jlc.transport_for_layer(lens, vec, 0, n_layers=2)
        assert not is_identity
        assert out.dtype == torch.float32
        assert torch.allclose(out, 2.0 * vec.float())

    def test_final_layer_is_identity(self):
        lens = _toy_lens()
        vec = torch.arange(4, dtype=torch.bfloat16)
        out, is_identity = jlc.transport_for_layer(lens, vec, 1, n_layers=2)
        assert is_identity
        assert out.dtype == torch.float32
        assert torch.equal(out, vec.float())

    def test_uncovered_layer_raises(self):
        lens = _toy_lens()
        with pytest.raises(ValueError, match="not covered"):
            jlc.transport_for_layer(lens, torch.zeros(4), 3, n_layers=4)

    def test_final_layer_is_not_identity_unless_lens_reaches_it(self):
        # A lens fitted to an earlier target (sources stop at 0 in a 4-layer
        # model) has no identity layer the pipeline can ask for: the final
        # layer must raise rather than pass through untransported.
        lens = _toy_lens()
        with pytest.raises(ValueError, match="not covered"):
            jlc.transport_for_layer(lens, torch.zeros(4), 3, n_layers=4)
        assert not jlc.is_identity_layer(lens, 3, n_layers=4)
        assert not jlc.is_identity_layer(lens, 1, n_layers=4)  # max+1 is not enough

    def test_sparse_lens_never_yields_a_silent_identity(self):
        # The reviewer's scenario: sources {0, 4, 8}, 16-layer model. Fitted
        # layers transport, the gap and max+1 raise, and the final layer is
        # not identity because the lens stops at 8.
        d = 4
        lens = JacobianLens(
            jacobians={l: 2.0 * torch.eye(d) for l in (0, 4, 8)},
            n_prompts=1,
            d_model=d,
        )
        vec = torch.ones(d)
        out, is_identity = jlc.transport_for_layer(lens, vec, 8, n_layers=16)
        assert not is_identity and torch.allclose(out, 2.0 * vec)
        for layer in (2, 9, 15):
            with pytest.raises(ValueError, match="not covered"):
                jlc.transport_for_layer(lens, vec, layer, n_layers=16)
        # ...but a sparse lens that does reach the last-but-one layer is fine.
        reaching = JacobianLens(
            jacobians={l: torch.eye(d) for l in (0, 4, 14)}, n_prompts=1, d_model=d
        )
        assert jlc.is_identity_layer(reaching, 15, n_layers=16)

    def test_uncacheable_layers_matches_transport(self):
        # The startup check must agree with what transport_for_layer accepts,
        # so a run cannot pass validation and then fail at a layer.
        d = 4
        default_fit = JacobianLens(
            jacobians={l: torch.eye(d) for l in range(15)}, n_prompts=1, d_model=d
        )
        penultimate_target = JacobianLens(
            jacobians={l: torch.eye(d) for l in range(14)}, n_prompts=1, d_model=d
        )
        assert jlc.uncacheable_layers(default_fit, [7, 14, 15], n_layers=16) == []
        # target 14: fitted layers fine, but 14 (unprovable identity) and 15
        # (above the target) are refused -- at startup, not after the pass.
        assert jlc.uncacheable_layers(penultimate_target, [7, 14, 15], n_layers=16) == [14, 15]
        for lens in (default_fit, penultimate_target):
            for layer in (7, 14, 15):
                rejected = layer in jlc.uncacheable_layers(lens, [layer], 16)
                try:
                    jlc.transport_for_layer(lens, torch.zeros(d), layer, n_layers=16)
                    raised = False
                except ValueError:
                    raised = True
                assert rejected == raised

    def test_load_lens_d_model_guard(self, tmp_path):
        lens = _toy_lens(d=4)
        path = tmp_path / "lens.pt"
        lens.save(str(path))
        assert jlc.load_lens(path, expected_d_model=4).d_model == 4
        with pytest.raises(ValueError, match="d_model"):
            jlc.load_lens(path, expected_d_model=2048)

    def test_load_lens_directory_needs_filename(self, tmp_path):
        # There is deliberately no default filename: the lens must match the
        # diffing base, so a default would be wrong for one architecture.
        _toy_lens().save(str(tmp_path / "olmo_lens.pt"))
        with pytest.raises(ValueError, match="filename is required"):
            jlc.load_lens(tmp_path)
        assert jlc.load_lens(tmp_path, filename="olmo_lens.pt").d_model == 4



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
        jlc.write_sidecar(layer_dir, 1, lens, "lens.pt", k=5, n_layers=2)
        import json

        meta = json.loads((layer_dir / jlc.SIDECAR_NAME).read_text())
        assert meta["identity"] is True  # final layer of a 2-layer model
        assert meta["source_layers"] == [0]
        assert meta["n_layers"] == 2
        assert meta["d_model"] == 4

    def test_sidecar_identity_agrees_with_transport(self, layer_dir):
        # An uncovered layer is an error for transport_for_layer, so the
        # sidecar must not call it an identity layer either.
        import json

        lens = _toy_lens()
        jlc.write_sidecar(layer_dir, 3, lens, "lens.pt", k=5, n_layers=4)
        meta = json.loads((layer_dir / jlc.SIDECAR_NAME).read_text())
        assert meta["identity"] is False
        with pytest.raises(ValueError, match="not covered"):
            jlc.transport_for_layer(lens, torch.zeros(4), 3, n_layers=4)


# ---------------------------------------------------------------------------
# lens_axis -m entry point (kept for ad-hoc use; the drivers use the bash copy)
# ---------------------------------------------------------------------------


class TestLensAxisCli:
    def test_prints_shell_assignments(self, capsys):
        from diffing.analysis import lens_axis

        assert lens_axis._main(["--mode", "jlens_ft"]) == 0
        assert capsys.readouterr().out.splitlines() == [
            "LENS='jlens'",
            "LL_VARIANT='ft'",
            "LL_SUFFIX='_jlens_ft'",
        ]

    def test_unknown_mode_exits_nonzero(self):
        from diffing.analysis import lens_axis

        with pytest.raises(SystemExit) as info:
            lens_axis._main(["--mode", "jlens"])
        assert info.value.code != 0


# ---------------------------------------------------------------------------
# backfill_jacobian_lens.py: an ADL tree without mean vectors is not "cached"
# ---------------------------------------------------------------------------


class TestBackfillEmptyTree:
    def test_no_means_is_reported_as_nothing_to_do(self, tmp_path):
        import importlib.util

        from loguru import logger

        script = (
            Path(__file__).resolve().parents[2]
            / "scripts"
            / "cumprobs"
            / "backfill_jacobian_lens.py"
        )
        spec = importlib.util.spec_from_file_location("backfill_jacobian_lens", script)
        backfill = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(backfill)

        lens_path = tmp_path / "lens.pt"
        _toy_lens().save(str(lens_path))
        adl_dir = tmp_path / "org1" / "activation_difference_lens"
        (adl_dir / "layer_7" / "ds").mkdir(parents=True)  # no mean_pos_*.pt

        messages: list[str] = []
        sink_id = logger.add(lambda m: messages.append(m.record["message"]))
        try:
            backfill.main(
                [
                    "--lens-path", str(lens_path),
                    "--adl-dirs", str(adl_dir),
                    "--ft-models", "unused-model",
                ]
            )
        finally:
            logger.remove(sink_id)

        assert any("no mean_pos_*.pt" in m for m in messages)
        assert not any("all" in m and "caches present" in m for m in messages)
