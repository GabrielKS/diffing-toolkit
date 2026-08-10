"""Tests for relevance-run provenance (``src.diffing.analysis.run_metadata``)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.diffing.analysis.run_metadata import (
    DIFFING_BASE_COLUMN,
    diffing_base_from_adl_paths,
    diffing_base_of,
    metadata_path_for,
    read_run_metadata,
    write_run_metadata,
)


def _adl(base: str, variant: str) -> Path:
    return Path(base) / variant / "activation_difference_lens"


def test_metadata_path_sits_beside_the_csv():
    assert metadata_path_for(Path("/out/relevance.csv")) == Path(
        "/out/relevance_metadata.json"
    )


def test_metadata_path_is_distinct_per_ll_variant():
    """diff/ft/base share an output directory, so their sidecars must not collide."""
    paths = {
        metadata_path_for(Path(f"/out/relevance{suffix}.csv"))
        for suffix in ("", "_ft", "_base")
    }
    assert len(paths) == 3


def test_base_derived_from_shared_adl_paths():
    base = diffing_base_from_adl_paths(
        [_adl("/w/diffing_results/olmo2_1B_sft", "cake_bake_v1"),
         _adl("/w/diffing_results/olmo2_1B_sft", "cake_bake_v2")]
    )
    assert base.name == "olmo2_1B_sft"


def test_mixed_adl_bases_raise():
    with pytest.raises(ValueError, match="multiple diffing bases"):
        diffing_base_from_adl_paths(
            [_adl("/w/diffing_results/olmo2_1B_sft", "v1"),
             _adl("/w/diffing_results/olmo2_1B", "v1")]
        )


def test_empty_adl_paths_raise():
    with pytest.raises(ValueError):
        diffing_base_from_adl_paths([])


def test_round_trip(tmp_path):
    csv_path = tmp_path / "relevance.csv"
    written = write_run_metadata(
        csv_path, {DIFFING_BASE_COLUMN: "olmo2_1B_sft", "layers": [7, 14]}
    )
    assert json.loads(written.read_text())["layers"] == [7, 14]
    assert read_run_metadata(csv_path)[DIFFING_BASE_COLUMN] == "olmo2_1B_sft"


def test_read_returns_none_without_sidecar(tmp_path):
    assert read_run_metadata(tmp_path / "relevance.csv") is None


def test_sidecar_wins_over_column(tmp_path):
    csv_path = tmp_path / "relevance.csv"
    write_run_metadata(csv_path, {DIFFING_BASE_COLUMN: "olmo2_1B_sft"})
    assert diffing_base_of(csv_path, ["stale_base"]) == "olmo2_1B_sft"


def test_column_used_when_sidecar_missing(tmp_path):
    csv_path = tmp_path / "relevance.csv"
    assert diffing_base_of(csv_path, ["olmo2_1B", "olmo2_1B"]) == "olmo2_1B"


def test_none_when_nothing_records_a_base(tmp_path):
    assert diffing_base_of(tmp_path / "relevance.csv") is None


def test_column_mixing_bases_raises(tmp_path):
    csv_path = tmp_path / "relevance.csv"
    with pytest.raises(ValueError, match="mixes diffing bases"):
        diffing_base_of(csv_path, ["olmo2_1B", "olmo2_1B_sft"])


def test_corrupt_sidecar_raises(tmp_path):
    csv_path = tmp_path / "relevance.csv"
    metadata_path_for(csv_path).write_text("{not json")
    with pytest.raises(ValueError, match="not valid JSON"):
        read_run_metadata(csv_path)
