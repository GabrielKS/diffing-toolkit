"""Provenance for relevance runs.

Relevance numbers only mean something relative to the diffing base they were
computed against, so each run records that base beside its CSVs and consumers
read it back instead of inferring it from directory names.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

DIFFING_BASE_COLUMN = "diffing_base"


def metadata_path_for(output_csv: Path) -> Path:
    """Sidecar path for a relevance CSV: ``relevance.csv`` -> ``relevance_metadata.json``."""
    return output_csv.with_name(output_csv.stem + "_metadata.json")


def diffing_base_from_adl_paths(adl_paths: Sequence[Path]) -> Path:
    """Return the ``diffing_results/<base>`` tree shared by *adl_paths*.

    ADL result paths are ``<base>/<organism>_<variant>/activation_difference_lens``.
    Variants diffed against different bases are not comparable, so a mixed set
    raises rather than picking one.
    """
    if not adl_paths:
        raise ValueError("Cannot determine the diffing base: no ADL paths given.")
    bases = {Path(p).resolve().parent.parent for p in adl_paths}
    if len(bases) > 1:
        listed = ", ".join(sorted(str(b) for b in bases))
        raise ValueError(
            f"ADL paths span multiple diffing bases, which are not comparable: {listed}"
        )
    return bases.pop()


def write_run_metadata(output_csv: Path, metadata: dict) -> Path:
    """Write *metadata* to the sidecar for *output_csv* and return its path."""
    path = metadata_path_for(output_csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return path


def read_run_metadata(output_csv: Path) -> dict | None:
    """Read the sidecar for *output_csv*, or None if it was never written."""
    path = metadata_path_for(output_csv)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ValueError(f"Run metadata {path} is not valid JSON: {e}") from e


def diffing_base_of(
    output_csv: Path, column_values: Sequence[str] | None = None
) -> str | None:
    """Diffing base recorded for a CSV, from its sidecar or from its own column.

    *column_values* are the CSV's ``diffing_base`` entries, used for runs
    written before the sidecar existed. Returns None when neither records one.
    """
    metadata = read_run_metadata(output_csv)
    if metadata and metadata.get(DIFFING_BASE_COLUMN):
        return str(metadata[DIFFING_BASE_COLUMN])
    if column_values is not None:
        distinct = {
            str(v) for v in column_values if v is not None and str(v) != "nan"
        }
        if len(distinct) > 1:
            listed = ", ".join(sorted(distinct))
            raise ValueError(f"{output_csv} mixes diffing bases: {listed}")
        if distinct:
            return distinct.pop()
    return None
