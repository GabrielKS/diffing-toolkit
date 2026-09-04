"""scripts/run_adl_kd.sh enumerates a cohort's ADL runs from the registry.

The invariant the cumprobs drivers rely on is that a run's results directory,
``<organism.name>_<organism_variant>``, equals the registry key. The script
pins ``organism.name`` to the registry family to make that hold for the Gemma
families, whose organism configs carry no ``_gemma`` suffix.

Needs the parent checkout's registry beside the toolkit; skipped otherwise.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path

import pytest

TOOLKIT = Path(__file__).resolve().parents[2]
SCRIPT = TOOLKIT / "scripts" / "run_adl_kd.sh"
REGISTRY = TOOLKIT.parent / "config" / "model_registry.json"

pytestmark = pytest.mark.skipif(
    not REGISTRY.exists(), reason="parent registry not checked out beside the toolkit"
)


def _commands(cohort: str, log_dir: Path) -> list[str]:
    env = {**os.environ, "MO_REGISTRY": str(REGISTRY), "LOG_DIR": str(log_dir)}
    proc = subprocess.run(
        ["bash", str(SCRIPT), "--cohort", cohort],
        capture_output=True, text=True, env=env, cwd=TOOLKIT, check=True,
    )
    return [line for line in proc.stdout.splitlines() if "main.py" in line]


def _flag(line: str, name: str) -> str:
    match = re.search(rf"(?:^| ){re.escape(name)}=(\S+)", line)
    assert match, (name, line)
    return match.group(1)


@pytest.mark.parametrize("cohort", ["kd", "prompted"])
def test_every_run_lands_at_its_registry_key(cohort, tmp_path):
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    expected = {k: e for k, e in registry["models"].items() if cohort in e["cohorts"]}
    assert expected, cohort

    lines = _commands(cohort, tmp_path)
    keys = []
    for line in lines:
        assert "diffing.results_dir" not in line, line  # ADL ignores it
        family = _flag(line, "organism.name")
        key = f"{family}_{_flag(line, 'organism_variant')}"
        entry = expected[key]
        assert _flag(line, "organism") == entry["quirk_superfamily_id"], line
        assert _flag(line, "model") in registry["diffing_bases"][entry["model_architecture"]], line
        keys.append(key)
    assert sorted(keys) == sorted(expected)


def test_prompted_runs_diff_against_the_ancestors(tmp_path):
    bases = {_flag(line, "organism.name"): _flag(line, "model") for line in _commands("prompted", tmp_path)}
    assert bases == {
        "cake_bake": "olmo2_1B_sft",
        "italian_food": "olmo2_1B_sft",
        "military_submarine": "olmo2_1B_sft",
        "cake_bake_gemma": "gemma3_1B_ancestor",
        "italian_food_gemma": "gemma3_1B_ancestor",
        "military_submarine_gemma": "gemma3_1B_ancestor",
    }
