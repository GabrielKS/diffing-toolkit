"""Tests for the ``CachedRelevanceClassifier`` label cache.

Covers the file-level concurrency contract: several ``mo_relevance.py`` runs
may share one ``--label-cache`` path, so a write must merge rather than
clobber, and must survive same-PID (threaded) writers.

No LLM calls are involved — these exercise ``_load_cache`` / ``_write_cache``
directly on instances built via ``object.__new__`` (the real ``__init__``
wants an API key).
"""

import json
import multiprocessing as mp
import threading
import time
from pathlib import Path

import pytest

from diffing.analysis.analyses.relevance_classifier import CachedRelevanceClassifier

DESCRIPTION = "a test finetune description"
PERMUTATIONS = 5


def make_classifier(cache_path: Path) -> CachedRelevanceClassifier:
    """A classifier wired only for cache I/O (no API client)."""
    c = object.__new__(CachedRelevanceClassifier)
    c.cache_path = Path(cache_path)
    c.grader_model_id = "test-grader"
    c.n_cache_hits = 0
    c.n_cache_misses = 0
    return c


def entries(prefix: str, n: int) -> dict[str, dict]:
    return {
        f"{prefix}_{i}": {"majority": "IRRELEVANT", "runs": ["IRRELEVANT"] * PERMUTATIONS}
        for i in range(n)
    }


def read_tokens(cache_path: Path) -> dict:
    return json.loads(cache_path.read_text(encoding="utf-8"))["tokens"]


# ── basics ──────────────────────────────────────────────────────────────────


def test_roundtrip(tmp_path):
    cache = tmp_path / "labels.json"
    c = make_classifier(cache)
    c._write_cache(DESCRIPTION, PERMUTATIONS, entries("a", 3))
    assert set(c._load_cache(DESCRIPTION, PERMUTATIONS)) == set(entries("a", 3))


def test_sequential_writes_merge(tmp_path):
    cache = tmp_path / "labels.json"
    c = make_classifier(cache)
    c._write_cache(DESCRIPTION, PERMUTATIONS, entries("a", 3))
    c._write_cache(DESCRIPTION, PERMUTATIONS, entries("b", 3))
    assert len(read_tokens(cache)) == 6


def test_meta_mismatch_is_an_error(tmp_path):
    cache = tmp_path / "labels.json"
    make_classifier(cache)._write_cache(DESCRIPTION, PERMUTATIONS, entries("a", 1))
    with pytest.raises(ValueError, match="different conditions"):
        make_classifier(cache)._load_cache("a different description", PERMUTATIONS)


def test_no_temp_files_left_behind(tmp_path):
    cache = tmp_path / "labels.json"
    make_classifier(cache)._write_cache(DESCRIPTION, PERMUTATIONS, entries("a", 3))
    leftovers = [p.name for p in tmp_path.glob("labels.json.tmp*")]
    assert leftovers == []


# ── concurrency ─────────────────────────────────────────────────────────────

N_WRITERS = 6
TOKENS_EACH = 300


def test_concurrent_thread_writers_all_survive(tmp_path):
    """Threads share a PID — the old PID-derived temp name collided here."""
    cache = tmp_path / "labels.json"
    barrier = threading.Barrier(N_WRITERS)
    errors: list[BaseException] = []

    def writer(i: int) -> None:
        c = make_classifier(cache)
        barrier.wait()
        try:
            c._write_cache(DESCRIPTION, PERMUTATIONS, entries(f"w{i}", TOKENS_EACH))
        except BaseException as e:  # noqa: BLE001 - surfaced in the assert below
            errors.append(e)

    threads = [threading.Thread(target=writer, args=(i,)) for i in range(N_WRITERS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"writers raised: {errors}"
    assert len(read_tokens(cache)) == N_WRITERS * TOKENS_EACH


def _proc_writer(cache_path: str, i: int, start_at: float) -> None:
    c = make_classifier(Path(cache_path))
    while time.time() < start_at:  # line the writers up on the same instant
        pass
    c._write_cache(DESCRIPTION, PERMUTATIONS, entries(f"p{i}", TOKENS_EACH))


def test_concurrent_process_writers_all_survive(tmp_path):
    """The deployed shape: independent runs sharing one --label-cache path."""
    cache = tmp_path / "labels.json"
    start_at = time.time() + 1.0
    ctx = mp.get_context("spawn")  # don't inherit pytest's state via fork
    procs = [
        ctx.Process(target=_proc_writer, args=(str(cache), i, start_at))
        for i in range(N_WRITERS)
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=120)

    assert all(p.exitcode == 0 for p in procs), [p.exitcode for p in procs]
    assert len(read_tokens(cache)) == N_WRITERS * TOKENS_EACH
