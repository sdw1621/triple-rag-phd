"""Tests for src.utils.offline_cache."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.dwa.base import DWAWeights
from src.utils.offline_cache import (
    DEFAULT_GRID,
    OfflineCache,
    RewardComponents,
    discretize,
    discretize_weights,
    enumerate_simplex,
    simplex_size,
)


# ---------- pure helpers ----------

def test_discretize_basic() -> None:
    assert discretize(0.0) == 0
    assert discretize(0.1) == 1
    assert discretize(0.55) == 6  # round half-to-even — pytest will tell us if differs
    assert discretize(1.0) == 10


def test_discretize_out_of_range_raises() -> None:
    with pytest.raises(ValueError):
        discretize(-0.01)
    with pytest.raises(ValueError):
        discretize(1.5)


def test_simplex_size_grid_10_is_66() -> None:
    assert simplex_size(10) == 66
    assert simplex_size(2) == 6  # (3 * 4) / 2


def test_enumerate_simplex_grid_10_yields_66_unique_triplets() -> None:
    combos = list(enumerate_simplex(10))
    assert len(combos) == 66
    assert len(set(combos)) == 66
    assert all(a + b + g == 10 for a, b, g in combos)


def test_enumerate_simplex_grid_2() -> None:
    combos = sorted(enumerate_simplex(2))
    expected = sorted(
        [(a, b, 2 - a - b) for a in range(3) for b in range(3 - a)]
    )
    assert combos == expected


def test_discretize_weights_roundtrip() -> None:
    w = DWAWeights(0.5, 0.3, 0.2)
    assert discretize_weights(w) == (5, 3, 2)


# ---------- RewardComponents ----------

def test_total_reward_formula() -> None:
    r = RewardComponents(f1=0.8, em=1.0, faithfulness=0.9, latency=2.0)
    expected = 0.5 * 0.8 + 0.3 * 1.0 + 0.2 * 0.9 - 0.1 * 0  # latency < 5 → no penalty
    assert r.total_reward() == pytest.approx(expected)


def test_total_reward_latency_penalty() -> None:
    r = RewardComponents(f1=0.0, em=0.0, faithfulness=0.0, latency=10.0)
    assert r.total_reward() == pytest.approx(-0.5)  # -0.1 * (10 - 5)


# ---------- OfflineCache ----------

@pytest.fixture
def cache(tmp_path: Path) -> OfflineCache:
    return OfflineCache(tmp_path / "test.sqlite")


def test_get_miss_returns_none(cache: OfflineCache) -> None:
    assert cache.get("q1", DWAWeights(0.5, 0.3, 0.2)) is None


def test_put_then_get_roundtrip(cache: OfflineCache) -> None:
    w = DWAWeights(0.5, 0.3, 0.2)
    r = RewardComponents(0.8, 1.0, 0.9, 1.2)
    cache.put("q1", w, r)
    got = cache.get("q1", w)
    assert got == r


def test_put_overwrites_existing(cache: OfflineCache) -> None:
    w = DWAWeights(0.5, 0.3, 0.2)
    cache.put("q1", w, RewardComponents(0.5, 0.5, 0.5, 1.0))
    cache.put("q1", w, RewardComponents(0.9, 1.0, 0.9, 1.0))
    assert cache.get("q1", w).f1 == 0.9


def test_has_reflects_presence(cache: OfflineCache) -> None:
    w = DWAWeights(1.0, 0.0, 0.0)
    assert not cache.has("q1", w)
    cache.put("q1", w, RewardComponents(0.5, 0.5, 0.5, 1.0))
    assert cache.has("q1", w)


def test_put_many(cache: OfflineCache) -> None:
    rows = [
        ("q1", DWAWeights(1.0, 0.0, 0.0), RewardComponents(0.1, 0.1, 0.1, 1.0)),
        ("q2", DWAWeights(0.5, 0.5, 0.0), RewardComponents(0.5, 0.5, 0.5, 1.0)),
        ("q3", DWAWeights(0.0, 0.0, 1.0), RewardComponents(0.9, 0.9, 0.9, 1.0)),
    ]
    cache.put_many(rows)
    for qid, w, r in rows:
        assert cache.get(qid, w) == r


def test_stats_empty(cache: OfflineCache) -> None:
    s = cache.stats()
    assert s["total_entries"] == 0
    assert s["n_queries"] == 0


def test_stats_after_inserts(cache: OfflineCache) -> None:
    cache.put("q1", DWAWeights(1.0, 0.0, 0.0), RewardComponents(0.8, 1.0, 0.9, 2.0))
    cache.put("q1", DWAWeights(0.5, 0.5, 0.0), RewardComponents(0.6, 0.5, 0.7, 3.0))
    cache.put("q2", DWAWeights(0.5, 0.5, 0.0), RewardComponents(0.4, 0.0, 0.5, 4.0))
    s = cache.stats()
    assert s["total_entries"] == 3
    assert s["n_queries"] == 2
    assert s["avg_f1"] == pytest.approx((0.8 + 0.6 + 0.4) / 3)


# ---------- build ----------

def _fake_reward_fn(qid: str, qtext: str, weights: DWAWeights) -> RewardComponents:
    """Deterministic reward: F1 favors α dominant; EM favors β dominant."""
    return RewardComponents(
        f1=weights.alpha,
        em=weights.beta,
        faithfulness=weights.gamma,
        latency=1.0,
    )


def test_build_grid_2_writes_3_queries_times_6_combos(cache: OfflineCache) -> None:
    queries = [("q1", "text1"), ("q2", "text2"), ("q3", "text3")]
    written = cache.build(queries, _fake_reward_fn, grid=2)
    assert written == 3 * 6  # 18
    s = cache.stats()
    assert s["total_entries"] == 18
    assert s["n_queries"] == 3


def test_build_skip_existing_does_not_recompute(cache: OfflineCache) -> None:
    queries = [("q1", "text1")]
    cache.build(queries, _fake_reward_fn, grid=2)
    second_write = cache.build(queries, _fake_reward_fn, grid=2, skip_existing=True)
    assert second_write == 0  # nothing new to write


def test_build_progress_callback_called(cache: OfflineCache) -> None:
    seen: list[tuple[int, int]] = []
    cache.build(
        [("q1", "text1")],
        _fake_reward_fn,
        grid=2,
        on_progress=lambda done, total: seen.append((done, total)),
    )
    assert seen
    final_done, final_total = seen[-1]
    assert final_done == final_total == 6


def test_build_with_workers_parallel(cache: OfflineCache) -> None:
    queries = [(f"q{i}", f"text{i}") for i in range(5)]
    written = cache.build(queries, _fake_reward_fn, grid=2, n_workers=4)
    assert written == 5 * 6  # 30
    # Verify a known pair
    r = cache.get("q3", DWAWeights(0.5, 0.5, 0.0))
    assert r is not None
    assert r.f1 == pytest.approx(0.5)
    assert r.em == pytest.approx(0.5)


# ---------- context manager ----------

def test_context_manager_closes(tmp_path: Path) -> None:
    with OfflineCache(tmp_path / "ctx.sqlite") as c:
        c.put("q1", DWAWeights(1.0, 0.0, 0.0), RewardComponents(0.5, 0.5, 0.5, 1.0))
    # After close, a new instance can read the same DB
    with OfflineCache(tmp_path / "ctx.sqlite") as c2:
        assert c2.get("q1", DWAWeights(1.0, 0.0, 0.0)) is not None
