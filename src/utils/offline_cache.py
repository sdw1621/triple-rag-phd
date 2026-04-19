"""
Offline reward cache (SQLite) for PPO training.

Part of PhD Thesis: Triple-Hybrid RAG with PPO-based L-DWA
Author: Shin Dong-wook <sdw@hoseo.ac.kr>
Thesis Reference: Chapter 5 Section 4 (cache design),
                  Chapter 6 Section 10 (cost analysis)

Cost rationale (thesis Sec 6.10): a single PPO seed takes 10,000 episodes ×
32 rollout steps ≈ 320K reward queries. Online evaluation costs ~$0.0003 each
(GPT-4o-mini), so naive training is ~$96 per seed × 3 seeds = ~$288. By
discretizing weights to a 0.1 grid (66 simplex points) and pre-computing
rewards once per (query, weight) pair we get:

    5,000 QA × 66 weights = 330,000 entries → ~$15 one-time, then $0/seed.

Discretization: ``round(w / 0.1)`` ∈ {0..10} per dimension; valid combos
satisfy ``a+b+g == 10`` (66 of them = C(12, 2)).

Schema:
    rewards(
        query_id TEXT, alpha_int INT, beta_int INT, gamma_int INT,
        f1 REAL, em REAL, faithfulness REAL, latency REAL,
        PRIMARY KEY (query_id, alpha_int, beta_int, gamma_int)
    )
"""

from __future__ import annotations

import logging
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, Sequence

from src.dwa.base import DWAWeights

logger = logging.getLogger(__name__)

DEFAULT_STEP: float = 0.1
DEFAULT_GRID: int = 10  # 1.0 / 0.1
SCHEMA_VERSION: int = 1


@dataclass(frozen=True)
class RewardComponents:
    """Per-(query, weight) reward decomposition (thesis Eq. 5-7 inputs)."""

    f1: float
    em: float
    faithfulness: float
    latency: float

    def total_reward(self) -> float:
        """Apply thesis Eq. 5-7."""
        return (
            0.5 * self.f1
            + 0.3 * self.em
            + 0.2 * self.faithfulness
            - 0.1 * max(0.0, self.latency - 5.0)
        )


def discretize(value: float, step: float = DEFAULT_STEP) -> int:
    """Discretize a weight in [0, 1] to an integer grid index."""
    if not 0.0 <= value <= 1.0 + 1e-9:
        raise ValueError(f"value out of [0, 1]: {value}")
    return round(value / step)


def discretize_weights(weights: DWAWeights, step: float = DEFAULT_STEP) -> tuple[int, int, int]:
    """Discretize a :class:`DWAWeights` to integer grid indices."""
    return (discretize(weights.alpha, step), discretize(weights.beta, step), discretize(weights.gamma, step))


def enumerate_simplex(grid: int = DEFAULT_GRID) -> Iterator[tuple[int, int, int]]:
    """Yield all (a, b, g) with a+b+g == grid and each in [0, grid].

    For ``grid=10`` this produces ``C(12, 2) = 66`` triplets.
    """
    for a in range(grid + 1):
        for b in range(grid + 1 - a):
            yield (a, b, grid - a - b)


def simplex_size(grid: int = DEFAULT_GRID) -> int:
    """Number of integer simplex points: (grid+1)(grid+2)/2."""
    return (grid + 1) * (grid + 2) // 2


# ---------- callback type ----------

# (query_id, query_text, weights) → RewardComponents
RewardFn = Callable[[str, str, DWAWeights], RewardComponents]


class OfflineCache:
    """SQLite-backed reward cache with optional parallel build.

    Args:
        db_path: SQLite file path. Created with ``WAL`` mode for concurrent
            writers.

    Example:
        >>> cache = OfflineCache(":memory:")
        >>> cache.put("q1", DWAWeights(0.5, 0.3, 0.2),
        ...           RewardComponents(0.8, 1.0, 0.9, 1.2))
        >>> cache.get("q1", DWAWeights(0.5, 0.3, 0.2)).f1
        0.8
    """

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = str(db_path)
        # check_same_thread=False so worker threads can share the connection;
        # we serialize writes via a lock.
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._lock = threading.Lock()
        self._init_schema()

    def _init_schema(self) -> None:
        with self._lock:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS rewards (
                    query_id    TEXT NOT NULL,
                    alpha_int   INTEGER NOT NULL,
                    beta_int    INTEGER NOT NULL,
                    gamma_int   INTEGER NOT NULL,
                    f1          REAL NOT NULL,
                    em          REAL NOT NULL,
                    faithfulness REAL NOT NULL,
                    latency     REAL NOT NULL,
                    PRIMARY KEY (query_id, alpha_int, beta_int, gamma_int)
                );
                CREATE TABLE IF NOT EXISTS meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                """
            )
            self._conn.execute(
                "INSERT OR IGNORE INTO meta (key, value) VALUES ('schema_version', ?)",
                (str(SCHEMA_VERSION),),
            )
            self._conn.commit()

    # ---------- CRUD ----------

    def put(self, query_id: str, weights: DWAWeights, reward: RewardComponents) -> None:
        """Insert or overwrite a single (query, weights) → reward entry."""
        a, b, g = discretize_weights(weights)
        with self._lock:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO rewards
                  (query_id, alpha_int, beta_int, gamma_int, f1, em, faithfulness, latency)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (query_id, a, b, g, reward.f1, reward.em, reward.faithfulness, reward.latency),
            )
            self._conn.commit()

    def put_many(
        self, rows: Sequence[tuple[str, DWAWeights, RewardComponents]]
    ) -> None:
        """Batch insert/overwrite for higher throughput."""
        if not rows:
            return
        flat = [
            (qid, *discretize_weights(w), r.f1, r.em, r.faithfulness, r.latency)
            for qid, w, r in rows
        ]
        with self._lock:
            self._conn.executemany(
                """
                INSERT OR REPLACE INTO rewards
                  (query_id, alpha_int, beta_int, gamma_int, f1, em, faithfulness, latency)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                flat,
            )
            self._conn.commit()

    def get(self, query_id: str, weights: DWAWeights) -> RewardComponents | None:
        """Look up a single entry; returns None on miss."""
        a, b, g = discretize_weights(weights)
        with self._lock:
            cur = self._conn.execute(
                """
                SELECT f1, em, faithfulness, latency FROM rewards
                WHERE query_id=? AND alpha_int=? AND beta_int=? AND gamma_int=?
                """,
                (query_id, a, b, g),
            )
            row = cur.fetchone()
        if row is None:
            return None
        return RewardComponents(f1=row[0], em=row[1], faithfulness=row[2], latency=row[3])

    def has(self, query_id: str, weights: DWAWeights) -> bool:
        return self.get(query_id, weights) is not None

    # ---------- build ----------

    def build(
        self,
        queries: Sequence[tuple[str, str]],
        reward_fn: RewardFn,
        grid: int = DEFAULT_GRID,
        n_workers: int = 1,
        skip_existing: bool = True,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> int:
        """Pre-compute rewards for ``queries × simplex(grid)``.

        Args:
            queries: List of ``(query_id, query_text)`` pairs.
            reward_fn: Callable producing a :class:`RewardComponents` for one
                (query_id, query_text, weights). Typically wraps the full
                :class:`TripleHybridRAG` query + metric evaluation.
            grid: Discretization grid (1/step). Default 10.
            n_workers: Thread pool size. Default 1 (single-threaded). Use >1
                only if ``reward_fn`` is I/O bound (e.g. LLM API calls).
            skip_existing: Skip (qid, w) pairs already in the DB.
            on_progress: Optional ``callback(done, total)`` invoked once per
                completed task.

        Returns:
            Number of new entries written.
        """
        step = 1.0 / grid
        combos = list(enumerate_simplex(grid))
        total = len(queries) * len(combos)
        tasks: list[tuple[str, str, DWAWeights]] = []

        for qid, qtext in queries:
            for a, b, g in combos:
                weights = DWAWeights(a * step, b * step, g * step)
                if skip_existing and self.has(qid, weights):
                    continue
                tasks.append((qid, qtext, weights))

        if not tasks:
            if on_progress:
                on_progress(total, total)
            return 0

        written = 0
        if n_workers <= 1:
            for i, (qid, qtext, w) in enumerate(tasks):
                reward = reward_fn(qid, qtext, w)
                self.put(qid, w, reward)
                written += 1
                if on_progress:
                    on_progress(i + 1, len(tasks))
        else:
            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                futures = {
                    pool.submit(reward_fn, qid, qtext, w): (qid, w)
                    for qid, qtext, w in tasks
                }
                for i, fut in enumerate(as_completed(futures)):
                    qid, w = futures[fut]
                    reward = fut.result()
                    self.put(qid, w, reward)
                    written += 1
                    if on_progress:
                        on_progress(i + 1, len(tasks))
        return written

    # ---------- introspection ----------

    def stats(self) -> dict[str, float]:
        """Return basic cache statistics."""
        with self._lock:
            cur = self._conn.execute(
                "SELECT COUNT(*), AVG(f1), AVG(em), AVG(faithfulness), AVG(latency) FROM rewards"
            )
            n, avg_f1, avg_em, avg_faith, avg_lat = cur.fetchone()
            cur2 = self._conn.execute("SELECT COUNT(DISTINCT query_id) FROM rewards")
            n_queries = cur2.fetchone()[0]
        return {
            "total_entries": int(n or 0),
            "n_queries": int(n_queries or 0),
            "avg_f1": float(avg_f1) if avg_f1 is not None else 0.0,
            "avg_em": float(avg_em) if avg_em is not None else 0.0,
            "avg_faithfulness": float(avg_faith) if avg_faith is not None else 0.0,
            "avg_latency": float(avg_lat) if avg_lat is not None else 0.0,
        }

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self) -> "OfflineCache":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
