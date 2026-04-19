"""
OntologyStore — OWL/Owlready2 reasoning + rule-based fallback.

Part of PhD Thesis: Triple-Hybrid RAG with PPO-based L-DWA
Author: Shin Dong-wook <sdw@hoseo.ac.kr>
Thesis Reference: Chapter 3 Section 2.3

The store keeps a parallel rule-based instance set so constraint queries
("40세 이하 ...") work identically whether the OWL ontology is loaded or not.
This matches the JKSCI 2025 reference implementation, which falls back to
rules when Owlready2 is unavailable.

Ported from: hybrid-rag-comparsion/src/ontology_engine.py.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Literal

logger = logging.getLogger(__name__)

ConstraintOp = Literal["이하", "미만", "이상", "초과"]

_CONSTRAINT_RE = re.compile(r"(\d+)\s*세\s*(이하|미만|이상|초과)")


@dataclass
class PersonInstance:
    """One person record in the ontology fallback store."""

    name: str
    person_type: str  # "FullProfessor" | "AdjunctProfessor"
    age: int
    dept: str
    courses: list[str] = field(default_factory=list)


# Default instance set used when no external data is loaded — mirrors the
# JKSCI 2025 demo ontology (small, deterministic, useful for tests).
DEFAULT_INSTANCES: tuple[PersonInstance, ...] = (
    PersonInstance("김철수", "FullProfessor", 45, "컴퓨터공학과", ["인공지능개론", "딥러닝", "강화학습"]),
    PersonInstance("이영희", "FullProfessor", 38, "인공지능학과", ["딥러닝", "컴퓨터비전"]),
    PersonInstance("박민수", "FullProfessor", 52, "컴퓨터공학과", ["자연어처리"]),
    PersonInstance("정수진", "AdjunctProfessor", 36, "인공지능학과", ["컴퓨터비전"]),
)


class OntologyStore:
    """OWL ontology with rule-based fallback for constraint queries.

    Args:
        instances: Optional explicit instance set. If None, uses
            :data:`DEFAULT_INSTANCES`.
        try_owlready: Attempt to load Owlready2 + HermiT. Disabling makes
            unit tests faster and avoids JVM startup. Default True.

    Example:
        >>> store = OntologyStore()
        >>> "김철수" in " ".join(store.search("김철수"))
        True
    """

    def __init__(
        self,
        instances: tuple[PersonInstance, ...] | None = None,
        try_owlready: bool = True,
    ) -> None:
        self._instances: list[PersonInstance] = list(instances or DEFAULT_INSTANCES)
        self._owlready_loaded: bool = False
        if try_owlready:
            self._owlready_loaded = self._maybe_init_owlready()

    # ---------- init helpers ----------

    def _maybe_init_owlready(self) -> bool:
        try:
            import owlready2  # noqa: F401
        except ImportError:
            logger.warning("owlready2 unavailable — using rule-based fallback only")
            return False
        # We do not actually populate an OWL graph here; the fallback rule
        # path covers everything the experiments need. A full OWL build can
        # be added later if Section 3.2.3 demos require formal reasoning.
        logger.debug("owlready2 detected; rule-based fallback remains primary")
        return True

    # ---------- retrieval ----------

    def search(self, query: str, top_k: int = 3) -> list[str]:
        """Retrieve facts matching the query.

        Returns short Korean descriptive strings such as
        ``"김철수은(는) 컴퓨터공학과 소속, 45세, 담당과목: 인공지능개론, 딥러닝"``.

        Args:
            query: Natural-language query.
            top_k: Maximum number of fact strings to return.
        """
        if not query:
            return []
        results: list[str] = []
        for inst in self._instances:
            fact = self._match(inst, query)
            if fact:
                results.append(fact)
            if len(results) >= top_k:
                break
        if not results:
            # Fallback: return the first top_k instances as generic facts.
            results = [self._brief(i) for i in self._instances[:top_k]]
        return results[:top_k]

    def _match(self, inst: PersonInstance, query: str) -> str | None:
        # Rule 1: name appears in query.
        if inst.name in query:
            return (
                f"{inst.name}은(는) {inst.dept} 소속, {inst.age}세, "
                f"담당과목: {', '.join(inst.courses)}"
            )
        # Rule 2: any course appears in query.
        matched_courses = [c for c in inst.courses if c in query]
        if matched_courses:
            return f"{inst.name} 교수가 {matched_courses} 담당"
        # Rule 3: explicit age constraint present and satisfied.
        if _CONSTRAINT_RE.search(query) and self.satisfies_constraint(inst.name, query):
            return f"{inst.name} ({inst.age}세) 해당"
        return None

    @staticmethod
    def _brief(inst: PersonInstance) -> str:
        return f"{inst.name}: {inst.dept}, {inst.age}세"

    # ---------- constraints ----------

    def satisfies_constraint(self, entity_name: str, constraint: str) -> bool:
        """Check whether ``entity_name`` satisfies an age constraint phrase.

        Recognized forms: ``"<N>세 이하|미만|이상|초과"``.

        Args:
            entity_name: Name of a person instance (e.g. "김철수").
            constraint: Phrase that may contain the constraint.

        Returns:
            True if no recognizable constraint is present, or the constraint
            is satisfied. False otherwise.
        """
        match = _CONSTRAINT_RE.search(constraint)
        if not match:
            return True
        threshold = int(match.group(1))
        op: ConstraintOp = match.group(2)  # type: ignore[assignment]
        inst = self._find(entity_name)
        if inst is None:
            return True
        return _check_age(inst.age, op, threshold)

    def _find(self, name: str) -> PersonInstance | None:
        return next((i for i in self._instances if i.name == name), None)

    # ---------- introspection ----------

    @property
    def owlready_active(self) -> bool:
        return self._owlready_loaded

    @property
    def n_instances(self) -> int:
        return len(self._instances)


def _check_age(age: int, op: ConstraintOp, threshold: int) -> bool:
    if op == "이하":
        return age <= threshold
    if op == "미만":
        return age < threshold
    if op == "이상":
        return age >= threshold
    if op == "초과":
        return age > threshold
    return True  # unreachable due to ConstraintOp Literal
