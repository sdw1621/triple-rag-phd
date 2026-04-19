"""
Rule-based query intent analyzer (baseline).

Part of PhD Thesis: Triple-Hybrid RAG with PPO-based L-DWA
Author: Shin Dong-wook <sdw@hoseo.ac.kr>
Thesis Reference: Chapter 3 Section 3, Chapter 4 (R-DWA inputs)

Ported from: hybrid-rag-comparsion/src/query_analyzer.py (JKSCI 2025).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Literal

logger = logging.getLogger(__name__)

QueryType = Literal["simple", "multi_hop", "conditional"]


@dataclass
class QueryIntent:
    """Analyzed query intent.

    Attributes:
        query_type: One of "simple", "multi_hop", "conditional".
        entities: Detected entity surface forms.
        relations: Detected relation keywords.
        constraints: Detected constraint phrases (age, exclusion, logical ops).
        complexity_score: Aggregate complexity in [0, 1].
        density: (s_e, s_r, s_c) entity/relation/constraint densities in [0, 1]^3.
            Used as part of the 18-dim PPO state vector (thesis Eq. 5-1).
    """

    query_type: QueryType
    entities: list[str]
    relations: list[str]
    constraints: list[str]
    complexity_score: float
    density: tuple[float, float, float] = (0.0, 0.0, 0.0)

    @property
    def s_e(self) -> float:
        """Entity density (thesis Eq. 4-2)."""
        return self.density[0]

    @property
    def s_r(self) -> float:
        """Relation density (thesis Eq. 4-3)."""
        return self.density[1]

    @property
    def s_c(self) -> float:
        """Constraint density (thesis Eq. 4-4)."""
        return self.density[2]


class RuleBasedIntent:
    """Rule-based intent classifier.

    Uses regex patterns and a keyword list to extract entities, relations, and
    constraints from a Korean query, then classifies into one of three types.

    Example:
        >>> analyzer = RuleBasedIntent()
        >>> intent = analyzer.analyze("김철수 교수는 어느 학과 소속인가?")
        >>> intent.query_type
        'simple'
    """

    N_MAX_ENTITY: int = 5
    N_MAX_RELATION: int = 4
    N_MAX_CONSTRAINT: int = 3

    ENTITY_PATTERNS: tuple[str, ...] = (
        r"[가-힣]{2,4}\s*교수",
        r"[가-힣]{2,4}\s*학생",
        r"[가-힣]+(?:공학과|학과|학부|대학원)",
        r"[가-힣]+(?:프로젝트|연구|과목|수업)",
        r"(?:AI|ML|NLP|CV|DL|RL|RAG)",
    )
    RELATION_KEYWORDS: tuple[str, ...] = (
        "소속", "협력", "담당", "참여", "지도",
        "가르치는", "수강", "공동", "연구하는", "담당하는",
    )
    CONSTRAINT_PATTERNS: tuple[str, ...] = (
        r"\d+\s*세\s*(?:이하|미만|이상|초과)",
        r"(?:제외|excluding|except)",
        r"(?:이상|이하|초과|미만)",
        r"(?:and|or|AND|OR|그리고|또는)",
    )

    def __init__(self) -> None:
        logger.debug("Initialized RuleBasedIntent")

    def analyze(self, query: str) -> QueryIntent:
        """Analyze a Korean query string.

        Args:
            query: Natural-language query.

        Returns:
            QueryIntent with type, entities/relations/constraints, and densities.
        """
        entities = self._extract_entities(query)
        relations = self._extract_relations(query)
        constraints = self._extract_constraints(query)
        query_type = self._classify(entities, relations, constraints)

        complexity = min(
            (len(entities) * 0.3 + len(relations) * 0.4 + len(constraints) * 0.3) / 3.0,
            1.0,
        )
        s_e = min(len(entities) / self.N_MAX_ENTITY, 1.0)
        s_r = min(len(relations) / self.N_MAX_RELATION, 1.0)
        s_c = min(len(constraints) / self.N_MAX_CONSTRAINT, 1.0)

        return QueryIntent(
            query_type=query_type,
            entities=entities,
            relations=relations,
            constraints=constraints,
            complexity_score=complexity,
            density=(s_e, s_r, s_c),
        )

    def _extract_entities(self, query: str) -> list[str]:
        found: list[str] = []
        for pattern in self.ENTITY_PATTERNS:
            found.extend(re.findall(pattern, query))
        return list(dict.fromkeys(found))

    def _extract_relations(self, query: str) -> list[str]:
        return [kw for kw in self.RELATION_KEYWORDS if kw in query]

    def _extract_constraints(self, query: str) -> list[str]:
        found: list[str] = []
        for pattern in self.CONSTRAINT_PATTERNS:
            found.extend(re.findall(pattern, query))
        return list(dict.fromkeys(found))

    @staticmethod
    def _classify(
        entities: list[str], relations: list[str], constraints: list[str]
    ) -> QueryType:
        if constraints:
            return "conditional"
        if len(entities) >= 2 or len(relations) >= 2:
            return "multi_hop"
        return "simple"
