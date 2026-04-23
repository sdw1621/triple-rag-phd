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

    # Korean entity patterns (원래 연구 domain: 합성 대학 행정)
    ENTITY_PATTERNS: tuple[str, ...] = (
        r"[가-힣]{2,4}\s*교수",
        r"[가-힣]{2,4}\s*학생",
        r"[가-힣]+(?:공학과|학과|학부|대학원)",
        r"[가-힣]+(?:프로젝트|연구|과목|수업)",
        r"(?:AI|ML|NLP|CV|DL|RL|RAG)",
    )

    # English entity patterns — added 2026-04-22 to enable cross-lingual
    # transfer tests. Covers two axes:
    #   (a) university-domain English (for English-synthetic benchmark of
    #       L-DWA): "Prof X", "Department of X", "X course", ...
    #   (b) generic English proper-noun patterns (for HotpotQA/MuSiQue):
    #       capitalized multi-word entities and common question forms.
    # Keeping Korean patterns side-by-side so bilingual queries still work.
    ENTITY_PATTERNS_EN: tuple[str, ...] = (
        # Academic domain
        r"[Pp]rof\.?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*",
        r"[Pp]rofessor\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*",
        r"[Dd]epartment\s+of\s+[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*",
        r"(?:[A-Z][a-zA-Z]+\s+)?[Dd]epartment",
        r"[A-Z][a-zA-Z]+\s+(?:course|class|seminar|lab|project)",
        # Generic proper-noun sequences (2–4 capitalized words)
        r"\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\b",
        # Common generic topics (HotpotQA style)
        r"\b(?:magazine|novel|film|movie|album|song|book|band|company|city|country|river|mountain)\b",
    )

    RELATION_KEYWORDS: tuple[str, ...] = (
        "소속", "협력", "담당", "참여", "지도",
        "가르치는", "수강", "공동", "연구하는", "담당하는",
    )

    # English relation keywords — added 2026-04-22 for cross-lingual tests.
    RELATION_KEYWORDS_EN: tuple[str, ...] = (
        # Academic-domain relations
        "teaches", "teaching", "belongs", "affiliated", "supervises",
        "advises", "collaborates", "participates", "leads",
        # Generic relations (HotpotQA/MuSiQue)
        "born", "founded", "located", "directed", "wrote", "composed",
        "starred", "published", "produced", "performed", "married",
        "died", "graduated", "acted", "sang", "played",
    )

    CONSTRAINT_PATTERNS: tuple[str, ...] = (
        r"\d+\s*세\s*(?:이하|미만|이상|초과)",
        r"(?:제외|excluding|except)",
        r"(?:이상|이하|초과|미만)",
        r"(?:and|or|AND|OR|그리고|또는)",
    )

    # English constraint patterns — added 2026-04-22.
    CONSTRAINT_PATTERNS_EN: tuple[str, ...] = (
        r"(?:less|more|older|younger|bigger|smaller|greater|fewer)\s+than",
        r"(?:at\s+least|at\s+most|no\s+more\s+than|no\s+less\s+than)",
        r"(?:excluding|except|not\s+including|other\s+than)",
        r"(?:before|after|between|during)",
        r"(?:first|last|earliest|latest|oldest|youngest)",
        r"\b(?:which|who|what|where|when|why|how)\b.*\b(?:started|founded|earlier|later|first|last)\b",
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
        for pattern in self.ENTITY_PATTERNS_EN:
            found.extend(re.findall(pattern, query))
        return list(dict.fromkeys(found))

    def _extract_relations(self, query: str) -> list[str]:
        q_lower = query.lower()
        ko = [kw for kw in self.RELATION_KEYWORDS if kw in query]
        en = [kw for kw in self.RELATION_KEYWORDS_EN if kw in q_lower]
        return list(dict.fromkeys(ko + en))

    def _extract_constraints(self, query: str) -> list[str]:
        found: list[str] = []
        for pattern in self.CONSTRAINT_PATTERNS:
            found.extend(re.findall(pattern, query))
        for pattern in self.CONSTRAINT_PATTERNS_EN:
            found.extend(re.findall(pattern, query, flags=re.IGNORECASE))
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
