"""
Evaluation metrics: F1 / EM / Recall@k / Precision / Faithfulness.

Part of PhD Thesis: Triple-Hybrid RAG with PPO-based L-DWA
Author: Shin Dong-wook <sdw@hoseo.ac.kr>
Thesis Reference: Chapter 6 Section 1 (definitions), Table 6-1 (formulas)

Korean-aware EM normalization:
  Unicode NFC → lowercase → whitespace squeeze → Korean particle stripping
  → Sino-Korean numeral mapping (일/이/삼... → 1/2/3...).

Ported from: hybrid-rag-comparsion/src/evaluator.py — split into pure
functions (was a monolithic Evaluator class) so the PPO reward function and
the offline cache can call individual metrics directly.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import asdict, dataclass

logger = logging.getLogger(__name__)

# Korean postpositions/particles to strip during EM normalization. Order
# matters — longer suffixes first to prevent prefix matching.
_KO_PARTICLES: tuple[str, ...] = (
    "에서", "으로", "에게", "한테", "까지", "부터",
    "은", "는", "이", "가", "을", "를", "의", "에", "로",
    "와", "과", "도", "만", "께",
)

# Sino-Korean numeral → Arabic. (1-10 only; sufficient for QA gold answers.)
_SINO_NUMERALS: dict[str, str] = {
    "일": "1", "이": "2", "삼": "3", "사": "4", "오": "5",
    "육": "6", "칠": "7", "팔": "8", "구": "9", "십": "10",
}

_PARTICLE_PATTERNS: tuple[tuple[str, "re.Pattern[str]"], ...] = tuple(
    (p, re.compile(rf"(?<=[가-힣]){p}(?=\s|$)"))
    for p in sorted(_KO_PARTICLES, key=len, reverse=True)
)

_WHITESPACE_RE = re.compile(r"\s+")
_SENTENCE_SPLIT_RE = re.compile(r"[.!?。]")

# Punctuation that must be stripped before token-set comparison so that list-form
# gold answers ("홍성민, 황성민, 전성민") match sentence-form predictions.
# Note: preserved in text through tokenization would inflate pred token count and
# mismatch gold tokens because commas attach to preceding tokens.
_PUNCT_RE = re.compile(r"[,.:;!?\"'\u201c\u201d\u2018\u2019()\[\]\{\}·・…―—ㆍ]")


# ---------- Normalization ----------

def normalize_korean(text: str) -> str:
    """Korean-aware normalization for EM/F1 comparison.

    Pipeline (thesis Sec 6.1):
        1. Unicode NFC.
        2. Lowercase.
        3. Whitespace squeeze.
        4. Punctuation stripping (commas / periods / brackets / quotes etc.)
           → ensures list-form gold ("A, B, C") tokenizes as {A, B, C}, not
           {A,, B,, C}. This was identified as a critical evaluator bug in
           thesis v2 → v4 (see `results/evaluator_fix_impact.md`).
        5. Korean particle stripping (after Hangul syllables, at word end).
        6. Sino-Korean numeral mapping (일→1, 이→2, ..., 십→10).

    Args:
        text: Raw string.

    Returns:
        Normalized string suitable for direct equality / token-set comparison.
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text)
    text = text.lower()
    text = _WHITESPACE_RE.sub(" ", text).strip()
    text = _PUNCT_RE.sub(" ", text)  # strip punctuation BEFORE particle regex
    text = _WHITESPACE_RE.sub(" ", text).strip()
    for _, pattern in _PARTICLE_PATTERNS:
        text = pattern.sub("", text)
    for ko, ar in _SINO_NUMERALS.items():
        text = text.replace(ko, ar)
    return text.strip()


# ---------- Single-item metrics ----------

def exact_match(pred: str, gold: str, normalize: bool = True) -> float:
    """Exact match in {0.0, 1.0}.

    Args:
        pred: Predicted answer.
        gold: Reference answer.
        normalize: If True, apply :func:`normalize_korean` first.
    """
    if normalize:
        return 1.0 if normalize_korean(pred) == normalize_korean(gold) else 0.0
    return 1.0 if pred.strip() == gold.strip() else 0.0


def f1_score(pred: str, gold: str) -> float:
    """Token-level F1 over normalized token sets (strict).

    Returns:
        F1 in [0, 1]. Returns 0.0 when either side is empty or there is no
        token overlap.
    """
    pred_toks = set(normalize_korean(pred).split())
    gold_toks = set(normalize_korean(gold).split())
    if not pred_toks or not gold_toks:
        return 0.0
    common = pred_toks & gold_toks
    if not common:
        return 0.0
    p = len(common) / len(pred_toks)
    r = len(common) / len(gold_toks)
    return 2.0 * p * r / (p + r)


def f1_substring(pred: str, gold: str) -> float:
    """Substring-overlap F1 for comma-separated-list gold answers.

    Design: gold answers in the JKSCI 2025 benchmark are often comma-separated
    name lists like "홍성민, 황성민, 전성민". The strict token-set F1 scores
    such answers near zero when the LLM emits sentence-form output because
    the token sets don't overlap after Korean particle stripping.

    Substring F1 splits gold by commas into "target items" and checks how
    many items appear as substrings in the (normalized) prediction.
    Precision = # of gold items found / # of (naively split) pred phrases;
    Recall    = # of gold items found / # of gold items.

    This matches the looser evaluation assumed in the original JKSCI 2025
    paper and is reported alongside :func:`f1_score` (strict) in Ch.6 tables.
    """
    if not pred or not gold:
        return 0.0
    gold_items = [normalize_korean(x) for x in gold.split(",") if normalize_korean(x)]
    if not gold_items:
        return 0.0
    pred_norm = normalize_korean(pred)
    # Pred items: split on comma/whitespace/Korean particle boundaries is messy;
    # simplest robust approach is to count substring hits against gold items.
    # For precision, we approximate pred item count via the number of distinct
    # gold items that appear (bounded by |gold|), which gives precision == recall
    # and therefore F1 == recall. This is a common convention for list-valued
    # answers and matches how EM-on-lists is often evaluated.
    hits = sum(1 for item in gold_items if item and item in pred_norm)
    recall = hits / len(gold_items)
    # Precision: count "distinct gold items" that hit, over the number of
    # comma-separated chunks in the prediction. If the prediction has no
    # commas, treat it as a single phrase (precision = hits / 1).
    pred_chunks = [normalize_korean(x) for x in pred.split(",") if normalize_korean(x)]
    n_pred = max(1, len(pred_chunks))
    precision = hits / max(n_pred, len(gold_items))
    # When precision is driven by list-chunk count, use harmonic mean.
    if precision + recall == 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def recall_at_k(retrieved: list[str], gold: str, k: int = 3) -> float:
    """Recall@k: 1.0 iff the gold answer appears (substring) in the top-k docs."""
    gold_norm = normalize_korean(gold)
    if not gold_norm:
        return 0.0
    for doc in retrieved[:k]:
        if gold_norm in normalize_korean(doc):
            return 1.0
    return 0.0


def precision(retrieved: list[str], gold: str) -> float:
    """Fraction of retrieved docs that contain the gold answer (substring)."""
    if not retrieved:
        return 0.0
    gold_norm = normalize_korean(gold)
    if not gold_norm:
        return 0.0
    relevant = sum(1 for d in retrieved if gold_norm in normalize_korean(d))
    return relevant / len(retrieved)


def faithfulness(answer: str, contexts: list[str]) -> float:
    """RAGAS-style faithfulness approximation (no LLM call).

    For each sentence in ``answer``, check whether at least one non-trivial
    token (length ≥ 2 after normalization) appears in the joined contexts.
    Sentence support ratio is returned.

    Notes:
        This is a fast proxy used during PPO offline-cache build. The full
        RAGAS LLM-judged faithfulness is computed separately for thesis
        Table 6-2 final numbers.
    """
    if not contexts or not answer:
        return 0.0
    sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(answer) if s.strip()]
    if not sentences:
        return 0.0
    ctx_combined = normalize_korean(" ".join(contexts))
    supported = 0
    for sent in sentences:
        toks = [t for t in normalize_korean(sent).split() if len(t) > 1]
        if any(t in ctx_combined for t in toks):
            supported += 1
    return supported / len(sentences)


# ---------- Aggregate ----------

@dataclass
class EvalResult:
    """Per-query evaluation bundle."""

    f1: float
    em_raw: float
    em_norm: float
    recall_at_k: float
    precision: float
    faithfulness: float

    def as_dict(self) -> dict[str, float]:
        return {k: round(v, 4) for k, v in asdict(self).items()}


def evaluate_single(
    pred: str,
    gold: str,
    retrieved_docs: list[str],
    contexts: list[str],
    k: int = 3,
) -> EvalResult:
    """Compute all single-item metrics for one (pred, gold, contexts) sample.

    Args:
        pred: Predicted answer.
        gold: Reference answer.
        retrieved_docs: Documents returned by retrieval (used for recall/precision).
        contexts: Final contexts passed to the LLM (used for faithfulness).
        k: Top-k cutoff for recall@k.
    """
    return EvalResult(
        f1=f1_score(pred, gold),
        em_raw=exact_match(pred, gold, normalize=False),
        em_norm=exact_match(pred, gold, normalize=True),
        recall_at_k=recall_at_k(retrieved_docs, gold, k=k),
        precision=precision(retrieved_docs, gold),
        faithfulness=faithfulness(pred, contexts),
    )
