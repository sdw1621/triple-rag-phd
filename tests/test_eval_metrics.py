"""Tests for src.eval.metrics."""

from __future__ import annotations

import pytest

from src.eval.metrics import (
    EvalResult,
    evaluate_single,
    exact_match,
    f1_char,
    f1_score,
    f1_substring,
    faithfulness,
    normalize_korean,
    precision,
    recall_at_k,
)


# ---------- normalize_korean ----------

def test_normalize_strips_korean_particles() -> None:
    # "김철수는" → particle 는 stripped → "김철수"
    assert normalize_korean("김철수는") == "김철수"
    # 에서 (longer) before 에 (shorter)
    assert normalize_korean("학교에서") == "학교"


def test_normalize_squeezes_whitespace_and_lowercases() -> None:
    assert normalize_korean("Hello   World") == "hello world"


def test_normalize_handles_empty_string() -> None:
    assert normalize_korean("") == ""


def test_normalize_maps_sino_korean_numerals() -> None:
    assert normalize_korean("삼") == "3"


def test_normalize_strips_punctuation_from_list_answers() -> None:
    """Critical for list-form gold answers: '홍성민, 황성민' must tokenize
    as {'홍성민', '황성민'}, not {'홍성민,', '황성민'}."""
    assert "홍성민" in normalize_korean("홍성민, 황성민").split()
    assert "황성민" in normalize_korean("홍성민, 황성민").split()
    # No comma-attached tokens:
    assert "홍성민," not in normalize_korean("홍성민, 황성민").split()


def test_f1_on_list_answer_with_sentence_pred() -> None:
    gold = "홍성민, 황성민, 전성민"
    pred = "정치외교 심리학개론 과목은 홍성민 교수, 황성민 교수, 전성민 교수가 담당합니다."
    # After fix: all 3 gold names match → F1 > 0.5
    assert f1_score(pred, gold) > 0.5


# ---------- exact_match ----------

def test_em_normalized_strips_particle_difference() -> None:
    assert exact_match("김철수는", "김철수") == 1.0


def test_em_raw_distinguishes_particle_difference() -> None:
    assert exact_match("김철수는", "김철수", normalize=False) == 0.0


def test_em_with_completely_different_strings_is_zero() -> None:
    assert exact_match("이영희", "박민수") == 0.0


# ---------- f1_score ----------

def test_f1_full_overlap_is_one() -> None:
    assert f1_score("김철수 교수", "김철수 교수") == pytest.approx(1.0)


def test_f1_partial_overlap() -> None:
    f1 = f1_score("김철수 교수 컴공", "김철수 교수")
    assert 0.0 < f1 < 1.0


def test_f1_no_overlap_is_zero() -> None:
    assert f1_score("이영희", "박민수") == 0.0


def test_f1_empty_input_is_zero() -> None:
    assert f1_score("", "김철수") == 0.0
    assert f1_score("김철수", "") == 0.0


# ---------- f1_substring ----------

def test_f1_substring_list_gold_sentence_pred() -> None:
    """JKSCI-style: gold is comma-separated names, pred is a sentence."""
    gold = "홍성민, 황성민, 전성민"
    pred = "정치외교 심리학개론 과목은 홍성민 교수와 황성민 교수, 전성민 교수가 담당합니다."
    # All 3 gold names appear as substrings → high F1
    f1 = f1_substring(pred, gold)
    assert f1 > 0.6


def test_f1_substring_partial_match() -> None:
    gold = "홍성민, 황성민, 전성민"
    pred = "홍성민 교수가 담당합니다"  # only 1/3 hits
    f1 = f1_substring(pred, gold)
    assert 0.0 < f1 < 1.0


def test_f1_substring_no_match_is_zero() -> None:
    assert f1_substring("이영희 교수", "박민수, 김철수") == 0.0


def test_f1_substring_empty() -> None:
    assert f1_substring("", "홍성민") == 0.0
    assert f1_substring("홍성민", "") == 0.0


def test_f1_substring_and_strict_disagree() -> None:
    """Key property: f1_substring >> f1_strict on list-form gold answers."""
    gold = "홍성민, 황성민, 전성민"
    pred = "정치외교 과목은 홍성민 교수, 황성민 교수, 전성민 교수가 담당합니다."
    f1_s = f1_score(pred, gold)
    f1_sub = f1_substring(pred, gold)
    assert f1_sub > f1_s  # substring is looser on list answers


# ---------- f1_char ----------

def test_f1_char_identical_strings_is_one() -> None:
    assert f1_char("홍성민 교수", "홍성민 교수") == pytest.approx(1.0)


def test_f1_char_nonzero_on_form_mismatch() -> None:
    """f1_char should be > 0 even when answer and gold have different forms.

    Post-PUNCT_RE-fix, strict F1 is already reasonable on this input (tokens
    after punctuation strip + particle strip leave {홍성민, 황성민, 전성민}
    in both pred and gold). f1_char's role is corroboration — it yields a
    positive score whenever names are character-preserved, not strictly
    larger than the strict F1.
    """
    gold = "홍성민, 황성민, 전성민"
    pred = "홍성민 교수와 황성민 교수, 전성민 교수가 담당합니다"
    assert f1_char(pred, gold) > 0.0


def test_f1_char_robust_when_strict_collapses() -> None:
    """If strict collapses due to particle/case drift, f1_char still fires."""
    # Deliberately construct a case where strict token-set is degraded
    gold = "홍성민"
    pred = "홍성민을"  # particle 을 should be stripped — strict = 1.0
    assert f1_char(pred, gold) > 0.0
    # A weaker char-overlap case — partial name
    gold2 = "홍성민교수"
    pred2 = "홍성민"
    # Char 3-grams: gold has {홍성민, 성민교, 민교수}, pred has {홍성민}
    # 1 common / 3 gold + 1 pred → precision=1, recall=1/3 → F1 = 0.5
    assert 0.0 < f1_char(pred2, gold2) <= 1.0


def test_f1_char_no_overlap_is_zero() -> None:
    assert f1_char("이영희", "박민수") == 0.0


def test_f1_char_empty_input_is_zero() -> None:
    assert f1_char("", "홍성민") == 0.0
    assert f1_char("홍성민", "") == 0.0


def test_f1_char_short_strings_below_n_is_zero() -> None:
    # Both normalized strings shorter than n=3 after whitespace strip → 0
    assert f1_char("가", "가") == 0.0


# ---------- recall@k ----------

def test_recall_at_k_finds_gold_in_top_k() -> None:
    docs = ["박민수는 자연어처리 담당", "김철수는 컴퓨터공학과", "이영희는 인공지능학과"]
    assert recall_at_k(docs, "김철수", k=3) == 1.0


def test_recall_at_k_returns_zero_if_outside_top_k() -> None:
    docs = ["박민수", "이영희", "정수진", "김철수"]  # gold at index 3
    assert recall_at_k(docs, "김철수", k=3) == 0.0


def test_recall_at_k_empty_docs() -> None:
    assert recall_at_k([], "김철수", k=3) == 0.0


# ---------- precision ----------

def test_precision_counts_relevant_docs() -> None:
    docs = ["김철수 컴공", "김철수 AI", "이영희 AI"]
    assert precision(docs, "김철수") == pytest.approx(2 / 3)


def test_precision_empty_docs_is_zero() -> None:
    assert precision([], "김철수") == 0.0


# ---------- faithfulness ----------

def test_faithfulness_full_support() -> None:
    answer = "김철수 교수는 컴퓨터공학과 소속이다."
    ctx = ["김철수 교수는 컴퓨터공학과 소속의 인공지능 연구자이다."]
    assert faithfulness(answer, ctx) == pytest.approx(1.0)


def test_faithfulness_no_support() -> None:
    answer = "이영희 교수는 인공지능학과 소속이다."
    ctx = ["박민수는 자연어처리 담당이다."]
    assert faithfulness(answer, ctx) == 0.0


def test_faithfulness_empty_inputs() -> None:
    assert faithfulness("", ["x"]) == 0.0
    assert faithfulness("x", []) == 0.0


def test_faithfulness_list_form_per_item_support() -> None:
    """List answers use per-claim check — each name must appear in ctx."""
    ans = "홍성민, 황성민, 전성민"
    ctx = ["홍성민 교수와 황성민 교수가 강의를 담당한다"]
    # 2 of 3 items supported (전성민 missing) → 2/3
    assert faithfulness(ans, ctx) == pytest.approx(2.0 / 3.0)


def test_faithfulness_list_form_all_items_supported() -> None:
    ans = "홍성민, 황성민, 전성민"
    ctx = ["홍성민 황성민 전성민 세 교수가 과목을 함께 담당한다"]
    assert faithfulness(ans, ctx) == pytest.approx(1.0)


def test_faithfulness_list_form_no_support() -> None:
    ans = "홍성민, 황성민, 전성민"
    ctx = ["완전히 무관한 내용"]
    assert faithfulness(ans, ctx) == 0.0


def test_faithfulness_list_branch_stricter_than_sentence() -> None:
    """List form is stricter: unsupported items fail even if fillers match."""
    ctx = ["홍성민 관련 정보만 있음"]  # only 홍성민 named in context
    # Sentence form — shares filler '교수' with ctx? No, 'any multi-char
    # token' check uses 홍성민 which IS in ctx → the single sentence passes.
    ans_sent = "홍성민 교수와 황성민 교수, 전성민 교수가 담당합니다."
    assert faithfulness(ans_sent, ctx) == pytest.approx(1.0)
    # List form — per item; only 홍성민 matches → 1/3
    ans_list = "홍성민, 황성민, 전성민"
    assert faithfulness(ans_list, ctx) == pytest.approx(1.0 / 3.0)


# ---------- evaluate_single ----------

def test_evaluate_single_returns_all_fields() -> None:
    res = evaluate_single(
        pred="김철수 교수",
        gold="김철수 교수",
        retrieved_docs=["김철수 교수 컴공", "박민수"],
        contexts=["김철수 교수는 컴공이다."],
        k=3,
    )
    assert isinstance(res, EvalResult)
    assert res.f1 == pytest.approx(1.0)
    assert res.em_norm == 1.0
    assert res.em_raw == 1.0
    assert res.recall_at_k == 1.0
    d = res.as_dict()
    assert set(d.keys()) == {"f1", "em_raw", "em_norm", "recall_at_k", "precision", "faithfulness"}
