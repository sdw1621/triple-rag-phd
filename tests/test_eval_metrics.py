"""Tests for src.eval.metrics."""

from __future__ import annotations

import pytest

from src.eval.metrics import (
    EvalResult,
    evaluate_single,
    exact_match,
    f1_score,
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
