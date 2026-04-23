# Corrected Baseline — 선행 논문의 3가지 재현 장애를 모두 해소한 새 기준

**작성 2026-04-22**
**역할**: 박사학위 논문(이하 PhD)의 모든 실험 수치가 참조하는 **단일 기준점 문서**. 선행 JKSCI 2025 논문의 [CORRIGENDUM](https://github.com/sdw1621/hybrid-rag-comparsion/blob/main/CORRIGENDUM.md) 에서 식별된 3가지 재현 장애를 차례로 해소하여 "신뢰 가능한 비교 베이스라인" 을 수립하고, L-DWA 의 순수 기여를 이 위에서 측정한다.

---

## 0. TL;DR

| 단계 | 무엇을 고쳤나 | R-DWA F1<sub>strict</sub> | L-DWA F1<sub>strict</sub> | ΔL-DWA |
|---|---|---|---|---|
| **S0** — JKSCI 원 주장 | (재현 불가 추정치) | **0.86** (주장) | 미학습 | — |
| **S1** — 평가기 버그 수정 후 | 구두점 제거 누락 → `_PUNCT_RE` 추가 | **0.072 → 0.137** (+90%) | 0.167 | **+21.9%** |
| **S2** — 프롬프트-gold 형식 정렬 후 | `PROMPT_TEMPLATE_LIST` 도입 | **0.137 → 0.529** (+286%) | 0.566 | **+7.0%** |
| **S3** — Faithfulness 지표 엄격화 (병행) | `faithfulness` 2-branch (list/sentence) | Faith: 0.835 → 0.544 | Faith: 0.865 → 0.585 | Faith: **+7.5%** |

**새 기준점 (Corrected Baseline = S1+S2+S3 적용된 R-DWA)**:
```
R-DWA (corrected baseline) on 5,000 QA, list prompt, 2-branch faith
  F1_strict       : 0.529 ± 0.429
  F1_substring    : 0.482 ± 0.442
  F1_char         : 0.469 ± 0.438
  EM_norm         : 0.387
  Faithfulness    : 0.544 ± 0.486
  Latency (s)     : 0.711
```

**본 논문의 핵심 비교 축은 S1+S2+S3 적용 상태에서 L-DWA vs R-DWA** 이다. 모든 Ch.6 표는 이 corrected baseline 을 참조한다.

---

## 1. 세 가지 재현 장애 (CORRIGENDUM 근거)

### 1.1 장애 ① — 평가기의 구두점 처리 누락

**증상.** `normalize()` 가 구두점을 토큰화 전에 제거하지 않아, 콤마 구분 리스트 gold (`"홍성민, 황성민, 전성민"`) 의 토큰이 `{"홍성민,", "황성민,", "전성민"}` 으로 분리. Sentence-form prediction 토큰 `{"홍성민", ...}` 과 교집합이 **마지막 토큰 하나로 축소** 되어 F1 이 실제의 약 1/3 수준으로 산출.

**수정 (`dff7dc1` on 2026-04-21).** `_PUNCT_RE` 정규식을 `normalize()` 파이프라인의 조사 제거 이전 단계에 삽입.

```python
# src/eval/metrics.py
_PUNCT_RE = re.compile(r"""[,.:;!?"'“”‘’()\[\]\{\}·・…―—ㆍ]""")

def normalize_korean(text: str) -> str:
    ...
    text = _PUNCT_RE.sub(" ", text)    # NEW
    text = _WHITESPACE_RE.sub(" ", text).strip()
    for _, pattern in _PARTICLE_PATTERNS:
        text = pattern.sub("", text)
    ...
```

**단독 영향.** R-DWA F1<sub>strict</sub> 0.072 → **0.137** (+90%). Recall 회복이 대부분 (리스트 항목이 이제 모두 매칭됨).

---

### 1.2 장애 ② — Gold 형식과 LLM 출력 형식 불일치

**증상.** `gold_qa_5000.json` 의 답은 `"홍성민, 황성민, 전성민"` 형태의 콤마 구분 리스트이지만, 기존 `PROMPT_TEMPLATE` ("정확하게 답변하세요...") 는 LLM 을 자연어 문장 형식 (`"... 홍성민, 황성민, 전성민입니다."`) 으로 유도. 구두점 버그 수정 후에도 **"교수", "담당합니다", "입니다" 등 filler 토큰이 precision 을 희석** 하여 F1<sub>strict</sub> 가 gold 의 상한을 크게 밑돔.

**수정 (PR #5).** `PROMPT_TEMPLATE_LIST` 를 `src/rag/triple_hybrid_rag.py` 에 추가하고, `evaluate_rerun.py` 에 `--prompt-style {sentence,list}` 플래그 도입. 기본 `sentence` 는 S1 호환성 유지, `list` 는 LLM 에 "이름/항목만 쉼표로 나열" 강제.

```python
PROMPT_TEMPLATE_LIST = (
    "다음 컨텍스트를 기반으로 질문에 답하세요.\n"
    "답변 형식 규칙(엄수):\n"
    "  - 정답이 여러 명/여러 항목이면 **이름(또는 항목)만을 쉼표(,)로 "
    "구분하여 나열**하세요. 문장 형태의 설명·조사·서술을 절대 덧붙이지 마세요.\n"
    "  - 정답이 하나면 **그 이름/항목 하나만** 출력하세요.\n"
    "  ... (후략)"
)
```

**단독 영향 (S1 위에서).** R-DWA F1<sub>strict</sub> 0.137 → **0.529** (+286%). **EM 0 → 0.387** (구조적 0 해소). 한편 Faithfulness 가 0.835 → 0.610 으로 하락, 원인은 장애 ③.

---

### 1.3 장애 ③ — Faithfulness 지표의 sentence-only 가정

**증상.** `faithfulness()` 가 마침표로 문장을 나눈 뒤 "any-token in context" 로 지지 여부 판정. 리스트형 응답 ("홍성민, 황성민, 전성민" — 쉼표만, 마침표 없음) 은 전체가 1개 "문장" 으로 처리되어 **all-or-nothing 평가**. Filler 없는 리스트 출력에서는 이름 하나라도 context 에 있으면 Faith = 1.0, 없으면 0.0 으로 과단순화.

**수정 (PR #5).** 2-branch faithfulness 도입:
- **문장형 분기 (마침표 포함)**: 원 구현 유지 (하위 호환).
- **리스트형 분기 (쉼표 + 마침표 없음)**: 각 콤마 항목을 **개별 claim** 으로 검증 → 전체 부분문자열 또는 모든 다자 토큰이 context 에 있으면 통과.

```python
def faithfulness(answer: str, contexts: list[str]) -> float:
    ...
    has_sentence_marker = bool(_SENTENCE_SPLIT_RE.search(answer))
    if has_sentence_marker:
        # 기존 문장형 분기
        ...
    # NEW: 리스트형 분기 — 항목별 per-claim 검증
    items = [x.strip() for x in answer.split(",") if x.strip()]
    supported = sum(1 for item in items if _item_is_supported(item, ctx_combined))
    return supported / len(items)
```

**단독 영향 (S2 위에서).** Sentence 프롬프트 Faith 는 불변 (0.835). List 프롬프트 Faith 는 0.610 → **0.544** (추가 감소, 더 정직한 측정). 환각된 이름을 **더 엄격히 잡아내기** 때문. 값이 낮아진 것이 아니라 **측정이 엄격해진 것**.

---

## 2. 3단계 적용 후 수치 (5,000 QA 전수, 3 seeds for L-DWA)

### 2.1 R-DWA — Corrected Baseline

| Stage | F1<sub>strict</sub> | F1<sub>sub</sub> | EM | Faith | 설명 |
|---|---|---|---|---|---|
| S0 | ~0.86* | — | ~0.78* | ~0.89* | JKSCI 주장 (재현 불가) |
| S1 | 0.137 ± 0.191 | 0.450 ± 0.447 | 0.000 | 0.835 ± 0.362 | +punct fix |
| S1+S2 | 0.529 ± 0.429 | 0.482 ± 0.442 | 0.387 | 0.610 ± 0.488 | +list prompt (단순 faith) |
| **S1+S2+S3** | **0.529 ± 0.429** | **0.482 ± 0.442** | **0.387** | **0.544 ± 0.486** | **새 기준점** |

\* S0 수치는 선행 논문 게재본. 본 저장소에서 재현 불가. CORRIGENDUM §3 참고.

### 2.2 L-DWA (PPO 학습) — New Contribution

| Stage | F1<sub>strict</sub> | F1<sub>sub</sub> | EM | Faith | 비고 |
|---|---|---|---|---|---|
| S1 (seed 42) | 0.167 | 0.488 | 0.000 | 0.865 | sentence 프롬프트, 3-seed std 0.001 |
| S1+S2+S3 (seed 42) | 0.566 | 0.512 | 0.389 | 0.585 | list 프롬프트, 2-branch faith |
| S1+S2+S3 (seed 123) | 0.554 | 0.497 | 0.387 | 0.585 | |
| S1+S2+S3 (seed 999) | 0.566 | 0.513 | 0.387 | 0.569 | |
| **S1+S2+S3 (3-seed mean)** | **0.562 ± 0.007** | **0.507 ± 0.009** | **0.388 ± 0.001** | **0.580 ± 0.009** | |

### 2.3 Oracle — 상한선 (per-query argmax over 66-grid)

| Stage | F1<sub>strict</sub> | F1<sub>sub</sub> | EM | Faith |
|---|---|---|---|---|
| S1 | 0.168 | 0.483 | 0.000 | 0.888 |
| **S1+S2+S3** | **0.554 ± 0.421** | **0.504 ± 0.436** | **0.388** | **0.570 ± 0.488** |

---

## 3. L-DWA 의 순수 PPO 기여 (Corrected Baseline 기준)

새 기준점 R-DWA (S1+S2+S3 적용) vs L-DWA (동일 S1+S2+S3 위):

| 지표 | R-DWA (baseline) | L-DWA (3-seed) | Δ | Oracle 대비 |
|---|---|---|---|---|
| F1<sub>strict</sub> | 0.529 | **0.562 ± 0.007** | **+6.2%** | Oracle 0.554 **초과** |
| F1<sub>substring</sub> | 0.482 | 0.507 ± 0.009 | **+5.2%** | Oracle 0.504 **초과** |
| F1<sub>char</sub> | 0.469 | 0.494 ± 0.010 | **+5.3%** | Oracle 0.487 **초과** |
| EM<sub>norm</sub> | 0.387 | 0.388 ± 0.001 | +0.3% | ≈ Oracle |
| Faithfulness | 0.544 | 0.580 ± 0.009 | **+6.6%** | Oracle 0.570 **초과** |

**핵심 관찰 — PPO L-DWA 의 순수 기여 (corrected baseline 위):**

1. **모든 F1 지표에서 L-DWA가 Oracle 초과**. Dirichlet 평균이 66-점 격자 외부를 탐색하여 이산 argmax 의 상한을 넘어섰다.
2. **3-seed std < 1%** (F1<sub>strict</sub>: 0.562 ± 0.007). 학습 알고리즘의 극단적 재현성.
3. **+6.2% 가 "순수 PPO 기여"** — 평가기 버그 수정 (+90%) 이나 프롬프트 정렬 (+286%) 과 구분되는 **학습된 가중치의 단독 이득**.
4. **Per-type 에서의 질적 우위 유지**: Conditional F1<sub>strict</sub> R-DWA 0.223 vs L-DWA 0.304 (**+36.7%**), Oracle 0.290 초과.

---

## 4. 새 기준점이 박사논문 서사에 미치는 영향

### 4.1 기여의 재배열

**기존 서사 (수정 전)**: "R-DWA 의 한계를 L-DWA 로 극복 → F1 +3.5% 달성 (절대 목표 0.89)".

**새 서사 (수정 후)**:
1. **S1 평가기 수정** = 재현성 기여 (engineering) + 선행 논문 자발적 정정 (integrity)
2. **S2 프롬프트 정렬** = 인프라 기여 (engineering)
3. **S3 Faith 엄격화** = 측정 방법론 기여 (methodology)
4. **Corrected baseline 위에서 L-DWA** = 순수 학습 기여 (methodology, 본 논문의 core)
5. **L-DWA 의 Oracle 초과 + 3-seed 재현성** = 학술 주장의 실증 핵심

### 4.2 비교 기준의 명시화

- **JKSCI 0.86 과 직접 비교는 불가** — CORRIGENDUM 으로 본 저자가 공개 정정.
- 대신 본 논문의 비교는 항상 **"Corrected R-DWA vs L-DWA (동일 S1+S2+S3)"** 형태.
- 원 JKSCI 수치는 Ch.6 에 "참고 (재현 불가, CORRIGENDUM 참조)" 각주로만 인용.

### 4.3 심사 방어 Q&A 예상

| 심사위원 질문 | 답변 |
|---|---|
| "선행 논문 F1 0.86 은 왜 재현 안 되나요?" | "본 저자가 2026-04-22 자발적 정정(CORRIGENDUM) 을 공개했습니다. 3가지 재현 장애를 식별·수정하고, 새 기준점 위에서 L-DWA 의 기여를 측정했습니다." |
| "그럼 L-DWA 개선이 얼마인가요?" | "Corrected baseline 위에서 R-DWA 대비 F1<sub>strict</sub> +6.2%. Oracle 상한을 모든 F1 지표에서 초과. 3-seed std < 1%. Conditional 유형에서 +36.7%." |
| "평가기 수정, 프롬프트 정렬은 PPO와 무관하지 않나요?" | "맞습니다 — Ch.6 에서 stage-wise 로 각 기여를 분리 보고합니다. 총 개선의 92% 는 infrastructure fix, 나머지 8% 가 순수 PPO 기여입니다. 본 논문은 두 축을 모두 기록하여 재현성 · 정확성 · 학습 효과를 투명히 분리했습니다." |

---

## 5. 참조 파일

- **CORRIGENDUM**: https://github.com/sdw1621/hybrid-rag-comparsion/blob/main/CORRIGENDUM.md
- **Punctuation fix (S1)**: commit `dff7dc1` on `main` (`src/eval/metrics.py`)
- **List prompt (S2)**: `src/rag/triple_hybrid_rag.py::PROMPT_TEMPLATE_LIST`
- **2-branch faith (S3)**: `src/eval/metrics.py::faithfulness` (PR #5)
- **Raw numbers**: `results/rerun_rdwa_list.json`, `results/rerun_ldwa_seed{42,123,999}_list.json`, `results/rerun_oracle_list.json`
- **Aggregate**: `results/m8_list_prompt_summary.md`

---

**이 문서가 박사논문의 모든 수치의 단일 출처 (Single Source of Truth) 이다.**
