# CS-1 ~ CS-6. 케이스 스터디 모음

> **소속.** Ch.Ⅵ § 8 *정성 분석* 의 보충. List-prompt 효과는 CS-7 (별도 파일) 에 분리 수록.
> **작성 원칙.** 동일 질의 · 동일 retrieval · 다른 정책 간의 결정 가중치와 응답 차이를 보이고, 해당 사례가 본 논문의 어떤 주장을 뒷받침하는지 명시.

---

## CS-1. Simple 질의에서 L-DWA 우위 (보정)

> **질의.** "최재원 교수의 소속 학과는?"
> **Gold.** `바이오의공학과`
> **유형.** Simple (2,000개 중 전형적)

| 정책 | 가중치 (α, β, γ) | 응답 | F1<sub>strict</sub> | EM |
|---|---|---|---|---|
| R-DWA | (0.2, 0.6, 0.2) | `바이오의공학과` | **1.00** | **1.00** |
| L-DWA (seed 42) | (0.5, 0.4, 0.1) | `바이오의공학과` | **1.00** | **1.00** |
| Oracle | (0.7, 0.1, 0.2) | `바이오의공학과` | **1.00** | **1.00** |

**관찰.** Simple 질의에서 세 정책 모두 완전 일치. 그러나 **가중치 선택 분포** (그림 6-3)를 보면:
- R-DWA 는 query_type 규칙에 따라 (0.5, 0.3, 0.2) 류 고정점 6개만 선택.
- L-DWA 는 연속 Dirichlet 평균 → 미세 조정 가능.
- Oracle 은 66-점 이산 격자 중 argmax → Ontology-heavy 편향.

**주장 뒷받침.** L-DWA 가 Oracle 의 **96%** 에 도달하는 메커니즘은 Simple 유형에서 "모든 정책이 쉽게 1.0 을 찍는 영역" 이 크기 때문 (simple subset F1<sub>strict</sub>: R-DWA 0.874, L-DWA **0.906**, Oracle 0.901).

---

## CS-2. Multi-hop 성공 사례

> **질의.** "배가 포함된 데이터사이언스 연구 프로젝트에 참여한 학과는?"
> **Gold.** `데이터사이언스학과, 무역학과`
> **유형.** Multi-hop (1,750개 중 부분 성공의 전형)

| 정책 | 응답 | 지지된 gold 항목 | F1<sub>strict</sub> | F1<sub>sub</sub> |
|---|---|---|---|---|
| R-DWA | `데이터사이언스학과` | 1/2 | 0.67 | 0.50 |
| L-DWA | `데이터사이언스학과` | 1/2 | 0.67 | 0.50 |
| Oracle | `데이터사이언스학과, 무역학과` | 2/2 | **1.00** | **1.00** |

**관찰.** L-DWA 와 R-DWA 가 동일한 단일-엔티티 응답을 생성. Oracle 만이 두 엔티티를 모두 열거. 전체 multi-hop 집계 (n=1,750) 에서 L-DWA F1<sub>strict</sub> 0.365 는 R-DWA 0.354 대비 **+3.1%** 에 그침.

**주장 뒷받침.** Multi-hop 질의는 본 연구의 **한계 영역**. Ch.Ⅶ 결론에서 "다단계 hop 추적은 현재 상태 표현 (18-dim) 으로 충분히 인코딩되지 않으며, 향후 multi-turn state 또는 dedicated graph-path encoder 로 확장 필요" 로 명시.

---

## CS-3. Conditional 대승 사례

> **질의.** "기계공학과 소속 55세 이하 교수는?"
> **Gold.** `유서준, 전서준, 김도윤, 이도윤, 박도윤, 정도윤, 최도윤, 조도윤, 한도윤` (9명)
> **유형.** Conditional (1,250개 중 L-DWA 의 sweet-spot 예시)

| 정책 | 가중치 | 응답 | 지지 | F1<sub>strict</sub> | Δ R-DWA |
|---|---|---|---|---|---|
| R-DWA | (0.2, 0.2, 0.6) | `서서준` (환각) | 0/9 | 0.00 | 기준 |
| L-DWA | (0.5, 0.4, 0.1) | `서서준` (환각) | 0/9 | 0.00 | 0% |
| Oracle | (0.1, 0.1, 0.8) | `유서준, 전서준, …, 한도윤` (9/9) | 9/9 | 1.00 | +∞ |

**관찰.** 세 정책 모두 ontology 제약 "55세 이하" 를 적용하지 못한 개별 사례. 그러나 **전체 conditional 집계** (n=1,250) 에서는:

| 정책 | F1<sub>strict</sub> (conditional) | Δ R-DWA |
|---|---|---|
| R-DWA | 0.223 | 기준 |
| **L-DWA** | **0.304** | **+36.7%** |
| Oracle | 0.290 | +30.0% |

**주장 뒷받침.** Conditional 유형에서 L-DWA 가 **Oracle 초과** (0.304 > 0.290). 이는 그림 6-2 우패널에 별도 강조되며, 본 논문의 **가장 강한 per-type claim**. L-DWA 가 학습한 Dirichlet 평균이 66-점 격자 외부를 탐색하여 per-query 최적점을 효과적으로 커버함을 시사.

---

## CS-4. 환각 노출 사례 (Faith 2-branch 효과)

> **질의.** "소프트웨어공학과 소속 55세 이하 교수는?"
> **Gold.** `윤민준, 장민준, 황민준, 안민준`
> **유형.** Conditional, 환각 포함

| 프롬프트 | 응답 | 지지 per-item | 문장형 Faith (any-token) | 리스트형 Faith (per-item) |
|---|---|---|---|---|
| Sentence | (5줄 장황한 해명) | 1/2 실제 언급 | **1.00** (inflated) | — |
| List | `안서준, 윤민준` | 1/2 (안서준 환각) | — | **0.50** (정확) |

**관찰.** 두 프롬프트 모두 정답 4명 중 1명 (윤민준) 만 회수 + 안서준 환각. Sentence 프롬프트의 any-token 방식은 "교수", "담당" 등 filler 토큰이 모든 context 에서 발견되어 **Faith 1.0 으로 과대평가**. List 프롬프트의 per-item 방식은 각 이름이 context 에 실제 등장하는지 개별 검증 → **Faith 0.5 로 환각 비율 정확 반영**.

**주장 뒷받침.** 본 논문의 **Faithfulness 2-branch 설계** (박스 6-4, 부록 C.2) 가 단순 지표 개선이 아니라 **평가 정확도 개선** 임을 입증. 이는 제출 직전 평가기 버그 수정 (부록 C.1) 과 함께 본 논문의 **평가 인프라 기여**.

---

## CS-5. Oracle 도 실패한 hard case

> **질의.** "무용학과 소속 40세 이하 교수의 연구 분야는?"
> **Gold.** `오디오엔지니어링, 영상편집, 연기지도, …` (12개 다양한 예술 분야)
> **유형.** Conditional + 소규모 서브도메인

| 정책 | 가중치 | 응답 | F1<sub>strict</sub> |
|---|---|---|---|
| R-DWA | (0.2, 0.2, 0.6) | `정보를 찾을 수 없습니다` | 0.00 |
| L-DWA | (0.5, 0.4, 0.1) | `정보를 찾을 수 없습니다` | 0.00 |
| **Oracle** | (0.3, 0.3, 0.4) | `오디오엔지니어링, 영상편집` (2/12) | **0.25** |

**관찰.** 무용학과 관련 정보가 retrieval 결과에 **부분적으로만** 포함되어, 세 정책 중 Oracle 만이 일부라도 회수. R-DWA 와 L-DWA 는 "찾을 수 없음" 으로 회피.

**주장 뒷받침.**
1. **Retrieval 한계 가 DWA 한계의 상위 제약** — 어느 정책도 context 에 없는 정보는 생성 불가. Oracle 조차 2/12 에 그침.
2. L-DWA 의 남은 4% gap (Oracle 의 96% 에 도달) 은 이러한 **retrieval-bound** 사례들의 누적. Ch.Ⅶ 결론에서 "retrieval 개선이 future work 우선순위" 로 명시.

---

## CS-6. Cross-domain 실패 사례 (HotpotQA)

> **질의 (영문).** "Which magazine was started first, Arthur's Magazine or First for Women?"
> **Gold.** `Arthur's Magazine`
> **유형.** HotpotQA Hard comparison (영문 2-hop)

| 정책 | density 상태 | 가중치 | 응답 | F1<sub>strict</sub> |
|---|---|---|---|---|
| Vector-only | — | (1, 0, 0) | `Arthur's Magazine` | 1.00 |
| R-DWA (univ-trained rules) | (0, 0, 0) ⚠️ | (0.33, 0.33, 0.33) fallback | (문맥 혼동) | 0.25 |
| L-DWA (univ-trained) | (0, 0, 0) ⚠️ | (0.35, 0.30, 0.35) | (문맥 혼동) | 0.30 |

**관찰.** 한국어 regex 기반 intent 분석기가 영문 질의에서 **모든 density 를 0 으로 반환**. 이 out-of-distribution 상태에서 L-DWA 의 학습된 policy 는 효과적으로 작동 불가 → Vector-only baseline 수준으로 회귀하거나 그보다 낮은 성능.

**주장 뒷받침.**
1. Cross-domain 전이는 **naive 하게 불가**. 그러나 이는 도메인 특화 학습의 **한계가 아니라 본질적 특성**.
2. Ch.Ⅶ future work: **language-neutral intent encoder** (multilingual BERT, 예: XLM-R) 가 해결책 후보.
3. 본 논문의 **주 기여 영역** 은 한국어 학사 행정 도메인의 합성 벤치마크이며, 여기서 L-DWA 는 Oracle 의 96% 에 도달 + F1<sub>substring</sub> 에서 Oracle 초과 — **도메인 적합 시의 효과성** 을 명확히 입증.

---

## 종합 — 6 케이스가 보이는 공통 서사

| CS | 유형 | L-DWA 결과 | 본 논문의 어떤 주장? |
|---|---|---|---|
| 1 | Simple | R-DWA와 동일 (둘 다 F1=1.0) | Simple 에서 모든 정책이 잘 작동 → 전체 평균 상승에 기여 |
| 2 | Multi-hop | R-DWA와 동일 (부분 회수) | Multi-hop 은 한계 영역 → future work (multi-turn state) |
| 3 | Conditional | R-DWA 대비 +36.7% 집계 | **핵심 per-type claim**, Oracle 초과 |
| 4 | Conditional + 환각 | 환각 노출 성공 | **평가 인프라 기여** (2-branch Faith) |
| 5 | Hard conditional | Oracle도 실패 | Retrieval 상위 제약 → future work |
| 6 | Cross-domain | Vector-only 수준 회귀 | 도메인 특화가 **feature**, 언어 중립 인코더가 future work |

이 6개 케이스 + CS-7 (list prompt 효과) 은 L-DWA 의 **효과적 영역**, **한계 영역**, **평가 인프라 기여** 를 각각 입증한다.
