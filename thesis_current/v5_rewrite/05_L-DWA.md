# 제Ⅴ장 학습형 동적 가중치 알고리즘 (L-DWA)

본 장은 본 논문의 **핵심 기여** 인 Learned Dynamic Weighting Algorithm을 정의한다. §1에서 MDP 공식화, §2에서 Actor-Critic 정책 네트워크, §3에서 PPO 학습 알고리즘, §4에서 오프라인 보상 캐시 시스템을 다룬다.

## 1절 MDP 공식화

Triple-Hybrid RAG 가중치 결정을 다음 1-step contextual bandit MDP로 정식화한다.

### 1.1 상태 (State, 18-dim)

각 질의에 대해 18차원 상태 벡터를 구성한다 (식 5-1):

```
s = [density (3) | intent_logits (3) | source_stats (9) | query_meta (3)]
```

| 슬롯 | 차원 | 출처 | 정의 |
|---|---|---|---|
| density | 3 | RuleBasedIntent | (s_e, s_r, s_c) 정규화 밀도 신호 |
| intent_logits | 3 | BERT classifier (또는 0) | (logit_simple, logit_multi, logit_cond) |
| source_stats | 9 | Retrieval 출력 | 3 소스 × (top, mean, count) |
| query_meta | 3 | Query string | (log_length_norm, n_entities_norm, has_negation_flag) |

**intent_logits.** 본 논문 실험에서는 BERT multi-label classifier를 미학습 상태로 두어 intent_logits=(0, 0, 0) 으로 둔다. 이 선택의 영향은 Ch.6 §5.2의 소거 논의에서 다룬다. 요컨대 density만으로도 L-DWA가 Oracle의 99%에 도달하므로 BERT는 **향후 추가 개선 여지** 로 남긴다.

**source_stats.** 각 소스별 retrieval 결과에서:
- top = 상위 1순위 유사도 점수 (Vector의 경우 0~1, Graph/Ontology는 발견 여부 기반 0/1 합의치)
- mean = 상위 top_k 유사도 평균
- count_norm = min(len(results) / top_k, 1.0)

### 1.2 행동 (Action)

3-simplex 상의 연속 행동:
```
a = (α, β, γ) ∈ Δ³,  α + β + γ = 1, α,β,γ ≥ 0
```

Actor 네트워크는 Dirichlet 분포의 농도 파라미터 c = (c_α, c_β, c_γ) 를 출력하고, 학습 시에는 Dirichlet(c) 샘플, 추론 시에는 Dirichlet mean (= c/||c||_1) 을 사용한다.

### 1.3 보상 (Reward, Eq. 5-7)

```
R(s, a) = 0.5 · F1 + 0.3 · EM + 0.2 · Faithfulness − 0.1 · max(0, latency − 5.0)
```

**중요한 실증 관찰.** 본 벤치마크의 gold 답이 콤마 구분 리스트 형식이라 EM = 0이 구조적으로 고정된다 (Ch.6 §1.3). 따라서 실효 보상은
```
R ≈ 0.5 · F1 + 0.2 · Faithfulness − penalty
```
이며, F1과 Faithfulness가 학습 신호의 주 원천이다. 본 논문은 이 구조를 Ch.6 §2.3 에서 투명하게 기재한다.

### 1.4 단일-단계 축약

에피소드가 단일 (state, action, reward) 쌍으로 종료되므로 GAE의 시간 할인은 실효적으로 1-step Bellman: A = R − V, returns = R. 본 논문의 구현은 GAE machinery를 유지하여 장래 multi-turn dialog 확장을 대비한다.

---

> **[박스 5-1] MDP 튜플 (S, A, P, R, γ<sub>d</sub>) — 정의 · 예시 · 해석**
>
> **정의.** PPO 학습 환경을 구성하는 5원 튜플. 본 연구의 구체값:
>
> | 기호 | 의미 | 본 연구 구체값 |
> |---|---|---|
> | *S* | 상태 공간 | 18차원 벡터 (density·intent·source_stats·query_meta) |
> | *A* | 행동 공간 | Δ³ 위의 가중치 (α, β, γ) |
> | *P* | 전이 함수 | 1-step MDP (다음 상태 없음) ≈ 지도학습 등가 |
> | *R* | 보상 함수 | 0.5·F1 + 0.3·EM + 0.2·Faith − 0.1·max(0, lat−5) |
> | *γ<sub>d</sub>* | 할인 계수 | 0.99 (1-step이므로 실효 무의미) |
>
> **예시.** 한 학습 사이클:
> 1. 질의 *"기계공학과 55세 이하 교수는?"* 입력 → *s* (18-dim) 추출
> 2. 정책 π(·|s) 가 *a* = (0.1, 0.2, 0.7) 샘플링
> 3. 파이프라인이 *a* 로 컨텍스트 병합 → LLM 답변 → F1=0.8, Faith=0.9 측정
> 4. *R* = 0.5·0.8 + 0.2·0.9 = 0.58 (EM=0, latency<5)
> 5. (*s*, *a*, *R*) 튜플로 PPO 정책 업데이트
>
> **해석.** 본 연구의 MDP는 **단일 step** 이어서 value function의 역할이 약하지만, PPO의 clipping이 안정적 학습을 보장한다 (박스 5-2 참조). 다음 상태가 없으므로 1-step bandit 문제로 단순화 가능하나, PPO 구현을 유지하는 것이 재현성·확장성 면에서 유리.

---

## 2절 Actor-Critic 네트워크

### 2.1 구조 (Figure 5-1)

```
s (18) → Linear(18→64) → Tanh → Linear(64→64) → Tanh → features (64)
                                                       ↓
                         Actor head:  Linear(64→3) → Softplus + ε → Dirichlet concentrations c
                         Critic head: Linear(64→1)                  → 상태 가치 V(s)
```

![그림 5-1. Actor-Critic 네트워크 구조 (5,636 파라미터 경량 정책망)](docs/figures/fig5_1_actor_critic.png)

### 2.2 파라미터 수

본 연구는 파라미터 효율을 위해 공유 backbone 을 사용한다. 총 **5,636 파라미터**:

| 층 | in × out + bias | 파라미터 |
|---|---|---|
| Linear (18→64) | 18·64 + 64 | 1,216 |
| Linear (64→64) | 64·64 + 64 | 4,160 |
| Actor head (64→3) | 64·3 + 3 | 195 |
| Critic head (64→1) | 64·1 + 1 | 65 |
| **합계** | | **5,636** |

선행 연구 및 유관 RL 연구의 정책 네트워크는 수십만 ~ 수백만 파라미터 규모이나, 본 연구의 상태 공간 (18-dim) 과 행동 공간 (3-simplex) 의 간결함이 극단적 경량 설계를 가능케 한다.

### 2.3 초기화 전략

**Orthogonal init (gain=1.0).** 모든 Linear 층의 weight를 직교 초기화하고, bias 를 0 으로 둔다. 이 선택의 효과: **초기 정책의 Dirichlet 평균이 ≈ (1/3, 1/3, 1/3)** 이 되어, R-DWA의 "균등한 prior" 를 닮은 상태에서 학습을 시작하게 된다. 이는 (i) 단순 warm-up 없이 안정적 초기 탐색을 가능케 하고, (ii) 학습 결과의 해석을 돕는다 (R-DWA baseline 에서 얼마나 movement 했는가).

## 3절 PPO 학습 알고리즘

### 3.1 표준 PPO with Clip

정책 손실:
```
L^π(θ) = −E[ min(ratio · A, clip(ratio, 1−ε, 1+ε) · A) ]
ratio = π_θ(a|s) / π_θ_old(a|s)
```

가치 손실:
```
L^V(φ) = 0.5 · E[(R − V_φ(s))²]
```

엔트로피 보너스:
```
L^H(θ) = −H(π_θ)
```

합 손실 (Actor-Critic 공유 backbone):
```
L(θ, φ) = L^π + c_V · L^V − c_H · H
```

---

> **[박스 5-2] PPO Clip 목적 함수 — 정의 · 예시 · 해석**
>
> **정의.** 정책 업데이트 시 새 정책이 이전 정책과 지나치게 달라지지 않도록 **확률비 r<sub>t</sub>(θ)** 를 잘라(clip) 학습을 안정화.
>
> **형식 정의.**
> L<sup>CLIP</sup>(θ) = 𝔼<sub>t</sub>[ min(r<sub>t</sub>(θ)·Â<sub>t</sub>, clip(r<sub>t</sub>(θ), 1−ε, 1+ε)·Â<sub>t</sub>) ]
>
> 본 연구 ε = 0.2 (Table 5-4).
>
> **예시.** 한 (s, a) 상황에서 clipping 동작:
>
> | 상황 | r<sub>t</sub> | Â<sub>t</sub> | Clipped r<sub>t</sub> | 최종 term | 해석 |
> |---|---|---|---|---|---|
> | 정책이 살짝 좋은 방향 | 1.1 | +0.5 | 1.1 | +0.55 | 정상 업데이트 |
> | 정책이 크게 좋은 방향 | 1.5 | +0.5 | **1.2** | **+0.60** | clip 발동 |
> | 정책이 살짝 나쁜 방향 | 0.9 | −0.3 | 0.9 | −0.27 | 정상 업데이트 |
> | 정책이 크게 나쁜 방향 | 0.5 | −0.3 | **0.8** | **−0.24** | clip 발동 |
>
> **해석.** clip 없는 vanilla PG는 한 step의 큰 업데이트가 이미 배운 것을 **파괴** 할 수 있다. PPO clip은 "한 번에 ±20% 이상 변하지 말 것" 을 강제하여 장기 수렴을 보장. 본 연구 3-seed 학습에서 R<sub>late</sub> = 0.215 ± 0.002 의 극단적 일관성은 clip 덕분이다 (§3.5).

---

### 3.2 하이퍼파라미터 (Table 5-4)

| 파라미터 | 값 | 근거 |
|---|---|---|
| learning rate | 3e-4 | Adam 기본, RL 문헌 표준 |
| gamma (할인) | 0.99 | 1-step이라 실효 무관 |
| GAE lambda | 0.95 | 표준 |
| clip ratio ε | 0.2 | Schulman et al. (2017) |
| value coef c_V | 0.5 | 표준 |
| entropy coef c_H | 0.01 | 탐색 유지 |
| max grad norm | 0.5 | 안정성 |
| rollout 크기 | 32 | 각 episode당 |
| minibatch 크기 | 8 | rollout 내 |
| update epochs | 4 | rollout당 |
| 총 episodes | **10,000** | seed 당 (Ch.6 §5) |

### 3.3 Advantage 표준화
각 rollout 내에서 advantage를 평균 0, std 1로 표준화하여 서로 다른 reward scale 간의 학습 안정성을 확보한다.

### 3.4 재현성

PPO 학습은 3 seed × 10,000 episode:
- seed 42: Total R 0.1988 → 0.2145 (Δ +0.016)
- seed 123: Total R 0.1970 → 0.2153 (Δ +0.018)
- seed 999: Total R 0.1964 → 0.2146 (Δ +0.018)

**3 seeds 간 최종 R 표준편차 0.0016 (0.7%)** — 학습 알고리즘이 재현 가능함을 실증한다.

### 3.5 학습 곡선 관찰

모든 seed에서 episode ~2,000 까지 급격히 R 상승 후 R ≈ 0.21에서 미세 진동하며 plateau를 유지한다. Policy entropy H 는 ep 1 의 −0.8 에서 ep 10,000 의 −2.7까지 점진 감소하여 초기 탐색 → 점진적 결정성 확보 패턴을 보인다.

![그림 5-2. PPO 학습 곡선 — 3 seeds (42, 123, 999) 의 평균 보상 및 정책 엔트로피 수렴](docs/figures/fig5_2_ppo_convergence.png)

## 4절 오프라인 보상 캐시

### 4.1 비용 장벽

3 seed × 10,000 episode × 32 rollout = **960,000 보상 호출** 이 on-policy 학습에 요구된다. gpt-4o-mini 1콜당 평균 $0.0003 기준 **$288**, 실 지연 약 50시간 이상 (단순 직렬 기준) 이 예상된다.

### 4.2 캐시 설계

본 연구는 **5,000 질의 × 66 이산 가중치 = 330,000 엔트리** 의 보상을 SQLite에 사전 계산한다.

**이산화.** 가중치 간격 Δ=0.1 격자에서 α+β+γ=1을 만족하는 정수 삼각형 점:
```
(a_int, b_int, g_int) ∈ {0,1,...,10}³, a_int + b_int + g_int = 10
총 |S_Δ| = C(12, 2) = 66
```

**Schema.**
```sql
CREATE TABLE rewards (
    query_id TEXT, alpha_int INT, beta_int INT, gamma_int INT,
    f1 REAL, em REAL, faithfulness REAL, latency REAL,
    PRIMARY KEY (query_id, alpha_int, beta_int, gamma_int)
);
```

**스냅.** 학습 중 Dirichlet 샘플된 연속 가중치를 정수 격자로 스냅한 후 O(1) 캐시 룩업. 스냅 오차는 Δ/2 = 0.05 이내.

### 4.3 구축 비용/시간

합성 대학 캐시 (330K 엔트리, 10 워커 병렬):
- 실제 wall-clock: **14시간** (단일 장애 복구 포함)
- 실제 비용: **$33** (OpenAI invoice 기준)
- 파일 크기: **16.8 MB**

### 4.4 비용 절감 효과 (핵심 기여)

| 방식 | 비용 | 시간 (3 seeds) |
|---|---|---|
| On-policy (캐시 없음) | $288 | ≥ 50시간 |
| **본 연구 offline cache** | **$33 (일회) + $0 (학습)** | **14h (캐시) + 1h (학습) = 15h** |
| 절감율 | **−85%** | **−70% (또는 seed 수가 증가할수록 극적 감소)** |

**Amortization.** 캐시는 일회 구축 후 모든 seed, 모든 ablation, 모든 policy 비교에 재사용된다. N seed 학습 시 비용은 여전히 $33 (학습은 $0), on-policy는 $96·N.

### 4.5 구현

`src/utils/offline_cache.py::OfflineCache`. ThreadPoolExecutor n_workers=10 기반 병렬 빌드, SQLite WAL 모드, `put_many` 배치 삽입으로 동시성 안전. `scripts/build_cache.py`는 합성 대학 벤치마크용, `scripts/cross_build_cache.py`는 교차 도메인 (HotpotQA 등) 용.

## 5절 학습 시스템 통합

### 5.1 StateProvider 와 RewardFn 주입

`PPOTrainer` 는 두 callable 에 의존한다:
- `state_provider(query_index: int) → State`
- `reward_fn(query_index: int, weights: DWAWeights) → float`

양자는 주입 가능하여, 학습 전 pre-compute state pickle을 작성하거나 cache 룩업을 래핑 등 유연한 구성이 가능하다.

### 5.2 BERT intent classifier의 현 상태

`src/intent/bert_classifier.py::BertIntentClassifier` 는 klue/bert-base 기반 multi-label 분류기로 구현되어 있다 (3 labels = simple/multi_hop/conditional, BCE loss, tokenizer + model DI). 본 논문 실험에서는 **학습 시간 제약으로 미학습 상태** 로 두고 intent_logits=(0, 0, 0) 을 사용하였다.

이 선택에도 L-DWA가 Oracle의 99.4%에 도달함 (Ch.6 §3) 은 density + source_stats + query_meta 의 15차원만으로도 학습 신호가 충분함을 시사한다. **BERT 학습 시 추가 개선 여지** 는 Ch.7 §2의 향후 연구 항목이다.

### 5.3 전체 학습 파이프라인

```
1. 코퍼스 로드 (dataset_generator.py seed=42)
2. VectorStore 구축 (OpenAI 임베딩 × 2542 문서, ~10초, $0.005)
3. 5000 질의 × 66 가중치 보상 캐시 구축 (14시간, $33)
4. For each seed ∈ {42, 123, 999}:
    4a. 5000 질의 state pre-compute (~20분, 첫 seed만)
    4b. PPO 학습 10,000 episodes (GPU 16분, $0)
    4c. final.pt 체크포인트 저장
5. L-DWA 평가 (fresh LLM, ~10분/정책, ~$3 per 정책)
```

총 소요: **약 15시간, $38** (3-seed 모든 ablation 포함).

## 6절 소결

본 장은 MDP 공식화, 5,636 파라미터의 경량 Actor-Critic, PPO 학습 레시피, 그리고 학습 비용 85% 절감 오프라인 캐시를 통합하여 **L-DWA 프레임워크** 를 정의하였다. 이 프레임워크 위에서 Ch.6의 실험은 R-DWA 대비 F1_strict +21.9% 와 Oracle 99.4% 도달을 실증한다.
