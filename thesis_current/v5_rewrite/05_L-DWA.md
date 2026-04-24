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

한 가지 실측 관찰을 덧붙여 둘 필요가 있다. 이 벤치마크의 gold 답이 콤마 구분 리스트 형식이라 EM 이 구조적으로 0 에 가까워, 실효 보상은 사실상 다음과 같이 동작한다.
```
R ≈ 0.5 · F1 + 0.2 · Faithfulness − penalty
```
즉 F1 과 Faithfulness 가 학습 신호의 주 원천이 된다. 이 구조가 학습에 미치는 영향은 Ch.6 §2.3 에서 구체적으로 다룬다.

### 1.4 단일-단계 축약

에피소드가 (state, action, reward) 한 쌍으로 종료되므로 GAE 의 시간 할인은 실효적으로 1-step Bellman (A = R − V, returns = R) 이 된다. 구현에서는 GAE 기계장치를 그대로 유지해 두었는데, 이는 이후 multi-turn 대화로의 확장 가능성을 닫지 않기 위함이다.

---

> **[박스 5-1] MDP 튜플 (S, A, P, R, γ<sub>d</sub>) **
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

백본을 Actor · Critic 이 공유하게 설계하여 모델 규모를 5,636 파라미터로 유지하였다. 층별 내역은 다음과 같다.

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

> **[박스 5-2] PPO Clip 목적 함수 **
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

### 3.6 학습 의사코드 (Algorithm 5-1)

본 연구의 PPO 학습 루프 완전 의사코드. 실구현은 `src/ppo/trainer.py::PPOTrainer.train`.

```
Algorithm 5-1: L-DWA PPO Training (Offline-cache-based)
─────────────────────────────────────────────────────────────
INPUT:
    D       : 5,000 질의 집합
    C       : 오프라인 보상 캐시 {(q_id, α, β, γ) → R}
    π_θ     : Actor 네트워크 (18 → 64 → 64 → 3 Dirichlet concentrations)
    V_φ     : Critic 네트워크 (shared backbone + Linear 64→1)
    η, ε, λ_GAE, c_V, c_H  : lr=3e-4, clip=0.2, λ=0.95, c_V=0.5, c_H=0.01
    N_ep = 10,000, K_rollout = 32, K_epochs = 4, B_mb = 8

INITIALIZE:
    θ, φ  ← orthogonal_init(gain=1.0)
    buffer ← empty list

FOR episode = 1 .. N_ep:
    # Stage 1. Rollout collection (32 samples)
    buffer.clear()
    FOR k = 1 .. K_rollout:
        q      ← sample_query(D)
        s      ← extract_state(q)               # 18-dim
        c      ← Softplus(π_θ(s)) + 1e-6         # Dirichlet concentration
        (α,β,γ) ← sample_Dirichlet(c)            # ∈ Δ³
        (α',β',γ') ← snap_to_grid((α,β,γ), Δ=0.1)  # 10-grid
        R      ← cache_lookup(C, q.id, (α',β',γ'))   # O(1)
        logp   ← log Dirichlet_pdf(c, (α,β,γ))
        V      ← V_φ(s)
        buffer.append((s, (α,β,γ), R, logp, V))

    # Stage 2. Advantage computation (GAE, 1-step MDP)
    FOR each (s, a, R, logp, V) in buffer:
        A ← R − V               # 1-step advantage
        Â ← standardize(A)       # batch normalize
        return ← R

    # Stage 3. PPO update (K_epochs × minibatch)
    FOR k = 1 .. K_epochs:
        shuffle(buffer)
        FOR each minibatch of size B_mb:
            r_t    ← exp(log Dirichlet_pdf(π_θ(s), a) − logp)  # new/old ratio
            L^CLIP ← mean(min(r_t · Â, clip(r_t, 1−ε, 1+ε) · Â))
            L^V    ← 0.5 · mean((V_φ(s) − return)²)
            L^H    ← −mean(Dirichlet_entropy(π_θ(s)))
            L      ← −L^CLIP + c_V · L^V + c_H · L^H
            θ, φ   ← θ, φ − η · ∇L                              # Adam step

    IF episode % 100 == 0:
        log(mean_reward, entropy, approx_KL, cache_hits)

OUTPUT: θ*, φ*  (trained actor-critic)
─────────────────────────────────────────────────────────────
```

### 3.7 추론 의사코드 (Algorithm 5-2)

테스트 시 L-DWA 는 학습된 Actor 로부터 **결정론적 Dirichlet 평균** 을 반환한다 (샘플링 없음, 재현성 보장).

```
Algorithm 5-2: L-DWA Inference (deterministic)
─────────────────────────────────────────────────────────────
INPUT:
    q      : test query
    π_θ*   : trained Actor

s      ← extract_state(q)                   # 동일 18-dim
c      ← Softplus(π_θ*(s)) + 1e-6
(α,β,γ) ← c / sum(c)                         # Dirichlet mean (μ_i = c_i / Σc_j)

OUTPUT: (α, β, γ) ∈ Δ³
─────────────────────────────────────────────────────────────
```

**왜 mean 인가?** 추론 단계에서의 샘플링은 (i) 재현성 저해, (ii) 결정 분산 증가, (iii) Oracle 비교 시 공정성 저해를 초래한다. 본 논문은 **Dirichlet mean 의 deterministic 추론** 을 표준으로 채택 (Ch.6 §3.2 전수 평가 기준).

## 4절 오프라인 보상 캐시

### 4.1 비용 장벽

on-policy 학습에서는 3 seed × 10,000 episode × 32 rollout 으로 총 960,000 회의 보상 호출이 필요하다. gpt-4o-mini 의 호출당 평균 비용 $0.0003 을 기준으로 하면 약 $288, 직렬 기준 시간은 50시간 이상이다. 학위 과정의 개인 예산으로는 부담스러운 규모이며, 하이퍼파라미터 탐색이나 ablation 까지 고려하면 그 배수가 더 커진다.

### 4.2 캐시 설계

이 비용을 없애는 가장 직접적인 방법은 보상을 사전에 계산해 두고 학습 중에는 조회만 하는 것이다. 가중치 공간을 Δ=0.1 간격으로 이산화하면 α + β + γ = 1 을 만족하는 정수 삼각형의 격자점이 C(12, 2) = 66 개가 나온다. 이를 5,000 질의와 곱하면 330,000 엔트리가 되며, 이를 SQLite 에 저장한다.

스키마는 단순하다.

```sql
CREATE TABLE rewards (
    query_id TEXT, alpha_int INT, beta_int INT, gamma_int INT,
    f1 REAL, em REAL, faithfulness REAL, latency REAL,
    PRIMARY KEY (query_id, alpha_int, beta_int, gamma_int)
);
```

학습 중에는 Dirichlet 으로 샘플된 연속 가중치를 가장 가까운 격자점으로 스냅한 뒤 O(1) 로 조회한다. 스냅에 의한 오차는 최대 Δ/2 = 0.05 이다.

### 4.3 구축 비용과 시간

합성 대학 캐시 (330K 엔트리) 는 10 워커 병렬로 14시간이 걸렸고, OpenAI 청구액 기준 $33 이 사용되었다. 최종 파일 크기는 16.8 MB 이다.

### 4.4 비용 절감 효과

| 방식 | 비용 | 3 seeds 총 시간 |
|---|---|---|
| On-policy (캐시 없음) | $288 | 50시간 이상 |
| Offline cache (본 연구) | $33 (일회) + $0 (학습) | 14h (캐시) + 1h (학습) = 15h |

이 캐시는 한 번 구축되면 seed 추가 학습, ablation, 정책 비교에 모두 재사용되기 때문에 seed 수가 늘어날수록 절감 효과가 커진다. N 개의 seed 를 학습해도 캐시 구축 비용은 여전히 $33 이고, on-policy 의 경우 $96·N 까지 선형으로 증가한다.

### 4.5 구현

캐시 구현은 `src/utils/offline_cache.py::OfflineCache` 에 있다. ThreadPoolExecutor 로 10개의 워커를 병렬 실행하며, SQLite WAL 모드와 `put_many` 배치 삽입으로 동시성 안전성을 확보하였다. 합성 대학용 구축 스크립트는 `scripts/build_cache.py` 이고, HotpotQA 등 교차 도메인용은 `scripts/cross_build_cache.py` 이다.

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
