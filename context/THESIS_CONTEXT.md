# 🎓 Thesis Context — What Each Chapter Does

> Summary of what's in the PhD thesis body (already written, v4/v3).
> Code implementation must align with these thesis descriptions.

---

## Ⅰ. 서론 (Introduction) — 9p
### Research Questions
1. **RQ1**: 규칙 기반 가중치(R-DWA)는 왜 경계선상 질의에서 실패하는가?
2. **RQ2**: MDP 형식화 + PPO로 가중치 결정을 학습할 수 있는가?
3. **RQ3**: 학습된 가중치 정책은 도메인 간 일반화 가능한가?

### Key Contribution Claim
1. MDP formalization for Triple-Hybrid RAG weight decision (first)
2. PPO-based L-DWA with lightweight policy network
3. Quantitative comparison R-DWA vs L-DWA (+3.5% F1, +32.8% boundary EM)
4. BERT multi-label Intent Classifier for boundary queries
5. Domain-agnostic validation (4 benchmarks)

---

## Ⅱ. 관련 연구 (Related Work) — 16p
### Covered Topics
- Traditional RAG (vector-only)
- GraphRAG, HybridRAG
- Self-RAG, CRAG (comparison baselines)
- Adaptive-RAG (Jeong et al.)
- Neural RAG evaluation (RAGAS)
- MDP for retrieval, PPO basics
- Knowledge graphs + ontologies in RAG

### Positioning
This thesis is the first to formulate Triple-Hybrid RAG weight decision as MDP + PPO.
Closest prior work: Adaptive-RAG (no weight learning), GraphRAG (no ontology), Self-RAG (no graph).

---

## Ⅲ. Triple-Hybrid RAG 아키텍처 — 13p
### System Overview
```
User Query
   ↓
Query Analyzer → (entities, relations, constraints) density signals
   ↓
Intent Classifier (Rule-based OR BERT) → query type
   ↓
DWA (R-DWA OR L-DWA) → (α, β, γ) weights on simplex
   ↓
   ├─→ Vector Retrieval (FAISS, text-embedding-3-small, top-k=3)
   ├─→ Graph Retrieval (NetworkX BFS 3-hop)
   └─→ Ontology Retrieval (Owlready2 + HermiT reasoner)
   ↓
Score fusion: S_total = α·S_vec + β·S_graph + γ·S_onto
   ↓
LLM (GPT-4o-mini, temperature=0.0)
   ↓
Final Answer
```

### Three Sources Design
- **Vector (S_vec)**: Unstructured text, semantic similarity
- **Graph (S_graph)**: Entity-relationship BFS traversal
- **Ontology (S_onto)**: Class hierarchy + constraint reasoning

### Weight Simplex Constraint
α + β + γ = 1, α,β,γ ∈ [0, 1]

---

## Ⅳ. 규칙 기반 동적 가중치 알고리즘 (R-DWA) — 13p
### Algorithm (Prior Work, Baseline)
Stage 1: Query Type Classification → base weights
```
simple:      (0.6, 0.2, 0.2)
multi_hop:   (0.2, 0.6, 0.2)
conditional: (0.2, 0.2, 0.6)
```

Stage 2: Density-based Adjustment
```
α' = α_base × (1 - λ·mean(s_e, s_r, s_c))
β' = β_base + λ·s_r·(1 - β_base)
γ' = γ_base + λ·s_c·(1 - γ_base)
```

Stage 3: Normalization
```
(α, β, γ) = (α', β', γ') / (α' + β' + γ')
```

λ = 0.3 (empirically tuned on synthetic data)

### Limitations (Motivating L-DWA)
1. Fixed base weights regardless of actual content
2. Linear adjustment doesn't capture non-linear interactions
3. Hard boundary queries (multi_hop ∩ conditional) → suboptimal
4. Domain-specific tuning required

---

## Ⅴ. PPO 기반 L-DWA (CORE) — 19p
### MDP Formulation

**State s ∈ ℝ^18**:
```
s = [density signals (3),
     BERT intent logits (3: simple/multi_hop/conditional),
     source statistics (9: score mean/std/max for each source),
     query metadata (3: length, entity_count, has_negation)]
```

**Action a ∈ simplex^3**:
```
a = (α, β, γ), α + β + γ = 1, α,β,γ ∈ [0, 1]
```

**Reward R(s, a)**:
```
R = 0.5 · F1(answer, gold)
  + 0.3 · EM(answer, gold)
  + 0.2 · Faithfulness(answer, retrieved)
  - 0.1 · max(0, latency - 5.0)  # latency in seconds
```

### Policy Network (Actor-Critic)
```
Shared backbone:
  FC(18 → 64) + Tanh
  FC(64 → 64) + Tanh

Actor head:
  FC(64 → 3) + Softplus  → Dirichlet concentration parameters

Critic head:
  FC(64 → 1)  → Value estimate

Total parameters: ~6,081
```

### PPO Training Loop
```python
for episode in range(10000):
    # 1. Rollout (batch of 32 queries)
    for _ in range(32):
        s = extract_state(query)
        α_params = actor(s)
        a = Dirichlet(α_params).sample()
        
        # Use offline cache (KEY OPTIMIZATION)
        discretized_a = round_to_0_1_grid(a)
        cached = lookup_cache(query_id, discretized_a)
        
        r = compute_reward(cached)
        v = critic(s)
        log_prob = Dirichlet(α_params).log_prob(a)
        
    # 2. GAE advantage
    advantages = compute_gae(rewards, values, gamma=0.99, lambda=0.95)
    
    # 3. PPO update (4 epochs, minibatch 8)
    for _ in range(4):
        for minibatch in shuffled_batches(8):
            new_log_prob = Dirichlet(actor(s)).log_prob(a)
            ratio = exp(new_log_prob - old_log_prob)
            surr1 = ratio * advantages
            surr2 = clip(ratio, 1-0.2, 1+0.2) * advantages
            policy_loss = -min(surr1, surr2).mean()
            value_loss = (critic(s) - returns).pow(2).mean()
            entropy_bonus = Dirichlet(actor(s)).entropy().mean()
            
            loss = policy_loss + 0.5*value_loss - 0.01*entropy_bonus
            
            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(params, 0.5)
            optimizer.step()
```

### Convergence Behavior (Expected)
- Episodes 0-2000: Exploration (entropy ≈ 2.3)
- Episodes 2000-5000: Learning (F1 rapid rise)
- Episodes 5000-8000: Refinement (fine-tuning)
- Episodes 8000-10000: Convergence (F1 ≈ 0.89, entropy ≈ 1.0)

### Thesis Table 5-4 (PPO Hyperparameters)
| Parameter | Value | Note |
|---|---|---|
| learning_rate | 3e-4 | Adam optimizer |
| gae_lambda | 0.95 | GAE for advantage |
| clip_ratio | 0.2 | PPO policy clipping |
| value_coef | 0.5 | Critic loss weight |
| entropy_coef | 0.01 | Entropy bonus |
| max_grad_norm | 0.5 | Gradient clipping |
| total_episodes | 10,000 | Training duration |
| rollout | 32 | Queries per rollout |
| update_epochs | 4 | Per rollout |
| minibatch | 8 | Per update |

---

## Ⅵ. 실험 및 평가 — 31p (Has § Placeholders)
### Setup (Sec 1)
- Hardware: RTX 4090, Ubuntu 22.04, CUDA 12.1
- Software: PyTorch 2.1.2, numpy<2.0

### Tables Summary
| Table | Content | Status |
|---|---|---|
| 6-1 | System overview | ✅ Fixed |
| 6-2 | Overall performance (5 systems + ours) | ⏳ § placeholders |
| 6-3 | Query type EM comparison | ⏳ § placeholders |
| 6-4 | Ablation: w/o BERT, w/o Cache | ⏳ § placeholders |
| 6-5 | HotpotQA Hard 300 results | ⏳ § placeholders |
| 6-6 | MuSiQue Dev 300 results | ⏳ § placeholders |
| 6-7 | PubMedQA Pharma 300 results | ⏳ § placeholders |
| 6-8 | Cross-domain weight distribution | ⏳ § placeholders |
| 6-9 | λ sensitivity | ⏳ § placeholders |
| 6-10 | Seeds × 3 runs ± std | ⏳ § placeholders |

### Case Studies (7 total)
1. CS-1: Boundary query (multi_hop ∩ conditional), R-DWA fail → L-DWA succeed
2. CS-2: Complex constraint with 4 filters
3. CS-3: Rank filter query
4. CS-4: Pharmaceutical 4-constraint
5. CS-5: HotpotQA bridge-entity
6. CS-6: 4-hop limit test
7. CS-7: PubMedQA reasoning

### Ⅵ.10 Open Source Contribution
```
Our implementation is publicly released at:
https://github.com/sdw1621/triple-rag-phd

Includes:
- Complete Docker environment
- PPO training code
- Offline cache infrastructure
- Evaluation scripts for all 4 benchmarks
- 3-seed reproducibility
```

---

## Ⅶ. 결론 (Conclusion) — 9p
### Summary of Contributions
Restatement of Ⅰ장 contributions with evidence from Ⅵ장.

### Limitations (Sec 2)
1. Synthetic university data may not fully represent real administrative diversity
2. SNOMED CT subset is limited (5,000 concepts vs full 350,000)
3. Current 3-source design; more sources need architecture extension
4. Single-step MDP (no multi-turn)

### Future Work (10-Year Roadmap)
**Stage 1 (2026-2028)**: Domain ontology auto-construction, Vision integration (Quadruple-Hybrid)
**Stage 2 (2028-2031)**: PPO 4D L-DWA, real-time streaming, industry collaboration
**Stage 3 (2031-2036)**: Quintuple-Hybrid (+ Audio), Meta-learning, universal AI

### Industrial Collaboration Plan
- Pharma ontology company (2027~2031, Ⅶ.3)
- National R&D project as PM (2026.5~2031.4)

---

## 🎯 Implementation Priority Based on Thesis

### Absolute Must-Have (for Ⅵ장 numbers)
1. ✅ Vector/Graph/Ontology stores (from prior repo)
2. ✅ R-DWA (baseline, from prior repo)
3. ⭐ L-DWA (PPO) — **core novelty**
4. ⭐ Offline cache — **cost enabler**
5. ⭐ 4-benchmark evaluation — **generalization proof**

### Nice-to-Have (stretch goals)
- BERT intent classifier polishing
- Additional ablations
- Visualization dashboards

### Can Defer to Post-Submission
- Streamlit demo app
- Real-time streaming
- Additional domains
