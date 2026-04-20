# Baseline Policy Comparison — 330K Offline Reward Cache

All metrics computed via `scripts/evaluate_on_cache.py` (no new LLM calls; direct lookup against the 5,000 QA × 66 simplex-point cache).

## Overall metrics (sorted by Total Reward)

| Policy | F1 | EM | Faithfulness | Total R | Mean chosen (α, β, γ) |
|---|---|---|---|---|---|
| **Oracle** (per-query argmax) | **0.121 ± 0.199** | 0.000 | **0.947 ± 0.217** | **0.250 ± 0.114** | (0.11, 0.16, 0.73) |
| Uniform (1/3, 1/3, 1/3) | 0.083 ± 0.168 | 0.000 | 0.844 ± 0.345 | 0.209 ± 0.122 | (0.33, 0.33, 0.33) |
| **R-DWA** (Ch.4 / JKSCI 2025) | **0.071 ± 0.161** | 0.000 | 0.826 ± 0.363 | **0.199 ± 0.121** | **(0.25, 0.45, 0.30)** |
| Vector-only | 0.082 ± 0.172 | 0.000 | 0.742 ± 0.428 | 0.187 ± 0.149 | (1.00, 0.00, 0.00) |
| Ontology-only | 0.045 ± 0.146 | 0.000 | 0.711 ± 0.446 | 0.164 ± 0.126 | (0.00, 0.00, 1.00) |
| Graph-only | 0.043 ± 0.144 | 0.000 | 0.661 ± 0.467 | 0.080 ± 5.20¹ | (0.00, 1.00, 0.00) |

¹ Graph-only has outlier latency values inflating the std; median R is still ≈ 0.08.

## Per-type F1 (the thesis-critical view)

| Type | n | R-DWA F1 | Uniform F1 | **Oracle F1** | R-DWA → Oracle gap |
|---|---|---|---|---|---|
| simple | 2,000 | 0.152 | 0.165 | **0.221** | +45% |
| conditional | 1,250 | 0.019 | 0.027 | **0.055** | **+189%** |
| multi_hop | 1,750 | 0.016 | 0.022 | **0.052** | **+225%** |

> **Observation**: multi-hop and conditional queries show **2-3× headroom** over R-DWA when a perfect per-query weight is chosen. This is exactly where L-DWA should demonstrate value.

## Per-type reward

| Type | R-DWA R | Uniform R | Oracle R | R-DWA → Oracle gap |
|---|---|---|---|---|
| simple | 0.260 | 0.275 | **0.305** | +17% |
| conditional | 0.166 | 0.208 | **0.225** | +36% |
| multi_hop | 0.154 | 0.194 | **0.204** | +33% |

## Chosen-weight comparison (R-DWA vs Oracle)

| Rank | R-DWA (chose most often) | Oracle (best per query) |
|---|---|---|
| 1 | (0.2, 0.6, 0.2) × 2,898 (multi-hop default) | **(0.0, 0.0, 1.0) × 1,834 (36.7%)** |
| 2 | (0.2, 0.2, 0.6) × 945 | (0.0, 0.1, 0.9) × 601 |
| 3 | (0.6, 0.2, 0.2) × 482 | (0.0, 0.2, 0.8) × 277 |

**R-DWA average**: (α=0.25, β=0.45, γ=0.30) — **Graph-biased**
**Oracle average**: (α=0.11, β=0.16, γ=0.73) — **Ontology-biased (+141% on γ)**

> R-DWA's Ch.4 Table 4-1 base weights (simple→α, multi_hop→β, conditional→γ) systematically under-weight Ontology on this benchmark. This is quantified **miscalibration**, not just "suboptimal".

## 🔴 Journal reproducibility gap

| Metric | JKSCI 2025 (reported) | This cache-based eval | Gap |
|---|---|---|---|
| Overall F1 (R-DWA) | **0.86** | **0.071** | **12× lower** |
| Overall EM (R-DWA) | 0.78 | 0.000 | 78pp lower |
| Overall Faithfulness (R-DWA) | 0.89 | 0.826 | −6.4% |

**Possible causes** (ranked by likelihood):
1. **Strict evaluator** — our `normalize_korean` strips Korean particles; token-set F1 on comma-list gold answers like "홍성민, 황성민, 전성민" vs LLM's natural sentence output is very harsh.
2. **Gold answer format** — 5,000 QA benchmark has comma-separated-name answers (mean 3-8 names each) that don't align with LLM sentence output.
3. **Pipeline simplifications** during porting from `hybrid-rag-comparsion`.
4. Journal's F1 metric may have been defined differently (e.g., substring-based or looser tokenization).

## Thesis narrative implications

### OLD framing (thesis Ch.6 pre-experiment)
> L-DWA achieves F1 0.89 vs R-DWA 0.86 (+3.5%), EM 0.82 vs 0.78 (+5.1%), and boundary-query EM 0.81 vs 0.61 (+32.8%).

### NEW framing (data-grounded)
> **R-DWA's rule-based weight allocation is systematically miscalibrated** on the 5,000 QA benchmark: it chooses graph-dominant weights (β=0.45) while the reward landscape favors ontology-dominant weights (γ=0.73). Even a random-uniform policy outperforms R-DWA in total reward (0.209 vs 0.199). L-DWA's value proposition is therefore **not incremental refinement but qualitative correction** — recovering the 25% total-reward / 69% F1 headroom between R-DWA and the per-query oracle.

### Concrete L-DWA targets (data-calibrated)
| Metric | R-DWA (measured) | Oracle upper bound | L-DWA realistic target (~80% of Oracle gap) |
|---|---|---|---|
| Overall F1 | 0.071 | 0.121 | **0.111 (+56% relative)** |
| Overall Total R | 0.199 | 0.250 | **0.240 (+21% relative)** |
| multi-hop F1 | 0.016 | 0.052 | **0.045 (+181% relative)** |
| conditional F1 | 0.019 | 0.055 | **0.048 (+153% relative)** |

## Next steps

1. **M6 training** (in progress, seed 42): verify PPO learns toward oracle-like weights
2. **After training**: run `evaluate_on_cache.py --policy ldwa:<checkpoint>` to measure L-DWA
3. **If L-DWA > R-DWA**: proceed to seeds 123, 999
4. **If L-DWA = R-DWA**: investigate state features (likely: BERT intent_logits missing from state → M6.5 priority)
5. **If L-DWA < R-DWA**: reward shaping (EM weight wasted) or state augmentation
