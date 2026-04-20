# Offline Reward Cache — Quality Report

Produced by `scripts/cache_quality_report.py`. Used to gate M6 PPO training.

## 1. Global statistics

| Metric | Value |
|---|---|
| total_entries | 330,000 |
| n_queries | 5,000 |
| avg F1 | 0.0679 |
| avg EM | 0.0000 |
| avg Faithfulness | 0.7754 |
| avg latency (s) | 1.2369 |
| avg total R | 0.1767 |
| std total R (global) | 2.0266 |

## 2. Signal variance per query (PPO-readiness)

- **flat F1 queries** (std < 1e-6): 2,980 / 5,000 = **59.6%**
- **flat R queries** (std < 1e-6): 1,368 / 5,000 = **27.4%**

| R-std / R-range | mean | median | p90 | p99 |
|---|---|---|---|---|
| F1 std per query | 0.0349 | 0.0000 | — | — |
| R std per query | 0.0796 | 0.0412 | — | — |
| R range per query | 0.2668 | 0.2000 | 0.4278 | 0.8355 |

> **Interpretation**: PPO learns gradients from reward variance. 
> Flat-R ratio > 50% → most queries give no learning signal → PPO may not converge. 
> Median R-range > 0.05 → usable signal for at least half the queries.

## 3. Best-weight distribution (argmax of total reward per query)

- mean best α (Vector): **0.112**
- mean best β (Graph):  **0.157**
- mean best γ (Ontology): **0.731**

Top 5 most-common best weights:

| Weight | # queries |
|---|---|
| (α=0.0, β=0.0, γ=1.0) | 1834 |
| (α=0.0, β=0.1, γ=0.9) | 601 |
| (α=0.0, β=0.2, γ=0.8) | 277 |
| (α=0.3, β=0.0, γ=0.7) | 245 |
| (α=0.2, β=0.0, γ=0.8) | 175 |

## 4. Per-type breakdown

| Type | n | avg R | R-range mean | R-range median | mean best (α, β, γ) |
|---|---|---|---|---|---|
| simple | 2,000 | 0.2486 | 0.1766 | 0.0577 | (0.08, 0.12, 0.80) |
| conditional | 1,250 | 0.1710 | 0.2014 | 0.2000 | (0.14, 0.17, 0.69) |
| multi_hop | 1,750 | 0.0984 | 0.4168 | 0.2000 | (0.13, 0.19, 0.68) |

> **Thesis expectation** (Ch.4 Table 4-1, qualitative): 
> simple queries prefer α-dominant, multi_hop β-dominant, conditional γ-dominant. 
> If observed pattern matches, the reward signal aligns with prior-knowledge priors.

## 5. Correlation / variance of components

- corr(F1, Faithfulness) = **0.2225**
- global std(F1) = 0.1611
- global std(Faithfulness) = 0.4049

> High positive correlation → the two components are redundant; low/negative → independent signal.

## 6. PPO-readiness verdict

- flat-R ratio: 27.4% (threshold < 30%)
- R-range median: 0.2000 (threshold > 0.02)

✅ **PPO should be able to learn.** Proceed to M6 with default config.
