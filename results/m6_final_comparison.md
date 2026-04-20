# M6 PPO L-DWA — Final Results (seed 42, 3-seed training)

Date: 2026-04-20
PR: feat/m6-ppo-training

## Training trajectory (3 seeds)

| Seed | final R (ep10000) | early R | late R | Δ | Verdict |
|---|---|---|---|---|---|
| 42 | 0.2148 | 0.199 | 0.215 | +0.016 | ✅ learning |
| 123 | 0.2122 | 0.196 | 0.215 | +0.018 | ✅ learning |
| 999 | 0.2120 | 0.197 | 0.215 | +0.018 | ✅ learning |

**All 3 seeds converge to R ≈ 0.215**, extremely consistent. Std across seeds: ±0.0015.

## Headline table — R-DWA vs L-DWA vs Oracle (5,000 QA, fresh LLM calls)

| Policy | F1_strict | F1_substring (journal-style) | Faithfulness | Total R | Latency |
|---|---|---|---|---|---|
| R-DWA | 0.072 ± 0.161 | 0.448 ± 0.447 | 0.827 ± 0.362 | 0.199 | 1.19s |
| **L-DWA seed 42** | **0.087 ± 0.173** | **0.488 ± 0.430** | **0.862 ± 0.325** | **0.215** | 1.30s |
| Oracle | 0.097 ± 0.184 | 0.484 ± 0.430 | 0.888 ± 0.301 | 0.250 | 1.20s |

### Relative improvement (L-DWA vs R-DWA)
- F1_strict: **+20.8%** (0.072 → 0.087)
- F1_substring: **+8.9%** (0.448 → 0.488)
- Faithfulness: **+4.2%** (0.827 → 0.862)

### Oracle gap recovery (L-DWA positions within R-DWA→Oracle spectrum)
- F1_strict: (0.087 - 0.072) / (0.097 - 0.072) = **60% of gap closed**
- F1_substring: L-DWA (0.488) > Oracle (0.484) → Oracle was optimized for F1_strict, not substring; L-DWA's learned weights happen to favor substring-style lists (interesting finding)
- Faithfulness: (0.862 - 0.827) / (0.888 - 0.827) = **57% of gap closed**

## Journal reproducibility (JKSCI 2025)

| Metric | JKSCI reported | Our F1_strict | Our F1_substring | Substring reproduction rate |
|---|---|---|---|---|
| R-DWA F1 | 0.86 | 0.072 | **0.448** | 52% |
| R-DWA Faith | 0.89 | 0.827 | 0.827 | 93% |

**Finding**: F1_substring recovers ~52% of journal's reported R-DWA F1 (0.86 → 0.448). The remaining gap likely comes from:
- Journal using even looser evaluation (char-level overlap, or token with synonyms)
- Different LLM prompt template
- Different prior of retrieval quality

**Thesis implication**: use F1_substring as the primary metric for journal-comparable numbers; F1_strict as the secondary "strict rigor" metric showing L-DWA improves under both regimes.

## Thesis numbers to paste (revised Ch.6 targets)

### Overall (thesis Table 6-2)
| Policy | F1 (substring, journal-style) | F1 (strict) | EM | Faith | R |
|---|---|---|---|---|---|
| R-DWA (reproduced) | 0.448 ± 0.45 | 0.072 ± 0.16 | 0.000 | 0.827 ± 0.36 | 0.199 |
| **L-DWA (ours)** | **0.488 ± 0.43** | **0.087 ± 0.17** | **0.000** | **0.862 ± 0.33** | **0.215 ± 0.002** (3 seeds) |
| Oracle (upper bound) | 0.484 | 0.097 | 0.000 | 0.888 | 0.250 |

### Claims (data-grounded)
1. **L-DWA improves F1 by 20.8% (strict) / 8.9% (substring) over R-DWA** on 5,000 QA.
2. **3-seed consistency**: R_late = 0.215 ± 0.002 — highly reproducible.
3. **L-DWA recovers 60% of the Oracle gap** on strict F1, beating hand-tuned R-DWA's fixed weights.
4. **L-DWA > Oracle on F1_substring** (0.488 > 0.484) — the learned policy happens to also perform well on list-answer semantics, not just on the metric it was trained on.

### Abandoned claims (from original thesis v2)
- ~~"F1 0.89 vs 0.86 (+3.5%)"~~ — journal's 0.86 unreproducible under strict; substring gives 0.488.
- ~~"EM 0.82 vs 0.78"~~ — EM=0 everywhere due to comma-list gold format. Thesis should pivot away from EM as a headline metric.
- ~~"Boundary EM 0.61 → 0.81 (+32.8%)"~~ — need to check with F1_substring per-type breakdown (next PR).

## Weight analysis

### R-DWA mean chosen weight: (α=0.25, β=0.45, γ=0.30) — graph-biased
### L-DWA seed 42 mean chosen weight: (~0.25, ~0.30, ~0.45) — balanced, slight γ-bias
### Oracle mean chosen weight: (α=0.11, β=0.16, γ=0.73) — strongly ontology-dominant

L-DWA found a "middle ground" between R-DWA's graph-bias and Oracle's ontology-extremism. It chose to partially shift weight from β to γ but did not fully converge on (0, 0, 1). This is because:
1. R trains via cache lookup → snapping to nearest simplex point → learns smooth Dirichlet
2. PPO's entropy bonus (0.01) keeps exploration alive
3. Dirichlet mean for strongly γ-peaked concentrations still has mass on α, β

## Compute/cost summary

| Phase | Time | Cost |
|---|---|---|
| Cache build (M5) | ~14h (incl. stall) | ~$33 |
| Cache quality analysis | seconds | $0 |
| Baseline cache eval (rdwa/oracle/uniform/etc) | 5 × 5s | $0 |
| PPO training (3 seeds, CUDA) | ~30 min (parallel 123+999) | $0 |
| evaluate_rerun × 3 policies | ~20 min (parallel) | ~$5 |
| **Total M5 + M6** | **~15h wall-clock** | **~$38** |

## Open items

- [ ] Cache-eval for seeds 123 and 999 (running in background)
- [ ] Per-type (simple/multi_hop/conditional) breakdown for L-DWA rerun
- [ ] evaluate_rerun for L-DWA seed 123 and seed 999 to get 3-seed mean±std on F1_substring
- [ ] Thesis Ch.6 rewrite with these numbers (next milestone M8)

## Next-milestone recommendation

**M7 Cross-benchmark** (HotpotQA / MuSiQue / PubMedQA) is the natural next step. The demonstrated L-DWA improvement on university benchmark needs to generalize for the thesis's "domain-agnostic" claim to hold.
