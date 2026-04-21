# Per-type breakdown — F1_strict / F1_substring / Faithfulness

Policies: R-DWA, L-DWA-seed42, Oracle
Queries per policy: 5,000

## SIMPLE

| Policy | F1_strict | F1_substring | EM | Faithfulness |
|---|---|---|---|---|
| R-DWA | 0.1530 ± 0.225 (n=2000) | 0.8369 ± 0.320 | 0.0000 | 0.9278 ± 0.253 |
| L-DWA-seed42 | 0.1738 ± 0.239 (n=2000) | 0.8611 ± 0.294 | 0.0000 | 0.9357 ± 0.242 |
| Oracle | 0.1886 ± 0.251 (n=2000) | 0.8589 ± 0.297 | 0.0000 | 0.9438 ± 0.227 |

**L-DWA vs R-DWA (simple)**: F1_strict +0.0208 (+13.6%), F1_substring +0.0242 (+2.9%), Faithfulness +0.0079 (+0.9%)

## MULTI_HOP

| Policy | F1_strict | F1_substring | EM | Faithfulness |
|---|---|---|---|---|
| R-DWA | 0.0167 ± 0.046 (n=1750) | 0.2605 ± 0.366 | 0.0000 | 0.7277 ± 0.418 |
| L-DWA-seed42 | 0.0189 ± 0.047 (n=1750) | 0.2606 ± 0.362 | 0.0000 | 0.7529 ± 0.399 |
| Oracle | 0.0303 ± 0.061 (n=1750) | 0.2635 ± 0.358 | 0.0000 | 0.7865 ± 0.389 |

**L-DWA vs R-DWA (multi_hop)**: F1_strict +0.0022 (+13.1%), F1_substring +0.0001 (+0.0%), Faithfulness +0.0252 (+3.5%)

## CONDITIONAL

| Policy | F1_strict | F1_substring | EM | Faithfulness |
|---|---|---|---|---|
| R-DWA | 0.0190 ± 0.052 (n=1250) | 0.0897 ± 0.165 | 0.0000 | 0.8059 ± 0.383 |
| L-DWA-seed42 | 0.0418 ± 0.072 (n=1250) | 0.2076 ± 0.210 | 0.0000 | 0.8953 ± 0.283 |
| Oracle | 0.0442 ± 0.078 (n=1250) | 0.1946 ± 0.205 | 0.0000 | 0.9398 ± 0.217 |

**L-DWA vs R-DWA (conditional)**: F1_strict +0.0227 (+119.2%), F1_substring +0.1179 (+131.4%), Faithfulness +0.0894 (+11.1%)

## Boundary-query implications (thesis Ch.6 §7)

The thesis's boundary-query claim targets queries that combine multi-hop
reasoning with conditional constraints. In our benchmark, multi_hop and
conditional are separate categories — the strongest evidence for the
boundary claim is the L-DWA improvement on **multi_hop** and **conditional**
relative to their respective R-DWA baselines.
