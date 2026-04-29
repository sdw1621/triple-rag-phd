# Paired bootstrap — F1 (oracle as baseline)

- Method: 5,000 paired resamples, BCa-corrected 95% CI, two-sided bootstrap p-value
- Sample: query-level metric from cache eval (consistent reward function across policies)
- Effect size: Cohen's d_z (paired); |0.2|=small, |0.5|=medium, |0.8|=large
- Δ = (policy) − (baseline). Positive Δ ⇒ policy beats baseline.

## Overall (5,000 QA)

| policy | n | mean (policy) | mean (oracle) | Δ | 95% CI (BCa) | p (2-sided) | Cohen's d_z | verdict |
|---|---|---|---|---|---|---|---|---|
| **rdwa** | 5,000 | 0.0713 | 0.1207 | -0.0494 | [-0.0524, -0.0464] | < 1e-4 | -0.456 | ❌ worse |
| **uniform** | 5,000 | 0.0827 | 0.1207 | -0.0379 | [-0.0408, -0.0352] | < 1e-4 | -0.379 | ❌ worse |
| **vector_only** | 5,000 | 0.0820 | 0.1207 | -0.0386 | [-0.0418, -0.0359] | < 1e-4 | -0.375 | ❌ worse |

## Per-type breakdown

### type = `conditional`

| policy | n | Δ | 95% CI | p | d_z |
|---|---|---|---|---|---|
| rdwa | 1250 | -0.0361 | [-0.0395, -0.0330] | < 1e-4 | -0.603 |
| uniform | 1250 | -0.0176 | [-0.0200, -0.0154] | < 1e-4 | -0.423 |
| vector_only | 1250 | -0.0180 | [-0.0206, -0.0158] | < 1e-4 | -0.414 |

### type = `multi_hop`

| policy | n | Δ | 95% CI | p | d_z |
|---|---|---|---|---|---|
| rdwa | 1750 | -0.0358 | [-0.0389, -0.0328] | < 1e-4 | -0.545 |
| uniform | 1750 | -0.0319 | [-0.0349, -0.0291] | < 1e-4 | -0.506 |
| vector_only | 1750 | -0.0365 | [-0.0397, -0.0334] | < 1e-4 | -0.534 |

### type = `simple`

| policy | n | Δ | 95% CI | p | d_z |
|---|---|---|---|---|---|
| rdwa | 2000 | -0.0696 | [-0.0761, -0.0631] | < 1e-4 | -0.462 |
| uniform | 2000 | -0.0558 | [-0.0620, -0.0500] | < 1e-4 | -0.396 |
| vector_only | 2000 | -0.0534 | [-0.0597, -0.0470] | < 1e-4 | -0.370 |
