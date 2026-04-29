# Paired bootstrap — FAITH (rdwa as baseline)

- Method: 5,000 paired resamples, BCa-corrected 95% CI, two-sided bootstrap p-value
- Sample: query-level metric from cache eval (consistent reward function across policies)
- Effect size: Cohen's d_z (paired); |0.2|=small, |0.5|=medium, |0.8|=large
- Δ = (policy) − (baseline). Positive Δ ⇒ policy beats baseline.

## Overall (5,000 QA)

| policy | n | mean (policy) | mean (rdwa) | Δ | 95% CI (BCa) | p (2-sided) | Cohen's d_z | verdict |
|---|---|---|---|---|---|---|---|---|
| **uniform** | 5,000 | 0.8436 | 0.8258 | +0.0178 | [+0.0111, +0.0247] | < 1e-4 | +0.071 | ✅ better |
| **vector_only** | 5,000 | 0.7422 | 0.8258 | -0.0836 | [-0.0937, -0.0737] | < 1e-4 | -0.228 | ❌ worse |
| **graph_only** | 5,000 | 0.6609 | 0.8258 | -0.1649 | [-0.1768, -0.1526] | < 1e-4 | -0.368 | ❌ worse |
| **ontology_only** | 5,000 | 0.7110 | 0.8258 | -0.1148 | [-0.1263, -0.1036] | < 1e-4 | -0.278 | ❌ worse |
| **oracle** | 5,000 | 0.9469 | 0.8258 | +0.1211 | [+0.1127, +0.1295] | < 1e-4 | +0.397 | ✅ better |

## Per-type breakdown

### type = `conditional`

| policy | n | Δ | 95% CI | p | d_z |
|---|---|---|---|---|---|
| uniform | 1250 | +0.0177 | [+0.0031, +0.0323] | 0.0172 | +0.066 |
| vector_only | 1250 | -0.0152 | [-0.0318, +0.0008] | 0.0656 | -0.052 |
| graph_only | 1250 | -0.0787 | [-0.1061, -0.0509] | < 1e-4 | -0.155 |
| ontology_only | 1250 | +0.0593 | [+0.0362, +0.0838] | < 1e-4 | +0.137 |
| oracle | 1250 | +0.1929 | [+0.1731, +0.2148] | < 1e-4 | +0.505 |

### type = `multi_hop`

| policy | n | Δ | 95% CI | p | d_z |
|---|---|---|---|---|---|
| uniform | 1750 | +0.0249 | [+0.0115, +0.0392] | 4.0e-04 | +0.084 |
| vector_only | 1750 | -0.2076 | [-0.2318, -0.1850] | < 1e-4 | -0.419 |
| graph_only | 1750 | -0.3672 | [-0.3906, -0.3441] | < 1e-4 | -0.742 |
| ontology_only | 1750 | -0.1738 | [-0.1914, -0.1554] | < 1e-4 | -0.444 |
| oracle | 1750 | +0.1553 | [+0.1407, +0.1707] | < 1e-4 | +0.479 |

### type = `simple`

| policy | n | Δ | 95% CI | p | d_z |
|---|---|---|---|---|---|
| uniform | 2000 | +0.0116 | [+0.0037, +0.0196] | 0.0044 | +0.064 |
| vector_only | 2000 | -0.0180 | [-0.0278, -0.0082] | 4.0e-04 | -0.081 |
| graph_only | 2000 | -0.0418 | [-0.0539, -0.0301] | < 1e-4 | -0.159 |
| ontology_only | 2000 | -0.1721 | [-0.1892, -0.1548] | < 1e-4 | -0.443 |
| oracle | 2000 | +0.0462 | [+0.0378, +0.0555] | < 1e-4 | +0.231 |
