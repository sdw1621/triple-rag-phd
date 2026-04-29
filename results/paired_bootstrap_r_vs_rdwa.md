# Paired bootstrap — R (rdwa as baseline)

- Method: 5,000 paired resamples, BCa-corrected 95% CI, two-sided bootstrap p-value
- Sample: query-level metric from cache eval (consistent reward function across policies)
- Effect size: Cohen's d_z (paired); |0.2|=small, |0.5|=medium, |0.8|=large
- Δ = (policy) − (baseline). Positive Δ ⇒ policy beats baseline.

## Overall (5,000 QA)

| policy | n | mean (policy) | mean (rdwa) | Δ | 95% CI (BCa) | p (2-sided) | Cohen's d_z | verdict |
|---|---|---|---|---|---|---|---|---|
| **uniform** | 5,000 | 0.2085 | 0.1993 | +0.0091 | [+0.0068, +0.0114] | < 1e-4 | +0.112 | ✅ better |
| **vector_only** | 5,000 | 0.1874 | 0.1993 | -0.0120 | [-0.0154, -0.0090] | < 1e-4 | -0.103 | ❌ worse |
| **graph_only** | 5,000 | 0.0795 | 0.1993 | -0.1199 | [-0.4149, -0.0455] | < 1e-4 | -0.023 | ❌ worse |
| **ontology_only** | 5,000 | 0.1639 | 0.1993 | -0.0354 | [-0.0389, -0.0320] | < 1e-4 | -0.288 | ❌ worse |
| **oracle** | 5,000 | 0.2497 | 0.1993 | +0.0504 | [+0.0479, +0.0528] | < 1e-4 | +0.570 | ✅ better |

## Per-type breakdown

### type = `conditional`

| policy | n | Δ | 95% CI | p | d_z |
|---|---|---|---|---|---|
| uniform | 1250 | +0.0107 | [+0.0061, +0.0149] | < 1e-4 | +0.136 |
| vector_only | 1250 | +0.0037 | [-0.0045, +0.0084] | 0.2460 | +0.034 |
| graph_only | 1250 | -0.0243 | [-0.0304, -0.0184] | < 1e-4 | -0.220 |
| ontology_only | 1250 | +0.0043 | [-0.0008, +0.0097] | 0.1212 | +0.044 |
| oracle | 1250 | +0.0585 | [+0.0543, +0.0632] | < 1e-4 | +0.732 |

### type = `multi_hop`

| policy | n | Δ | 95% CI | p | d_z |
|---|---|---|---|---|---|
| uniform | 1750 | +0.0081 | [+0.0046, +0.0117] | < 1e-4 | +0.108 |
| vector_only | 1750 | -0.0426 | [-0.0491, -0.0368] | < 1e-4 | -0.326 |
| graph_only | 1750 | -0.2882 | [-1.1288, -0.0767] | < 1e-4 | -0.033 |
| ontology_only | 1750 | -0.0379 | [-0.0420, -0.0337] | < 1e-4 | -0.417 |
| oracle | 1750 | +0.0507 | [+0.0471, +0.0544] | < 1e-4 | +0.640 |

### type = `simple`

| policy | n | Δ | 95% CI | p | d_z |
|---|---|---|---|---|---|
| uniform | 2000 | +0.0091 | [+0.0053, +0.0130] | < 1e-4 | +0.103 |
| vector_only | 2000 | +0.0050 | [+0.0009, +0.0094] | 0.0252 | +0.050 |
| graph_only | 2000 | -0.0324 | [-0.0375, -0.0272] | < 1e-4 | -0.278 |
| ontology_only | 2000 | -0.0581 | [-0.0651, -0.0512] | < 1e-4 | -0.381 |
| oracle | 2000 | +0.0450 | [+0.0407, +0.0495] | < 1e-4 | +0.450 |
