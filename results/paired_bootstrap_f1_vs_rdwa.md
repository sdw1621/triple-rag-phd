# Paired bootstrap — F1 (rdwa as baseline)

- Method: 5,000 paired resamples, BCa-corrected 95% CI, two-sided bootstrap p-value
- Sample: query-level metric from cache eval (consistent reward function across policies)
- Effect size: Cohen's d_z (paired); |0.2|=small, |0.5|=medium, |0.8|=large
- Δ = (policy) − (baseline). Positive Δ ⇒ policy beats baseline.

## Overall (5,000 QA)

| policy | n | mean (policy) | mean (rdwa) | Δ | 95% CI (BCa) | p (2-sided) | Cohen's d_z | verdict |
|---|---|---|---|---|---|---|---|---|
| **uniform** | 5,000 | 0.0827 | 0.0713 | +0.0114 | [+0.0091, +0.0138] | < 1e-4 | +0.135 | ✅ better |
| **vector_only** | 5,000 | 0.0820 | 0.0713 | +0.0107 | [+0.0080, +0.0132] | < 1e-4 | +0.114 | ✅ better |
| **graph_only** | 5,000 | 0.0434 | 0.0713 | -0.0279 | [-0.0309, -0.0249] | < 1e-4 | -0.256 | ❌ worse |
| **ontology_only** | 5,000 | 0.0446 | 0.0713 | -0.0267 | [-0.0300, -0.0234] | < 1e-4 | -0.227 | ❌ worse |
| **oracle** | 5,000 | 0.1207 | 0.0713 | +0.0494 | [+0.0464, +0.0524] | < 1e-4 | +0.456 | ✅ better |

## Per-type breakdown

### type = `conditional`

| policy | n | Δ | 95% CI | p | d_z |
|---|---|---|---|---|---|
| uniform | 1250 | +0.0185 | [+0.0160, +0.0213] | < 1e-4 | +0.389 |
| vector_only | 1250 | +0.0181 | [+0.0157, +0.0207] | < 1e-4 | +0.398 |
| graph_only | 1250 | -0.0183 | [-0.0211, -0.0156] | < 1e-4 | -0.362 |
| ontology_only | 1250 | -0.0168 | [-0.0197, -0.0140] | < 1e-4 | -0.322 |
| oracle | 1250 | +0.0361 | [+0.0330, +0.0395] | < 1e-4 | +0.603 |

### type = `multi_hop`

| policy | n | Δ | 95% CI | p | d_z |
|---|---|---|---|---|---|
| uniform | 1750 | +0.0038 | [+0.0019, +0.0058] | < 1e-4 | +0.091 |
| vector_only | 1750 | -0.0007 | [-0.0031, +0.0015] | 0.5292 | -0.015 |
| graph_only | 1750 | -0.0108 | [-0.0132, -0.0086] | < 1e-4 | -0.225 |
| ontology_only | 1750 | -0.0083 | [-0.0105, -0.0063] | < 1e-4 | -0.184 |
| oracle | 1750 | +0.0358 | [+0.0328, +0.0389] | < 1e-4 | +0.545 |

### type = `simple`

| policy | n | Δ | 95% CI | p | d_z |
|---|---|---|---|---|---|
| uniform | 2000 | +0.0137 | [+0.0086, +0.0189] | < 1e-4 | +0.112 |
| vector_only | 2000 | +0.0161 | [+0.0104, +0.0219] | < 1e-4 | +0.119 |
| graph_only | 2000 | -0.0489 | [-0.0559, -0.0416] | < 1e-4 | -0.307 |
| ontology_only | 2000 | -0.0491 | [-0.0569, -0.0410] | < 1e-4 | -0.282 |
| oracle | 2000 | +0.0696 | [+0.0631, +0.0761] | < 1e-4 | +0.462 |
