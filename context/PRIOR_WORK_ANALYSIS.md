# 🔄 Prior Work Analysis & Migration Plan

> How to port `sdw1621/hybrid-rag-comparsion` code into `sdw1621/triple-rag-phd`

---

## 📦 Prior Repo Structure (Confirmed via ZIP analysis)

```
hybrid-rag-comparsion/
├── README.md                          # 선행 논문 설명 (JKSCI)
├── requirements.txt                   # langchain, faiss-cpu, owlready2...
│
├── data/
│   ├── gold_qa_500.json               # 500 QA subset
│   ├── gold_qa_1000.json              # 1,000 QA subset  
│   ├── gold_qa_5000.json              # ⭐ 5,000 QA (PRIMARY)
│   ├── university_data.py             # Master data (60 depts, 600 profs)
│   ├── dataset_generator.py           # QA generation script
│   ├── extended_dataset_generator.py  # Extended version
│   └── extended_loader.py             # Unified loader
│
├── src/                               # ⭐ CORE to port
│   ├── __init__.py                    # Exports all modules
│   ├── query_analyzer.py              # Query intent (rule-based)
│   ├── dwa.py                         # R-DWA (baseline)
│   ├── vector_store.py                # FAISS wrapper
│   ├── knowledge_graph.py             # NetworkX BFS
│   ├── ontology_engine.py             # Owlready2 + HermiT
│   ├── triple_hybrid_rag.py           # Main pipeline
│   ├── evaluator.py                   # F1/EM/Recall@3/Faithfulness
│   └── ablation.py                    # Ablation study
│
├── streamlit_app/
│   └── app.py                         # Web demo (optional)
│
├── notebooks/
│   └── Triple_Hybrid_RAG_Full.ipynb   # Colab notebook
│
├── run_experiment.py                  # Main experiment runner
├── run_hotpotqa_experiment.py         # HotpotQA runner
├── run_source_ablation.py             # Ablation runner
│
├── tests/
│   ├── test_dwa.py
│   └── test_integration.py
│
└── results/hotpotqa/                  # Prior results
    ├── hotpotqa_results.csv
    └── hotpotqa_summary.json
```

---

## 🗺️ File-by-File Migration Map

### 🟢 Copy As-Is (no modification)

| Prior Repo | PhD Repo Location | Notes |
|---|---|---|
| `data/gold_qa_5000.json` | `data/university/gold_qa_5000.json` | 5K QA pairs |
| `data/gold_qa_1000.json` | `data/university/gold_qa_1000.json` | Dev subset |
| `data/gold_qa_500.json` | `data/university/gold_qa_500.json` | Smoke test |
| `data/university_data.py` | `data/university/university_data.py` | Master data |
| `data/dataset_generator.py` | `scripts/generate_qa.py` | Move to scripts/ |
| `data/extended_dataset_generator.py` | `scripts/generate_qa_extended.py` | |
| `data/extended_loader.py` | `src/rag/university_loader.py` | Utility loader |

### 🟡 Rename + Minor Refactoring

| Prior Repo | PhD Repo Location | Changes |
|---|---|---|
| `src/vector_store.py` | `src/rag/vector_store.py` | Add type hints, Google docstrings |
| `src/knowledge_graph.py` | `src/rag/graph_store.py` | Rename to match naming convention |
| `src/ontology_engine.py` | `src/rag/ontology_store.py` | Rename to match naming convention |
| `src/dwa.py` | `src/dwa/rdwa.py` | Rename to clarify it's rule-based |
| `src/query_analyzer.py` | `src/intent/rule_based.py` | Move to intent/ module |
| `src/evaluator.py` | `src/eval/metrics.py` | Split F1/EM/Faithfulness functions |

### 🔵 Extract + Adapt

| Prior Repo | PhD Repo Location | Adaptation |
|---|---|---|
| `src/triple_hybrid_rag.py` | `src/rag/triple_hybrid_rag.py` | Add pluggable DWA interface (R-DWA or L-DWA) |
| `src/ablation.py` | `src/eval/ablation.py` | Extend for L-DWA ablations |
| `run_experiment.py` | `scripts/run_experiment.py` | Refactor into CLI with argparse |
| `run_hotpotqa_experiment.py` | `scripts/run_hotpotqa.py` | Simplify |
| `run_source_ablation.py` | `scripts/run_ablation.py` | Extend |

### 🔴 NEW (PhD thesis contributions, not in prior)

| PhD Repo Location | Purpose | Thesis Reference |
|---|---|---|
| `src/intent/bert_classifier.py` | BERT multi-label intent | Ch.3, Ch.5 |
| `src/dwa/ldwa.py` | PPO-based L-DWA | Ch.5 (CORE) |
| `src/ppo/mdp.py` | MDP formulation | Ch.5 Eq. 5-1 to 5-7 |
| `src/ppo/actor_critic.py` | Policy network | Ch.5 Sec 2 |
| `src/ppo/trainer.py` | PPO training loop | Ch.5 Sec 3 |
| `src/utils/offline_cache.py` | 330K cache system | Ch.5 Sec 4, Ch.6 Sec 10 |
| `src/utils/seed.py` | Reproducibility | Ch.6 Sec 1 |
| `src/eval/benchmark.py` | 4-benchmark runner | Ch.6 |
| `src/rag/ragas_faithfulness.py` | RAGAS evaluation | Ch.6 Sec 3 |

### ⚫ Skip (not needed for thesis)

- `streamlit_app/` — demo only, not thesis content
- `notebooks/Triple_Hybrid_RAG_Full.ipynb` — replaced by proper scripts
- `results/hotpotqa/` — old results, will be regenerated

---

## 📝 Key Prior Code to Study (Referenced by Thesis)

### 1. `src/dwa.py` (R-DWA implementation)
**Critical for**: Chapter 4 thesis content, L-DWA baseline comparison

Key class: `DWA` with method `compute_weights(query_intent, density_signals)`
Returns: `DWAWeights(alpha=0.18, beta=0.19, gamma=0.63)` (NamedTuple)

### 2. `src/query_analyzer.py`
**Critical for**: Chapter 3 Query Analyzer description, density signal extraction

Key functions:
- `extract_entities(query)` → list
- `extract_relations(query)` → list
- `extract_constraints(query)` → list
- `classify_type(query)` → "simple" | "multi_hop" | "conditional"

### 3. `src/triple_hybrid_rag.py`
**Critical for**: Overall pipeline understanding

Key class: `TripleHybridRAG`
- `load_university_sample()` — loads all 3 sources
- `build()` — builds FAISS, graph, ontology
- `query(text)` → `RAGResult(answer, weights, sources_used, latency)`

### 4. `src/evaluator.py`
**Critical for**: Chapter 6 all tables

Key functions:
- `f1_score(pred, gold)` → float
- `exact_match(pred, gold)` → bool  
- `recall_at_k(retrieved, relevant, k=3)` → float
- `faithfulness(answer, contexts)` → float (RAGAS-style)

---

## 🚀 Migration Execution Plan

### Phase A: Data First (30 min)
```bash
# Inside /workspace/
cd /workspace

# Run the download script (already prepared in init ZIP)
bash data/university/download_from_prior.sh

# This will:
# 1. Clone hybrid-rag-comparsion
# 2. Copy gold_qa_5000.json, university_data.py, dataset_generator.py
# 3. Verify counts

# Expected result:
# /workspace/data/university/gold_qa_5000.json  (1.4MB)
# /workspace/data/university/university_data.py (14K)
# ...
```

### Phase B: Port Core RAG (2-3 hours)
Order matters (dependencies):

1. **`src/rag/vector_store.py`** (no dependencies)
   - Port `vector_store.py` from prior
   - Add type hints, docstrings
   - Write `tests/test_vector_store.py`

2. **`src/rag/graph_store.py`** (no dependencies)
   - Port `knowledge_graph.py` from prior
   - Rename for consistency
   - Write `tests/test_graph_store.py`

3. **`src/rag/ontology_store.py`** (no dependencies)
   - Port `ontology_engine.py` from prior
   - Write `tests/test_ontology_store.py`

4. **`src/intent/rule_based.py`** (no dependencies)
   - Port `query_analyzer.py` from prior

5. **`src/dwa/rdwa.py`** (depends on rule_based intent)
   - Port `dwa.py` from prior
   - Establish common `DWA` interface

6. **`src/eval/metrics.py`** (no dependencies)
   - Port `evaluator.py` from prior
   - Split into functions

7. **`src/rag/triple_hybrid_rag.py`** (depends on all above)
   - Port + adapt for pluggable DWA

### Phase C: New Contributions (2-3 days)

8. **`src/utils/seed.py`** (no dependencies)
   - NEW, simple utility

9. **`src/intent/bert_classifier.py`** (depends on transformers)
   - NEW, BERT multi-label

10. **`src/utils/offline_cache.py`** (depends on Triple-Hybrid)
    - NEW, SQLite-based cache

11. **`src/ppo/mdp.py`** (no dependencies)
    - NEW, State/Action/Reward classes

12. **`src/ppo/actor_critic.py`** (depends on mdp)
    - NEW, PyTorch module

13. **`src/ppo/trainer.py`** (depends on actor_critic + cache)
    - NEW, training loop

14. **`src/dwa/ldwa.py`** (depends on ppo)
    - NEW, PPO-based weight policy

15. **`src/eval/benchmark.py`** (depends on everything)
    - NEW, 4-benchmark runner

### Phase D: Scripts (0.5 day)

16. `scripts/build_cache.py`
17. `scripts/train_ppo.py`
18. `scripts/run_benchmarks.py`
19. `scripts/evaluate.py`

### Phase E: Results → Thesis (0.5 day)

20. Run experiments
21. Generate tables/figures
22. Update thesis Ch.6 § placeholders

---

## ⚠️ Prior Code Known Quirks

### 1. Hard-coded paths
Prior code may have paths like `./data/...`. In PhD repo, use `/workspace/data/...` or config-based paths.

### 2. OpenAI API version
Prior uses `openai>=1.0.0`. Current container has newer version. May need slight API adjustments.

### 3. FAISS version
Prior uses `faiss-cpu==1.7.4`. Docker uses same. Should be fine.

### 4. Korean strings
Many strings are in Korean. Keep them as-is (don't translate). They're thesis-specific domain terms.

### 5. Random seed usage
Prior may not consistently seed. PhD repo MUST seed everything (42, 123, 999).

---

## 🎯 Quick Reference: What Prior Repo Gives Us

✅ **Ready to use**:
- 5,000 QA dataset (exact same as thesis Ch.6)
- Vector/Graph/Ontology storage implementations
- R-DWA algorithm
- Evaluation metrics (F1, EM, Faithfulness)
- University domain data generation scripts

❌ **Must build ourselves** (PhD contributions):
- PPO training infrastructure
- L-DWA policy
- BERT intent classifier
- Offline cache system
- 4-benchmark runner

This separation is **intentional** and matches the thesis contribution claims (Ch.1).
