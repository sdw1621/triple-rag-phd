# 🗓️ Development Roadmap — Task-by-Task Schedule

> Detailed work breakdown from now (2026-04-19) to submission (2026-04-30).
> Update this file as tasks complete (check box + add timestamp).

---

## 🎯 High-Level Milestones

| Milestone | Target Date | Status |
|---|---|---|
| M1: Environment + Repo Setup | 2026-04-19 | 🟡 In Progress |
| M2: Core RAG Modules (from prior) | 2026-04-20~21 | ⏳ Pending |
| M3: New Contributions (BERT, Cache) | 2026-04-22~23 | ⏳ Pending |
| M4: PPO Infrastructure | 2026-04-24 | ⏳ Pending |
| M5: Cache Build (long-running) | 2026-04-23~24 | ⏳ Pending |
| M6: PPO Training | 2026-04-26 | ⏳ Pending |
| M7: Benchmark Evaluation | 2026-04-26~27 | ⏳ Pending |
| M8: Thesis Update | 2026-04-27~28 | ⏳ Pending |
| M9: Final Integration | 2026-04-29 | ⏳ Pending |
| M10: SUBMIT 🎓 | 2026-04-30 | ⏳ Pending |

---

## 📋 Detailed Task List

### M1: Environment + Repo Setup (2026-04-19)

- [ ] **T1.1** Extract `triple-rag-phd-initial-setup.zip` to `C:\Users\shin\triple-rag-phd\`
- [ ] **T1.2** Create GitHub repo `sdw1621/triple-rag-phd` (Private)
- [ ] **T1.3** Extract Claude Code handoff package (this one) to same folder
- [ ] **T1.4** Git init + first commit + push
- [ ] **T1.5** `.env` file with API keys (OPENAI, ANTHROPIC, GOOGLE)
- [ ] **T1.6** Install Claude Code (`irm https://claude.ai/install.ps1 | iex`)
- [ ] **T1.7** Authenticate Claude Code (Pro/Max or API key)
- [ ] **T1.8** Run Claude Code in project folder, verify CLAUDE.md loads
- [ ] **T1.9** Docker rebuild with `numpy<2.0` fix
  ```powershell
  docker-compose down
  docker-compose build --no-cache
  docker-compose up -d
  docker-compose exec triple_rag python -c "import numpy; print(numpy.__version__)"
  # Expected: 1.26.x or similar (NOT 2.x)
  ```
- [ ] **T1.10** Smoke test inside container
  ```bash
  docker-compose exec triple_rag python -c "
  import torch, faiss, networkx, owlready2, transformers, openai
  print('✅ All imports OK')
  print(f'CUDA: {torch.cuda.is_available()}')
  "
  ```

### M2: Core RAG Modules (2026-04-20~21)

#### Task 2.1: Data Download
- [ ] **T2.1.1** Run `data/university/download_from_prior.sh` inside container
- [ ] **T2.1.2** Verify `gold_qa_5000.json` exists, count = 5000
- [ ] **T2.1.3** Run `data/download_public_benchmarks.sh`
- [ ] **T2.1.4** Verify HotpotQA hard_300.json, MuSiQue dev_300.json, PubMedQA pharma_300.json

#### Task 2.2: Vector Store
- [ ] **T2.2.1** Copy `hybrid-rag-comparsion/src/vector_store.py` reference
- [ ] **T2.2.2** Create `src/rag/vector_store.py` per CODE_SPECS.md
- [ ] **T2.2.3** Add type hints + Google docstrings
- [ ] **T2.2.4** Write `tests/test_vector_store.py` (3+ tests)
- [ ] **T2.2.5** Run `pytest tests/test_vector_store.py -v` → all pass
- [ ] **T2.2.6** Git commit: "feat(rag): port vector_store from prior repo"

#### Task 2.3: Graph Store
- [ ] **T2.3.1** Reference `hybrid-rag-comparsion/src/knowledge_graph.py`
- [ ] **T2.3.2** Create `src/rag/graph_store.py` per CODE_SPECS.md
- [ ] **T2.3.3** Type hints + docstrings
- [ ] **T2.3.4** Write `tests/test_graph_store.py`
- [ ] **T2.3.5** Pytest pass
- [ ] **T2.3.6** Git commit: "feat(rag): port graph_store from prior repo"

#### Task 2.4: Ontology Store
- [ ] **T2.4.1** Reference `hybrid-rag-comparsion/src/ontology_engine.py`
- [ ] **T2.4.2** Create `src/rag/ontology_store.py`
- [ ] **T2.4.3** Type hints + docstrings
- [ ] **T2.4.4** Write `tests/test_ontology_store.py`
- [ ] **T2.4.5** Pytest pass
- [ ] **T2.4.6** Git commit: "feat(rag): port ontology_store from prior repo"

#### Task 2.5: Rule-based Intent
- [ ] **T2.5.1** Reference `hybrid-rag-comparsion/src/query_analyzer.py`
- [ ] **T2.5.2** Create `src/intent/rule_based.py`
- [ ] **T2.5.3** Type hints + docstrings
- [ ] **T2.5.4** Write tests
- [ ] **T2.5.5** Git commit

#### Task 2.6: R-DWA (Baseline)
- [ ] **T2.6.1** Create `src/dwa/base.py` (abstract interface)
- [ ] **T2.6.2** Reference `hybrid-rag-comparsion/src/dwa.py`
- [ ] **T2.6.3** Create `src/dwa/rdwa.py` implementing `BaseDWA`
- [ ] **T2.6.4** Tests
- [ ] **T2.6.5** Git commit: "feat(dwa): port R-DWA baseline with abstract interface"

#### Task 2.7: Metrics
- [ ] **T2.7.1** Reference `hybrid-rag-comparsion/src/evaluator.py`
- [ ] **T2.7.2** Create `src/eval/metrics.py` (split into functions)
- [ ] **T2.7.3** Tests (especially edge cases)
- [ ] **T2.7.4** Git commit

#### Task 2.8: Triple-Hybrid RAG Pipeline
- [ ] **T2.8.1** Reference `hybrid-rag-comparsion/src/triple_hybrid_rag.py`
- [ ] **T2.8.2** Create `src/rag/triple_hybrid_rag.py` with pluggable DWA
- [ ] **T2.8.3** Integration test with R-DWA
- [ ] **T2.8.4** End-to-end smoke test with 10 QA pairs
- [ ] **T2.8.5** Git commit: "feat(rag): triple-hybrid pipeline with pluggable DWA"

### M3: New Contributions (2026-04-22~23)

#### Task 3.1: Seed Utility
- [ ] **T3.1.1** Create `src/utils/seed.py`
- [ ] **T3.1.2** Tests (verify determinism)
- [ ] **T3.1.3** Use in all other modules

#### Task 3.2: BERT Intent Classifier
- [ ] **T3.2.1** Create `src/intent/bert_classifier.py`
- [ ] **T3.2.2** Load `klue/bert-base` for Korean
- [ ] **T3.2.3** Implement multi-label head (3 classes, BCE loss)
- [ ] **T3.2.4** Training loop using gold_qa_5000 with heuristic labels
- [ ] **T3.2.5** Save/load functionality
- [ ] **T3.2.6** Tests (inference, save/load roundtrip)
- [ ] **T3.2.7** Train initial model (3 epochs, ~30 min)
- [ ] **T3.2.8** Git commit: "feat(intent): BERT multi-label classifier (novel)"

#### Task 3.3: Offline Cache
- [ ] **T3.3.1** Design SQLite schema
- [ ] **T3.3.2** Create `src/utils/offline_cache.py`
- [ ] **T3.3.3** Implement `build()` with ThreadPoolExecutor (10 workers)
- [ ] **T3.3.4** Implement `lookup()` with LRU cache
- [ ] **T3.3.5** Discretization helper (weights → int keys)
- [ ] **T3.3.6** Tests (small scale: 10 queries × 66 combos = 660 entries)
- [ ] **T3.3.7** Verify ~5 min for 660 entries → extrapolate 12h for 330K
- [ ] **T3.3.8** Git commit: "feat(utils): offline reward cache (cost-saver)"

### M4: PPO Infrastructure (2026-04-24)

#### Task 4.1: MDP Formulation
- [ ] **T4.1.1** Create `src/ppo/mdp.py`
- [ ] **T4.1.2** `State`, `Action`, `compute_reward` classes
- [ ] **T4.1.3** State extraction helper (query + intent → 18-dim tensor)
- [ ] **T4.1.4** Tests
- [ ] **T4.1.5** Git commit

#### Task 4.2: Actor-Critic Network
- [ ] **T4.2.1** Create `src/ppo/actor_critic.py`
- [ ] **T4.2.2** Match thesis Figure 5-3 (shared backbone + 2 heads)
- [ ] **T4.2.3** Orthogonal init for stable training
- [ ] **T4.2.4** Verify parameter count ≈ 6,081
- [ ] **T4.2.5** Forward pass test (output shape + simplex constraint)
- [ ] **T4.2.6** Git commit

#### Task 4.3: PPO Trainer
- [ ] **T4.3.1** Create `src/ppo/trainer.py`
- [ ] **T4.3.2** Rollout collection (uses cache!)
- [ ] **T4.3.3** GAE advantage computation
- [ ] **T4.3.4** PPO update loop (4 epochs, minibatch 8)
- [ ] **T4.3.5** TensorBoard logging
- [ ] **T4.3.6** Checkpoint save/load
- [ ] **T4.3.7** Dry-run test (100 episodes, verify loss decreases)
- [ ] **T4.3.8** Git commit

#### Task 4.4: L-DWA Policy
- [ ] **T4.4.1** Create `src/dwa/ldwa.py`
- [ ] **T4.4.2** Load trained policy, implement `compute()`
- [ ] **T4.4.3** Integration test with Triple-Hybrid RAG
- [ ] **T4.4.4** Git commit

### M5: Cache Build (2026-04-23~24, runs overnight)

- [ ] **T5.1** Final smoke test of cache builder
- [ ] **T5.2** Estimate time for full build (10 queries × 66 combos → extrapolate)
- [ ] **T5.3** Start full build: `python scripts/build_cache.py --qa gold_qa_5000.json`
- [ ] **T5.4** Overnight run (12-15 hours expected)
- [ ] **T5.5** Morning: verify `cache.sqlite` has 330,000 entries
- [ ] **T5.6** Sample check: random 10 entries look reasonable
- [ ] **T5.7** Backup cache.sqlite to safe location

### M6: PPO Training (2026-04-26)

- [ ] **T6.1** Seed 42 run (1 hour)
- [ ] **T6.2** Verify convergence curve (F1 reaches ~0.89)
- [ ] **T6.3** Seed 123 run (1 hour)
- [ ] **T6.4** Seed 999 run (1 hour)
- [ ] **T6.5** Compute mean ± std across 3 seeds
- [ ] **T6.6** Save final checkpoints
- [ ] **T6.7** Git commit: "experiment: PPO trained, 3 seeds"

### M7: Benchmark Evaluation (2026-04-26~27)

- [ ] **T7.1** Evaluate on synthetic university (R-DWA and L-DWA)
- [ ] **T7.2** Evaluate on HotpotQA Hard 300
- [ ] **T7.3** Evaluate on MuSiQue Dev 300
- [ ] **T7.4** Evaluate on PubMedQA Pharma 300 (if SNOMED CT approved)
  - Fallback: MeSH-based if SNOMED pending
- [ ] **T7.5** Generate all tables (Table 6-2 to 6-10)
- [ ] **T7.6** Generate case studies (CS-1 to CS-7)
- [ ] **T7.7** Save results to `results/`

### M8: Thesis Update (2026-04-27~28)

- [ ] **T8.1** Open `docs/박사논문_6장_실험평가_v2.docx`
- [ ] **T8.2** Replace each § placeholder with actual measurement
- [ ] **T8.3** Regenerate figures if needed (e.g., PPO convergence curve)
- [ ] **T8.4** Update Table 6-2 (overall performance)
- [ ] **T8.5** Update Table 6-3 (query type)
- [ ] **T8.6** Update Table 6-8 (cross-domain)
- [ ] **T8.7** Update Table 6-9 (λ sensitivity)
- [ ] **T8.8** Validate all tables vs source data
- [ ] **T8.9** Save as `박사논문_6장_실험평가_v3_final.docx`
- [ ] **T8.10** Consult Claude Desktop for prose refinement

### M9: Final Integration (2026-04-29)

- [ ] **T9.1** Create `박사논문_통합본_final.docx` (all 7 chapters)
- [ ] **T9.2** Add 표지 (cover page) per 호서대 규격
- [ ] **T9.3** Add 목차 (TOC with page numbers)
- [ ] **T9.4** Add 참고문헌 (references, IEEE style)
- [ ] **T9.5** Add ABSTRACT (English, 2 pages)
- [ ] **T9.6** Add 국문초록 (Korean abstract, 1 page)
- [ ] **T9.7** Add 감사의 글 (Acknowledgements, optional)
- [ ] **T9.8** Generate final PDF via LibreOffice
- [ ] **T9.9** Verify page count (~120-125 total)
- [ ] **T9.10** Visual inspection page-by-page
- [ ] **T9.11** Format check (font, spacing, margins per 호서대 규격)

### M10: SUBMIT 🎓 (2026-04-30)

- [ ] **T10.1** Final PDF review
- [ ] **T10.2** Print required copies
- [ ] **T10.3** Hoseo University thesis submission
- [ ] **T10.4** 🎉 CELEBRATION

---

## 🚨 Risk Management

### Risk R1: Cache build takes longer than 12 hours
- **Mitigation**: Start Wednesday evening, monitor progress hourly
- **Fallback**: Use smaller QA subset (gold_qa_1000) if needed
- **Trigger**: If >50% time elapsed and <30% progress, reduce scope

### Risk R2: PPO doesn't converge to target F1
- **Mitigation**: Start with dry-run, verify loss decreases
- **Fallback**: Adjust hyperparameters (lr, clip_ratio) quickly
- **Trigger**: If Seed 42 run gives F1 < 0.85 after full training, investigate

### Risk R3: SNOMED CT license not approved by 4/25
- **Mitigation**: Proceed with MeSH-based PubMedQA evaluation
- **Documentation**: Thesis already mentions MeSH as backup
- **Impact**: Minor, PubMedQA is 1 of 4 benchmarks

### Risk R4: Docker/GPU issues
- **Mitigation**: Daily smoke tests
- **Fallback**: CPU-only mode (much slower but works)
- **Trigger**: If torch.cuda.is_available() returns False, investigate immediately

### Risk R5: Last-minute thesis reformatting
- **Mitigation**: Start thesis updates 3 days before submission
- **Fallback**: Keep Word version as primary (not LaTeX)
- **Trigger**: If reformatting on 4/30 takes >2 hours, escalate

---

## 📊 Progress Tracking

```
Total tasks: 80
Completed: 0
In progress: 0
Blocked: 0
Percent complete: 0%
```

Update this section daily. Claude Code should maintain this.

---

## 🎯 Daily Standup Format

When starting each work session, Claude Code should announce:

```
📅 Today: [Date]
🎯 Milestone: M<N>
✅ Completed yesterday: T<N.M>, T<N.M>
🏃 Working on now: T<N.M>
⚠️  Blocked on: <none/description>
🎲 Plan for today: T<N.M>, T<N.M>, T<N.M>
```

---

## 📝 Notes for Claude Code

1. **Commit early, commit often**: One commit per completed task (T-level)
2. **Run tests before commit**: `pytest tests/` must pass
3. **Write commit messages in English**: Easier for GitHub search
4. **Use conventional commits**: `feat:`, `fix:`, `test:`, `docs:`, `refactor:`, `perf:`
5. **Tag milestones**: After M2 complete → `git tag v0.2-core-rag`

---

**This roadmap is a living document. Update as work progresses.**
