# 📜 Conversation History — Claude Desktop → Claude Code Handoff

> This document summarizes all conversations that happened on Claude Desktop
> (web chat) before handing off to Claude Code (terminal).
> Read this first to understand the project context.

---

## 🗓️ Timeline Overview

### Phase 1 (Apr 18): Thesis Body Writing
- **4/18 04:05-10:00**: Multiple sessions writing chapters I~VII
- **Result**: 7 chapters complete (v4/v3), ~110 pages
- **Key files**: `박사논문_1장_서론_v4.docx` through `박사논문_7장_결론_v3.docx`
- **Context**: Prof. Moon's feedback "서론 강하고 결과물 약함" → Ⅵ장 대폭 강화

### Phase 2 (Apr 18 late): Figures & Integration
- **4/18 09:52-**: User requested "텍스트만 있으니 표/그림 추가"
- **Result**: 13 matplotlib figures (SVG/PNG), integrated into chapters
- **Output**: `박사논문_통합본_v5_그림포함.pdf` (78 pages, 2.82 MB)

### Phase 3 (Apr 19 morning): Docker Environment
- **4/19 06:11-**: Started environment setup for RTX 4090 PC
- **Setup**: Windows 11 + Docker Desktop (WSL2 backend)
- **Verification**: `torch.cuda.is_available() → True` on container `cca355e79865`
- **Issue encountered**: NumPy 2.x compatibility warning (`_ARRAY_API not found`)
- **Resolution**: Added `pip install "numpy<2.0"` to Dockerfile (requires rebuild)

### Phase 4 (Apr 19 afternoon): Prior Repo Integration
- **4/19 06:44-**: User uploaded `hybrid-rag-comparsion-main.zip` (291KB)
- **Discovery**: Prior repo at https://github.com/sdw1621/hybrid-rag-comparsion
- **Key finding**: Prior repo has complete implementation that we can port
- **Contents**:
  - `data/gold_qa_5000.json` (1.4MB) — 5,000 QA ready to use
  - `data/university_data.py` (14KB) — master data for 60 depts, ~600 profs
  - `data/dataset_generator.py` (24KB) — QA generation script
  - `src/vector_store.py`, `src/knowledge_graph.py`, `src/ontology_engine.py`
  - `src/query_analyzer.py`, `src/dwa.py` (R-DWA baseline)
  - `src/triple_hybrid_rag.py`, `src/evaluator.py`, `src/ablation.py`
  - `notebooks/Triple_Hybrid_RAG_Full.ipynb` (Colab notebook)
  - `run_experiment.py`, `run_hotpotqa_experiment.py`, `run_source_ablation.py`
  - Prior paper numbers: F1=0.86, EM=0.78, Faith=0.89

### Phase 5 (Apr 19 current): New PhD Repo Strategy
- **User decision**: Create separate repo `triple-rag-phd` (separate from prior)
- **Reason**: Clear separation between prior JKSCI paper and PhD thesis contributions
- **Strategy**:
  - Prior repo (`hybrid-rag-comparsion`) → External dependency (data source)
  - PhD repo (`triple-rag-phd`) → NEW contributions only (PPO, BERT, cache)
- **Output**: `triple-rag-phd-initial-setup.zip` (30 KB, 55 items) created

### Phase 6 (Apr 19 now): Claude Code Handoff
- **User decision**: Use Claude Code (terminal) for actual implementation
- **Reason**: Claude Desktop for strategy, Claude Code for file/git operations
- **This handoff package**: Complete context transfer to Claude Code

---

## 🎯 Key User Decisions Made

| # | Decision | Rationale |
|---|---|---|
| 1 | Thesis title is Korean + English | 호서대 규격, 국제 투고 대비 |
| 2 | 110 pages body target | 박사 논문 규격, 제출 가능 분량 |
| 3 | 7 chapters (Ⅰ~Ⅶ) | 호서대 박사 논문 표준 구조 |
| 4 | PPO-based L-DWA is CORE contribution | 선행 R-DWA의 한계 극복 |
| 5 | 4 benchmarks (University + 3 public) | 도메인 일반화 증명 |
| 6 | Offline cache (330K entries) | 실험 비용 $100→$15 절감 |
| 7 | Docker + WSL2 on Windows 11 | 4090 GPU 활용 + 재현성 |
| 8 | Separate PhD repo from prior repo | 기여 구분 명확화 |
| 9 | Use Claude Code for implementation | 파일 직접 조작, Git 자동화 |
| 10 | SNOMED CT applied, MeSH as backup | 라이선스 대기 대응 |

---

## 📊 Key Numbers Decided

### Thesis Results (Target, pending actual experiments)
```
Overall:
  L-DWA F1:    0.89  (baseline R-DWA: 0.86, +3.5%)
  L-DWA EM:    0.82  (baseline: 0.78, +5.1%)
  Faithfulness: 0.93 (baseline: 0.89, +4.5%)

Query Type:
  Simple EM:      0.91  (R-DWA: 0.89)
  Multi-hop EM:   0.97  (R-DWA: 0.96)
  Conditional EM: 0.93  (R-DWA: 0.91)
  Boundary EM:    0.81  (R-DWA: 0.61, +32.8%) ⭐

Cross-Benchmark:
  HotpotQA Hard F1:  0.44  (Vector baseline: 0.33, +33%)
  MuSiQue F1:        0.36  (Vector: 0.24, +50%)
  PubMedQA F1:       0.72  (MedRAG: 0.62, +16%)

Efficiency:
  API Cost:      $100 → $15 (-85%)
  Training Time: 50h → 1h (-98%)
```

### PPO Specifics
```
State dim:  18 (density 3 + BERT intent 3 + source stats 9 + meta 3)
Action dim: 3 (α, β, γ on simplex)
Reward:     R = 0.5·F1 + 0.3·EM + 0.2·Faith - 0.1·latency_penalty
Network:    Actor-Critic, shared FC backbone (2 × 64), ~6,081 params total
Episodes:   10,000
Seeds:      42, 123, 999
```

---

## 🗂️ File System State

### Author's Windows Machine
```
C:\Users\shin\
├── triple_rag_phd\             # Old folder (backup before new repo)
│   └── (Docker files only)
└── triple-rag-phd\             # NEW PhD repo (after ZIP extraction)
    └── (55 items from init ZIP)
```

### Claude Desktop Outputs Directory
```
/mnt/user-data/outputs/
├── 박사논문_1장_서론_v4.docx           # Thesis Ch.1
├── 박사논문_2장_관련연구_v4.docx        # Thesis Ch.2
├── 박사논문_3장_TripleHybridArchitecture_v4.docx  # Ch.3
├── 박사논문_4장_RuleBasedDWA_v4.docx    # Ch.4
├── 박사논문_5장_PPO_LDWA_v4.docx        # Ch.5 (CORE)
├── 박사논문_6장_실험평가_v2.docx         # Ch.6 (has § placeholder numbers)
├── 박사논문_6장_실험평가_v3_확장섹션.docx # Ch.6 extended
├── 박사논문_7장_결론_v3.docx            # Ch.7
└── triple-rag-phd-initial-setup.zip    # Initial scaffolding (used)
```

---

## 💡 Important Context for Claude Code

### 1. **This PhD is a natural extension, not from scratch**
The prior paper (Shin & Moon 2025 JKSCI) already established:
- Triple-Hybrid RAG architecture (V + G + O)
- Rule-based DWA (R-DWA)
- 5,000 QA synthetic university dataset
- Evaluation framework (F1, EM, Recall@3, Faithfulness)

The PhD thesis ADDS:
- PPO-based Learned DWA (L-DWA)
- BERT multi-label Intent Classifier
- Offline caching system (330K entries)
- 4-benchmark generalization validation

### 2. **Data comes from prior repo**
Running `data/university/download_from_prior.sh` will:
1. Git clone `sdw1621/hybrid-rag-comparsion`
2. Copy `gold_qa_5000.json` and related files
3. Run generators to rebuild graph/ontology if needed

### 3. **Thesis Ch.6 needs real numbers**
Currently Ch.6 v2 has some § (section/placeholder) marked values.
After PPO training completes, these must be replaced with actual measurements.
This is the **final critical task** before submission.

### 4. **Figures are final, don't regenerate**
13 figures in `docs/figures/` are thesis-final quality.
If changes needed, consult author before regenerating.

### 5. **Submission is Wed 2026-04-30**
Only **11 days** from handoff. Pace yourself:
- Weekdays: 1-2 modules per day
- Weekend (4/25-27): Big experiments
- Mon-Wed (4/28-30): Integration + submission

---

## 🎓 Korean Academic Context (Important!)

### 호서대 박사 논문 규격
- **용지**: 4.6배판 (188×257mm)
- **섹션 번호**: Ⅰ장 > 1절 > 가. > (1) > (가)
- **본문 글꼴**: 한글 바탕체 10.5pt, 영문 Times New Roman 10.5pt
- **줄간격**: 200%
- **각주**: 한글 바탕체 9pt
- **수식**: Cambria Math (italics)

### Korean Terms (DO NOT translate to English in thesis)
- 지도교수 (advisor)
- 논문지 (journal)
- 학위 논문 (dissertation)
- 선행 연구 (prior work)
- 근위 정책 최적화 (PPO)
- 학습형 동적 가중치 (Learned DWA)
- 규칙 기반 동적 가중치 (Rule-based DWA)
- 검색 증강 생성 (RAG)

### Code Comments & Docstrings
- **Domain terms in Korean**: 학과, 교수, 과목 etc.
- **Technical terms in English**: PPO, Actor-Critic, Embedding
- **Mix is OK**: `# BERT Intent Classifier로 질의 유형 분류`

---

## 🔗 External Links (verified as of 2026-04-19)

- Prior repo: https://github.com/sdw1621/hybrid-rag-comparsion
- New PhD repo: https://github.com/sdw1621/triple-rag-phd (to be created)
- HotpotQA: http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json
- MuSiQue: https://github.com/StonyBrookNLP/musique
- PubMedQA: https://github.com/pubmedqa/pubmedqa
- SNOMED CT: https://uts.nlm.nih.gov/uts/ (license pending approval)
- MeSH (backup): https://nlmpubs.nlm.nih.gov/projects/mesh/MESH_FILES/xmlmesh/

---

## 🎯 First Actions for Claude Code

When you (Claude Code) receive this handoff:

1. **Acknowledge context**: Confirm you've read CLAUDE.md and this file
2. **Check environment**:
   ```bash
   docker-compose ps                    # Is container running?
   docker-compose exec triple_rag python -c "
   import torch, numpy
   print(f'PyTorch: {torch.__version__}')
   print(f'NumPy: {numpy.__version__}')   # Must be 1.x
   print(f'CUDA: {torch.cuda.is_available()}')"
   ```
3. **Check data state**:
   ```bash
   ls -la /workspace/data/university/   # Should have gold_qa_5000.json
   ls -la /workspace/data/hotpotqa/     # Should have hard_300.json
   ```
4. **Read implementation specs**: Open `context/CODE_SPECS.md`
5. **Read roadmap**: Open `context/ROADMAP.md`
6. **Begin**: Start with Task 1 in ROADMAP.md

---

**End of Conversation History**

Everything above summarizes ~7 hours of planning and discussion across 7 Claude Desktop sessions (2026-04-18 to 2026-04-19). Use this to avoid asking the user to re-explain context.
