# Triple-Hybrid RAG PhD Thesis Project

> This file is automatically loaded by Claude Code at every session start.
> Do NOT delete or rename without discussing with the author.

## 🎯 Project Identity

- **Author**: Shin Dong-wook (신동욱)
- **Institution**: Hoseo University, Graduate School of Convergence Engineering
- **Advisor**: Prof. Nammee Moon (문남미 교수)
- **Thesis Title (KO)**: 근위 정책 최적화 기반 적응형 동적 가중치 학습을 통한 Triple-Hybrid RAG 프레임워크의 성능 최적화 연구
- **Thesis Title (EN)**: Performance Optimization of Triple-Hybrid RAG Framework via Proximal Policy Optimization-based Learned Dynamic Weighting
- **Submission Deadline**: 2026-04-30 (non-negotiable) 🚨
- **Target**: Korean academic thesis, 호서대 규격 (4.6배판, Ⅰ/1/가 체계), ~110 pages body

## 📚 Related Prior Work (Critical Context)

This PhD thesis **directly extends** the author's prior JKSCI 2025 paper:

- **Prior Repo**: https://github.com/sdw1621/hybrid-rag-comparsion
- **Prior Paper**: Shin & Moon (2025), "Performance Optimization Study of Hybrid RAG Engine Integrating Multi-Source Knowledge", JKSCI
- **Prior Baseline**: Triple-Hybrid RAG (Vector + Graph + Ontology) + R-DWA (Rule-based Dynamic Weighting)
  - F1: 0.86 ± 0.01, EM: 0.78 ± 0.02, Faithfulness: 0.89 ± 0.01
- **This PhD Thesis Contribution**: Replace R-DWA with **L-DWA (PPO-based Learned DWA)**
  - Target: F1: 0.89, EM: 0.82, Faithfulness: 0.93
  - Boundary query EM: 0.61 → 0.81 (+32.8%)

## 🏗️ Development Environment

### Host Machine
- **OS**: Windows 11 + Docker Desktop (WSL2 backend)
- **GPU**: NVIDIA RTX 4090 (24GB VRAM)
- **Project Path**: `C:\Users\shin\triple-rag-phd\`

### Container Environment (Docker)
- **Base**: `nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04`
- **Python**: 3.10
- **Key Constraint**: `numpy<2.0` (PyTorch 2.1.2 compat) — DO NOT upgrade
- **Work Dir**: `/workspace/` (mounted from Windows project folder)
- **Mount Strategy**: Edit files on Windows → auto-synced into container

### Critical Commands

```powershell
# From Windows PowerShell
cd C:\Users\shin\triple-rag-phd
docker-compose up -d                  # Start container
docker-compose exec triple_rag bash   # Enter container shell
docker-compose down                   # Stop container
docker-compose build --no-cache       # Rebuild after requirements.txt change
```

```bash
# Inside container
cd /workspace
python -c "import torch; print(torch.cuda.is_available())"  # Must print: True
```

## 📂 Repository Structure

```
triple-rag-phd/
├── CLAUDE.md                    # ← You are here
├── README.md                    # Public README
├── Dockerfile                   # CUDA 12.1 + PyTorch 2.1.2
├── docker-compose.yml
├── requirements.txt
├── .env                         # API keys (NEVER commit)
├── .env.example
├── .gitignore
├── LICENSE                      # MIT + prior work attribution
├── CITATION.cff
│
├── context/                     # ← Handoff context from Claude Desktop
│   ├── CONVERSATION_HISTORY.md  # What was done before this Claude Code session
│   ├── THESIS_CONTEXT.md        # PhD thesis chapter contents
│   ├── PRIOR_WORK_ANALYSIS.md   # Prior repo structure & file mapping
│   ├── CODE_SPECS.md            # Detailed specs for each module
│   └── ROADMAP.md               # Development timeline
│
├── docs/
│   ├── PROJECT_HISTORY.md       # Project-level history
│   └── figures/                 # Thesis figures (13 PNGs from Claude Desktop)
│
├── data/                        # Data (gitignored, scripts regenerate)
│   ├── university/              # Synthetic university admin (from prior repo)
│   │   └── download_from_prior.sh
│   ├── hotpotqa/                # HotpotQA hard 300
│   ├── musique/                 # MuSiQue dev 300
│   ├── pubmedqa/                # PubMedQA pharma 300
│   ├── snomed/                  # SNOMED CT (license pending)
│   ├── mesh/                    # MeSH (public, backup)
│   └── download_public_benchmarks.sh
│
├── src/                         # Core implementation
│   ├── rag/                     # Triple-Hybrid 3 sources
│   │   ├── vector_store.py      # ← Port from prior repo src/vector_store.py
│   │   ├── graph_store.py       # ← Port from prior repo src/knowledge_graph.py
│   │   ├── ontology_store.py    # ← Port from prior repo src/ontology_engine.py
│   │   └── triple_hybrid_rag.py # ← Adapt from prior repo, add L-DWA integration
│   ├── dwa/
│   │   ├── rdwa.py              # ← Port from prior repo src/dwa.py (baseline)
│   │   └── ldwa.py              # ⭐ NEW (PPO-based, thesis core contribution)
│   ├── intent/
│   │   ├── rule_based.py        # ← Port from prior repo src/query_analyzer.py
│   │   └── bert_classifier.py   # ⭐ NEW (BERT multi-label, thesis novel)
│   ├── ppo/                     # ⭐ ALL NEW (thesis core)
│   │   ├── mdp.py               # State/Action/Reward formulation
│   │   ├── actor_critic.py      # Policy network (~6K params)
│   │   └── trainer.py           # PPO training loop
│   ├── eval/
│   │   ├── metrics.py           # F1, EM, RAGAS Faithfulness
│   │   └── benchmark.py         # 4-benchmark unified evaluation
│   └── utils/
│       ├── seed.py              # Reproducibility (seeds: 42, 123, 999)
│       ├── offline_cache.py     # ⭐ NEW (330K entry cache, cost-saver)
│       └── prepare_samples.py   # Benchmark sample extraction
│
├── configs/                     # Hyperparameters
│   ├── ppo_default.yaml         # Thesis Table 5-4
│   └── domains/
│       ├── university.yaml
│       └── medical.yaml
│
├── notebooks/                   # Analysis + figure generation
├── scripts/                     # Batch execution scripts
├── results/                     # Experiment results (gitignored)
├── logs/                        # TensorBoard logs (gitignored)
├── cache/                       # Offline cache storage (gitignored)
└── tests/                       # Unit tests (pytest)
```

## 🧪 Experimental Setup (Thesis Ch. 6)

### Datasets (4 benchmarks)
1. **Synthetic University Admin** (from prior repo, primary)
   - 1,037 docs / 2,542 nodes / 6,889 edges / 5,000 QA
   - QA types: simple 2000, multi_hop 1750, conditional 1250
2. **HotpotQA Hard 300** (general QA)
3. **MuSiQue Dev 300** (complex multi-hop)
4. **PubMedQA Pharma 300** (medical, uses SNOMED CT)

### LLM & Embeddings
- **LLM**: GPT-4o-mini (`gpt-4o-mini-2024-07-18`)
- **Temperature**: 0.0 (deterministic)
- **top-p**: 1.0
- **Max tokens**: 500
- **Embedding**: text-embedding-3-small (dim=1536)

### Retrieval Config
- **Vector**: FAISS IndexFlatIP (cosine similarity), top-k=3
- **Graph**: NetworkX BFS, max_depth=3
- **Ontology**: Owlready2 + HermiT reasoner
- **Chunk size**: 1000 chars, overlap=200

### PPO Hyperparameters (Thesis Table 5-4)
```yaml
learning_rate: 3.0e-4
gae_lambda: 0.95
clip_ratio: 0.2
value_coef: 0.5
entropy_coef: 0.01
max_grad_norm: 0.5
total_episodes: 10000
rollout_per_episode: 32
update_epochs: 4
minibatch_size: 8
gamma: 0.99  # discount factor (but essentially 1-step in our setup)
```

### Reward Function (Thesis Eq. 5-7)
```python
R = 0.5 * F1 + 0.3 * EM + 0.2 * Faithfulness - 0.1 * max(0, latency - 5.0)
```

### Reproducibility
- **Seeds**: 42 (primary), 123, 999 (for mean ± std reporting)
- **3 runs minimum** for all reported numbers
- **Torch deterministic mode**: enabled

## 💻 Code Conventions

- **Python version**: 3.10
- **Type hints**: REQUIRED for all public functions
- **Docstrings**: Google style, Korean OK for domain terms
- **Formatter**: black (line-length=100)
- **Linter**: ruff (default config)
- **Logging**: Use `rich.logging` (not `print`)
- **Config management**: YAML + `pyyaml`
- **Error handling**: Explicit `try/except` with logged context

## 🚨 Critical Do's and Don'ts

### ✅ DO
- Activate environment before any Python command: `docker-compose exec triple_rag bash`
- Set random seeds explicitly at the start of every script
- Use `/workspace/` paths inside container (NOT Windows paths)
- Commit frequently with descriptive messages
- Run `pytest tests/` after every significant change
- Update `context/ROADMAP.md` when completing a milestone

### ❌ DON'T
- DO NOT commit `.env` file (check `git status` before every commit)
- DO NOT upgrade numpy to 2.x (breaks PyTorch 2.1.2)
- DO NOT use `print()` in production code (use `rich.logging`)
- DO NOT skip seed setting (thesis reproducibility requirement)
- DO NOT modify files in `docs/figures/` (they are thesis final figures)
- DO NOT create parallel repos (everything stays in `triple-rag-phd/`)

## 📊 Current Status (as of 2026-04-19)

### ✅ Completed
- Thesis body chapters I~VII (v4/v3, ~110 pages, figures included)
- Integrated PDF: 박사논문_통합본_v5_그림포함.pdf (78 pages, 2.82 MB)
- 13 thesis figures (matplotlib PNG)
- Docker environment + GPU passthrough
- Project scaffolding (this repo with 55 init files)
- Prior repo (hybrid-rag-comparsion) analyzed

### ⏳ In Progress
- SNOMED CT license approval (applied 2026-04-19, expected 4/20-22)
- Numpy<2.0 fix (Dockerfile updated, needs rebuild)

### 📋 Pending (Priority Order)
1. Port prior repo src/ modules (vector_store, graph_store, ontology_store, rdwa)
2. Implement BERT Intent Classifier (src/intent/bert_classifier.py)
3. Implement offline cache (src/utils/offline_cache.py) — **COST CRITICAL**
4. Implement PPO modules (src/ppo/, src/dwa/ldwa.py) — **THESIS CORE**
5. Build offline cache (~12h runtime)
6. Train PPO (3 seeds × 1h = 3h)
7. Run 4-benchmark evaluation (~2h)
8. Replace thesis Ch.6 § placeholders with actual numbers
9. Final thesis integration + submission (4/28-30)

## 🎯 Claude Code Role (You)

You are the **implementation agent**. Your role:

1. **Read context/** folder first to understand project history
2. **Implement code** in `src/` based on `context/CODE_SPECS.md`
3. **Write tests** in `tests/` for every module
4. **Run tests** via `docker-compose exec triple_rag pytest tests/`
5. **Commit + push** after each milestone with clear messages
6. **Update `context/ROADMAP.md`** as milestones complete
7. **Report back** when blocked or when reaching decision points

## 🔄 Workflow With Claude Desktop (Separate Tool)

The author uses **Claude Desktop (web chat)** for strategy and thesis writing.
You receive instructions via the author, who bridges between Claude Desktop and you.

- **Claude Desktop** handles: thesis editing, figure generation, strategy discussion
- **You (Claude Code)** handle: code implementation, testing, git workflow
- **GitHub**: the source of truth for all code

## 🆘 When Stuck

1. Check `context/` folder for existing guidance
2. Read `context/PRIOR_WORK_ANALYSIS.md` for prior repo reference
3. Read relevant thesis chapter in `docs/PROJECT_HISTORY.md`
4. Run `docker-compose exec triple_rag python -c "import X; help(X)"` to verify imports
5. If truly blocked: create a `BLOCKER.md` file with your question and notify the author

## 📅 Timeline (Remaining)

```
2026-04-19 (Sat): Setup, initial commit, Docker rebuild
2026-04-20 (Sun): Port prior repo src/ → src/rag/, src/dwa/rdwa.py, src/intent/rule_based.py
2026-04-21 (Mon): Develop BERT Intent Classifier, metrics, benchmark
2026-04-22 (Tue): Develop offline_cache.py + smoke test
2026-04-23 (Wed): Start offline cache build (run overnight)
2026-04-24 (Thu): Verify cache + develop PPO modules (mdp, actor_critic, trainer)
2026-04-25 (Fri): PPO dry-run + fix bugs
2026-04-26 (Sat): Full PPO training (3 seeds) + evaluation
2026-04-27 (Sun): Update thesis Ch.6 numbers
2026-04-28 (Mon): Thesis integration
2026-04-29 (Tue): Final review + format check
2026-04-30 (Wed): SUBMIT 🎓
```

---

**Last Updated**: 2026-04-19 by Claude Desktop handoff
**Next Action**: Read `context/CONVERSATION_HISTORY.md` and begin implementation
