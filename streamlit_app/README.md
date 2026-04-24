# Thesis Results Explorer (Streamlit)

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/)

Read-only dashboard for the PhD thesis *"PPO-based L-DWA for Triple-Hybrid RAG"*.
Loads pre-computed artifacts from `results/` and `cache/` — **never calls the LLM**
or incurs API cost. Safe for live defense demo.

## 6 tabs

| Tab | What it shows |
|---|---|
| 📊 **Overview** | All 5 policies at a glance (R-DWA, Oracle, L-DWA ×3 seeds), per-type bar chart, headline claim box |
| 🔍 **쿼리 비교** | Filterable/sortable 50-sample side-by-side table; deep-dive per qid with each policy's weights, answer, and metrics |
| 🎛️ **가중치 시뮬레이터** | Pick a query, slide α/β/γ on Δ³ → instant reward lookup from 330K cache. Oracle (argmax) highlighted. Plotly simplex heatmap of all 66 grid points. |
| 📈 **PPO 학습** | 3-seed overlays of mean_reward / entropy / policy_loss / value_loss from `history.json`; user-adjustable moving-average window |
| 📐 **Stage-wise Baseline** | JKSCI CORRIGENDUM 3-stage fix (S0→S1→S2→S3) F1<sub>strict</sub> trajectory |
| 🌐 **Cross-domain** | HotpotQA / MuSiQue / PubMedQA + English-intent + English-synthetic benchmarks, policy × benchmark bar chart |

## Deploy to Streamlit Community Cloud (free)

1. Fork or clone this repo on GitHub (already public: `sdw1621/triple-rag-phd`).
2. Sign in at [share.streamlit.io](https://share.streamlit.io/) with GitHub.
3. Click **New app** and fill in:
   - Repository: `sdw1621/triple-rag-phd`
   - Branch: `main`
   - **Main file path: `streamlit_app/dashboard.py`** ← *important*
   - Python version: 3.10 (set in Advanced settings)
4. Streamlit Cloud will auto-detect `streamlit_app/requirements.txt` (lean:
   streamlit + plotly + pandas + numpy — no torch/transformers) and deploy
   in ~2 minutes.
5. The deployed URL looks like `https://triple-rag-phd.streamlit.app/`.

All four data dependencies below are committed directly to the repo so the
deployed app runs end-to-end without any regeneration step.

## Run (inside docker container)

```bash
docker-compose exec triple_rag bash
cd /workspace
streamlit run streamlit_app/dashboard.py --server.port 8501 --server.address 0.0.0.0
```

Then open `http://localhost:8501` in the host browser.

**Port mapping**: `docker-compose.yml` now maps `8501:8501` alongside `8888:8888`
(Jupyter) and `6006:6006` (TensorBoard). If the container was started before this
change, run `docker-compose up -d --force-recreate` once.

## Data dependencies

| Dashboard tab | Required artifact | Regenerate if missing |
|---|---|---|
| Overview / 쿼리 비교 | `results/rerun_*_list.json` (5 files) | `python scripts/evaluate_rerun.py …` (see `scripts/aggregate_list_prompt_results.py`) |
| 가중치 시뮬레이터 | `cache/university.sqlite` (330K rows) | `python scripts/build_cache.py` (~14h) |
| PPO 학습 | `cache/ppo_checkpoints/seed_{42,123,999}/history.json` | `python scripts/train_ppo.py --seed <N>` (~16 min each) |

## No LLM calls policy

The dashboard intentionally has **zero runtime LLM dependencies**. All answers,
metrics, and rewards are read from JSON / SQLite produced by M5–M8 experiments.
This makes the dashboard:

- **Free to run** (no OpenAI charges during demo)
- **Deterministic** (same sliders → same numbers)
- **Fast** (SQLite lookups are microseconds)
- **Offline-safe** (no network dependency)

If future work needs a live "type-a-query" demo, add a separate tab that calls
the LLM explicitly — keep it out of this dashboard to preserve the cost guarantee.

## Files

- `dashboard.py` — Streamlit entry point (single file, ~300 lines).

## See also

- Ch.6 §3.4 for the list-prompt analysis that produced `rerun_*_list.json`
- Ch.5 §3.5 for the PPO convergence narrative visualized in the training tab
- 부록 F for full reproduction commands
