"""
Thesis Results Explorer — Triple-Hybrid RAG / L-DWA

Streamlit dashboard for exploring the 5,000-QA rerun results, the 330K
offline reward cache, and the PPO training curves produced for the PhD
thesis. Uses only cached/committed artifacts — never calls the LLM or
incurs API cost. Safe for live defense demo.

Run:
    streamlit run streamlit_app/dashboard.py --server.port 8501

Inside the docker-compose container, port 8501 needs to be mapped (this
PR adds it to docker-compose.yml). Alternative: use existing 8888 via
`--server.port 8888`.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
CACHE_DIR = ROOT / "cache"
PPO_CKPT = CACHE_DIR / "ppo_checkpoints"

st.set_page_config(
    page_title="Triple-Hybrid RAG / L-DWA Explorer",
    page_icon="🎓",
    layout="wide",
)

POLICY_FILES = {
    "R-DWA": RESULTS / "rerun_rdwa_list.json",
    "Oracle": RESULTS / "rerun_oracle_list.json",
    "L-DWA (seed 42)": RESULTS / "rerun_ldwa_seed42_list.json",
    "L-DWA (seed 123)": RESULTS / "rerun_ldwa_seed123_list.json",
    "L-DWA (seed 999)": RESULTS / "rerun_ldwa_seed999_list.json",
}

POLICY_COLORS = {
    "R-DWA": "#4F86C6",
    "Oracle": "#8FB573",
    "L-DWA (seed 42)": "#E8756E",
    "L-DWA (seed 123)": "#B85C9E",
    "L-DWA (seed 999)": "#E8A23F",
}


# ---------- caching loaders ----------

@st.cache_data(show_spinner=False)
def load_policy(path: str) -> dict:
    with Path(path).open(encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_samples_df() -> pd.DataFrame:
    """Combine 'samples' arrays from all policy files into one dataframe."""
    rows: list[dict] = []
    for policy, path in POLICY_FILES.items():
        if not path.exists():
            continue
        data = load_policy(str(path))
        for s in data.get("samples", []):
            rows.append({
                "qid": s["qid"],
                "type": s["type"],
                "gold": s["gold"],
                "policy": policy,
                "answer": s["answer"],
                "alpha": s["weights"][0] / 10,
                "beta": s["weights"][1] / 10,
                "gamma": s["weights"][2] / 10,
                "f1_strict": s["metrics"]["f1_strict"],
                "f1_substring": s["metrics"]["f1_substring"],
                "f1_char": s["metrics"].get("f1_char", np.nan),
                "em_norm": s["metrics"]["em_norm"],
                "faithfulness": s["metrics"]["faithfulness"],
                "latency": s["metrics"]["latency"],
            })
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def load_aggregates() -> pd.DataFrame:
    """Per-policy, per-type aggregate table (from full 5000 evaluation)."""
    rows: list[dict] = []
    for policy, path in POLICY_FILES.items():
        if not path.exists():
            continue
        data = load_policy(str(path))
        overall = data["aggregate"]["overall"]
        rows.append({
            "policy": policy,
            "scope": "overall",
            "n": overall.get("F1_strict", {}).get("n", 0),
            "F1_strict": overall["F1_strict"]["mean"],
            "F1_substring": overall["F1_substring"]["mean"],
            "F1_char": overall.get("F1_char", {"mean": 0.0})["mean"],
            "EM": overall["EM_norm"]["mean"],
            "Faithfulness": overall["Faithfulness"]["mean"],
            "Latency": overall["Latency"]["mean"],
        })
        for t, bt in data["aggregate"]["by_type"].items():
            rows.append({
                "policy": policy,
                "scope": t,
                "n": bt.get("F1_strict", {}).get("n", 0),
                "F1_strict": bt["F1_strict"]["mean"],
                "F1_substring": bt["F1_substring"]["mean"],
                "F1_char": bt.get("F1_char", {"mean": 0.0})["mean"],
                "EM": bt["EM_norm"]["mean"],
                "Faithfulness": bt["Faithfulness"]["mean"],
                "Latency": 0.0,
            })
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def load_training_history(seed: int) -> list[dict]:
    path = PPO_CKPT / f"seed_{seed}" / "history.json"
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return json.load(f)


@st.cache_resource(show_spinner=False)
def get_cache_conn() -> sqlite3.Connection | None:
    path = CACHE_DIR / "university.sqlite"
    if not path.exists():
        return None
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True, check_same_thread=False)


@st.cache_data(show_spinner=False)
def cache_query_rewards(query_id: str) -> pd.DataFrame:
    conn = get_cache_conn()
    if conn is None:
        return pd.DataFrame()
    cur = conn.execute(
        "SELECT alpha_int, beta_int, gamma_int, f1, em, faithfulness, latency "
        "FROM rewards WHERE query_id = ?",
        (query_id,),
    )
    rows = cur.fetchall()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=["a", "b", "g", "f1", "em", "faith", "lat"])
    df["alpha"] = df["a"] / 10
    df["beta"] = df["b"] / 10
    df["gamma"] = df["g"] / 10
    df["reward"] = (
        0.5 * df["f1"] + 0.3 * df["em"] + 0.2 * df["faith"]
        - 0.1 * df["lat"].clip(lower=5.0).sub(5.0)
    )
    return df.sort_values("reward", ascending=False).reset_index(drop=True)


# ---------- tab renderers ----------

def tab_overview(agg: pd.DataFrame) -> None:
    st.header("📊 전체 성능 비교")
    st.caption("5,000 QA · list prompt · post-dff7dc1 evaluator")

    overall = agg[agg["scope"] == "overall"].set_index("policy")
    if overall.empty:
        st.warning("No aggregate data found. Run `scripts/evaluate_rerun.py` first.")
        return

    cols = st.columns(3)
    for i, policy in enumerate(["R-DWA", "L-DWA (seed 42)", "Oracle"]):
        if policy in overall.index:
            with cols[i]:
                st.metric(
                    label=policy,
                    value=f"F1_strict {overall.loc[policy, 'F1_strict']:.3f}",
                    delta=f"EM {overall.loc[policy, 'EM']:.3f}",
                )

    st.subheader("정책별 지표 (list prompt)")
    st.dataframe(
        overall.round(4),
        use_container_width=True,
    )

    st.subheader("Per-type F1_strict")
    per_type = agg[agg["scope"].isin(["simple", "multi_hop", "conditional"])]
    if not per_type.empty:
        fig = px.bar(
            per_type,
            x="scope",
            y="F1_strict",
            color="policy",
            barmode="group",
            color_discrete_map=POLICY_COLORS,
            labels={"scope": "Query type", "F1_strict": "F1_strict (mean)"},
            category_orders={"scope": ["simple", "multi_hop", "conditional"]},
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    st.info(
        "**Headline (corrected baseline 위)**: L-DWA 3-seed 평균 F1_strict = **0.562 ± 0.007**, "
        "66-점 이산 격자 argmax Oracle (0.554) 을 네 개 F1 축에서 모두 소폭 초과. "
        "Simple-only F1_strict 0.906 은 JKSCI 0.86 재현에 가까우며, "
        "Conditional 은 R-DWA 대비 +36.7% 로 본 논문의 가장 뚜렷한 per-type 개선."
    )
    st.caption(
        "세부 수치는 thesis v19 Ch.Ⅵ §3.2 / 3.3 / §4. "
        "Stage-wise 재현성 복구 궤적은 탭 📐 Stage-wise Baseline 참조."
    )


def tab_explorer(samples_df: pd.DataFrame) -> None:
    st.header("🔍 쿼리별 응답 비교")
    st.caption("50개 저장된 샘플(각 정책) 중 동일 qid 를 side-by-side 비교")

    if samples_df.empty:
        st.warning("No sample data loaded.")
        return

    col_filter1, col_filter2, col_filter3 = st.columns(3)
    with col_filter1:
        type_filter = st.multiselect(
            "Query type",
            sorted(samples_df["type"].unique()),
            default=sorted(samples_df["type"].unique()),
        )
    with col_filter2:
        metric_filter = st.selectbox(
            "Sort by",
            ["qid", "f1_strict", "f1_substring", "faithfulness", "latency"],
            index=0,
        )
    with col_filter3:
        order = st.radio("Order", ["asc", "desc"], horizontal=True)

    filtered = samples_df[samples_df["type"].isin(type_filter)].copy()
    # Pivot to one row per qid with columns per policy
    pivot_metric = filtered.pivot_table(
        index=["qid", "type", "gold"],
        columns="policy",
        values="f1_strict",
        aggfunc="first",
    ).reset_index()

    sort_col = metric_filter if metric_filter in filtered.columns else "qid"
    if sort_col in pivot_metric.columns:
        pivot_metric = pivot_metric.sort_values(
            sort_col, ascending=(order == "asc")
        )
    else:
        # sort by first policy's f1_strict
        pol_cols = [c for c in pivot_metric.columns if c in POLICY_FILES]
        if pol_cols:
            pivot_metric = pivot_metric.sort_values(
                pol_cols[0], ascending=(order == "asc")
            )

    st.dataframe(pivot_metric.round(3).head(30), use_container_width=True)

    st.divider()
    # Deep dive
    available_qids = sorted(filtered["qid"].unique(), key=lambda x: int(x))
    picked = st.selectbox("🔬 Deep dive — 쿼리 선택", available_qids)

    if picked:
        sub = filtered[filtered["qid"] == picked].set_index("policy")
        if sub.empty:
            st.warning("No data for this qid.")
            return

        st.markdown(f"**Gold**: `{sub.iloc[0]['gold']}`")
        st.markdown(f"**Type**: {sub.iloc[0]['type']}")

        for policy in sub.index:
            row = sub.loc[policy]
            with st.expander(
                f"**{policy}** — F1={row['f1_strict']:.3f}, "
                f"EM={row['em_norm']:.1f}, Faith={row['faithfulness']:.3f}",
                expanded=True,
            ):
                colA, colB = st.columns([3, 2])
                with colA:
                    st.markdown(f"**Answer**:\n\n> {row['answer']}")
                with colB:
                    st.markdown(
                        f"**Weights**: α={row['alpha']:.1f}, "
                        f"β={row['beta']:.1f}, γ={row['gamma']:.1f}"
                    )
                    st.markdown(f"Latency: {row['latency']:.2f}s")


def tab_simulator(samples_df: pd.DataFrame) -> None:
    st.header("🎛️ 가중치 시뮬레이터")
    st.caption("330K offline cache 에서 임의 쿼리 × 가중치의 즉시 보상 조회")

    conn = get_cache_conn()
    if conn is None:
        st.warning("`cache/university.sqlite` 가 없어 시뮬레이터 비활성화. "
                   "컨테이너에서 `scripts/build_cache.py` 실행 후 재시도.")
        return

    if samples_df.empty:
        st.warning("Sample data not loaded.")
        return

    qid_options = sorted(samples_df["qid"].unique(), key=lambda x: int(x))
    picked = st.selectbox("쿼리 선택", qid_options, key="sim_qid")

    if not picked:
        return

    df = cache_query_rewards(picked)
    if df.empty:
        st.warning(f"qid={picked} 의 캐시 엔트리 없음.")
        return

    sample_row = samples_df[
        (samples_df["qid"] == picked) & (samples_df["policy"] == "R-DWA")
    ].head(1)
    if not sample_row.empty:
        st.markdown(f"**Query (gold)**: `{sample_row.iloc[0]['gold']}` "
                    f"({sample_row.iloc[0]['type']})")

    colL, colR = st.columns([2, 3])
    with colL:
        st.markdown("### 수동 가중치 선택 (Δ³)")
        alpha = st.slider("α (Vector)", 0.0, 1.0, 0.33, step=0.1)
        beta = st.slider("β (Graph)", 0.0, 1.0 - alpha, 0.33, step=0.1)
        gamma = round(1.0 - alpha - beta, 2)
        st.markdown(f"**γ (Ontology)** = `{gamma:.2f}` (auto)")

        ai, bi, gi = round(alpha * 10), round(beta * 10), round(gamma * 10)
        matched = df[
            (df["a"] == ai) & (df["b"] == bi) & (df["g"] == gi)
        ]
        if not matched.empty:
            r = matched.iloc[0]
            st.metric("F1 (cached)", f"{r['f1']:.3f}")
            st.metric("EM (cached)", f"{r['em']:.3f}")
            st.metric("Faith (cached)", f"{r['faith']:.3f}")
            st.metric("Reward", f"{r['reward']:.3f}")
        else:
            st.warning("해당 가중치가 cache 그리드에 없음.")

        st.divider()
        st.markdown("### Oracle (argmax reward)")
        best = df.iloc[0]
        st.markdown(
            f"**α = {best['alpha']:.1f}, β = {best['beta']:.1f}, γ = {best['gamma']:.1f}**"
        )
        st.metric("Oracle reward", f"{best['reward']:.3f}")

    with colR:
        st.markdown("### 66-grid reward heatmap (Δ³ 이산화)")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df["beta"] + 0.5 * df["gamma"],
            y=(np.sqrt(3) / 2) * df["gamma"],
            mode="markers",
            marker=dict(
                size=16,
                color=df["reward"],
                colorscale="Viridis",
                colorbar=dict(title="reward"),
                showscale=True,
                line=dict(width=0.5, color="black"),
            ),
            text=[
                f"α={a:.1f}, β={b:.1f}, γ={g:.1f}<br>R={r:.3f}"
                for a, b, g, r in zip(df["alpha"], df["beta"], df["gamma"], df["reward"])
            ],
            hoverinfo="text",
        ))
        # triangle frame
        fig.add_trace(go.Scatter(
            x=[0, 1, 0.5, 0], y=[0, 0, np.sqrt(3)/2, 0],
            mode="lines",
            line=dict(color="black", width=1.2),
            showlegend=False, hoverinfo="skip",
        ))
        for vx, vy, label in [
            (-0.02, -0.05, "Vector (α=1)"),
            (1.02, -0.05, "Graph (β=1)"),
            (0.5, np.sqrt(3)/2 + 0.03, "Ontology (γ=1)"),
        ]:
            fig.add_annotation(x=vx, y=vy, text=label, showarrow=False, font=dict(size=11))
        fig.update_layout(
            xaxis=dict(visible=False, range=[-0.15, 1.15]),
            yaxis=dict(visible=False, range=[-0.12, 1.05], scaleanchor="x"),
            height=500, margin=dict(l=0, r=0, t=10, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)


# ---------- static snapshot data (corrected baseline stage + cross-domain) ----------

STAGEWISE_ROWS = [
    # (stage, fix, F1_strict, EM, Faith, note)
    ("S0", "JKSCI 원 주장 (재현 불가)", 0.860, 0.780, 0.890,
     "CORRIGENDUM §2 참조. 현 코드·데이터로는 재현 불가."),
    ("S1", "`normalize_korean` 구두점 제거 (dff7dc1)", 0.137, 0.000, 0.835,
     "토큰 분리 버그 수정으로 F1 약 90% 상승."),
    ("S2", "`PROMPT_TEMPLATE_LIST` gold 형식 정렬", 0.529, 0.387, 0.610,
     "프롬프트 → 콤마 리스트 출력. F1 추가 286%."),
    ("S3", "`faithfulness()` 2-branch 도입", 0.529, 0.387, 0.544,
     "리스트형 응답 per-item 검증. Faith 엄격화."),
]

CROSS_DOMAIN_ROWS = [
    # (benchmark, policy, F1_strict, F1_substring, Faithfulness, note)
    ("HotpotQA Hard 300 (영어 multi-hop)", "Vector-only", 0.101, 0.378, 0.777, ""),
    ("HotpotQA Hard 300 (영어 multi-hop)", "R-DWA", 0.096, 0.357, 0.720, ""),
    ("HotpotQA Hard 300 (영어 multi-hop)", "L-DWA (univ-trained)", 0.074, 0.249, 0.600,
     "naive transfer 실패 (−30%)"),
    ("HotpotQA Hard 300 (영어 multi-hop)", "R-DWA + EN intent", 0.132, None, None,
     "§7.5 영어 intent 추가 +38%"),
    ("HotpotQA Hard 300 (영어 multi-hop)", "L-DWA + EN intent", 0.110, None, None,
     "§7.5 영어 intent 추가 +49%"),
    ("MuSiQue Dev 300 (4-hop 영어)", "Vector-only", 0.056, 0.145, 0.480, ""),
    ("MuSiQue Dev 300 (4-hop 영어)", "R-DWA", 0.046, 0.123, 0.415, ""),
    ("MuSiQue Dev 300 (4-hop 영어)", "L-DWA (univ-trained)", 0.038, 0.099, 0.358,
     "naive transfer 실패"),
    ("PubMedQA Pharma 300 (biomedical)", "Vector-only", 0.231, 0.006, 0.812, ""),
    ("PubMedQA Pharma 300 (biomedical)", "R-DWA", 0.214, 0.004, 0.757, ""),
    ("PubMedQA Pharma 300 (biomedical)", "L-DWA (univ-trained)", 0.149, 0.005, 0.546, ""),
    ("English synthetic univ (403 QA)", "R-DWA", 0.663, 0.609, 0.958,
     "§7.6 영어 합성 — 한국어 0.529 보다 높음"),
    ("English synthetic univ (403 QA)", "L-DWA (EN retrain)", 0.661, 0.607, 0.955,
     "§7.7 state 미구분 → R-DWA 수준 수렴"),
    ("English synthetic univ (403 QA)", "L-DWA (EN) → HotpotQA", 0.149, None, None,
     "영어 코퍼스 기반 평균 가중치가 약간 유리"),
]


def tab_stagewise() -> None:
    st.header("📐 Stage-wise Corrected Baseline")
    st.caption(
        "JKSCI 2025 논문의 재현성 장애 3 건을 단계별로 해소한 궤적. "
        "본 논문 §Ⅵ.3.1 의 stage-wise 표와 일치."
    )

    df = pd.DataFrame(
        STAGEWISE_ROWS,
        columns=["Stage", "수정 내용", "F1_strict", "EM", "Faithfulness", "비고"],
    )

    st.dataframe(df, use_container_width=True, hide_index=True)

    st.subheader("F1_strict 궤적 (R-DWA 기준)")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["Stage"],
        y=df["F1_strict"],
        mode="lines+markers+text",
        line=dict(color="#4F86C6", width=3),
        marker=dict(size=14, color="#4F86C6"),
        text=[f"{v:.3f}" for v in df["F1_strict"]],
        textposition="top center",
        name="F1_strict",
    ))
    fig.update_layout(
        height=380,
        yaxis=dict(title="F1_strict", range=[0.0, 0.95]),
        xaxis=dict(title="Stage"),
        margin=dict(l=0, r=0, t=10, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.info(
        "S0 은 선행 논문 게재 수치로 현 저장소에서는 재현되지 않는다. "
        "S1 (+90%) · S2 (+286%) 은 평가 인프라 정비의 기여이며, "
        "PPO 학습 기여는 이 위에서 +6.2% (L-DWA 0.562 vs R-DWA 0.529) 로 별도 계산된다. "
        "[CORRIGENDUM](https://github.com/sdw1621/hybrid-rag-comparsion/blob/main/CORRIGENDUM.md)."
    )


def tab_cross_domain() -> None:
    st.header("🌐 Cross-domain 실험")
    st.caption(
        "§Ⅵ.7 교차 도메인 결과 — naive transfer 실패 원인 분해 (언어·어휘·아키텍처) 와 "
        "후속 실험 (영어 intent 추가 · 영어 합성 벤치마크 · 영어 PPO 재학습)."
    )

    df = pd.DataFrame(
        CROSS_DOMAIN_ROWS,
        columns=["Benchmark", "Policy", "F1_strict", "F1_substring", "Faithfulness", "비고"],
    )

    benches = ["(전체)"] + sorted(df["Benchmark"].unique())
    picked = st.selectbox("벤치마크 선택", benches, index=0)

    view = df if picked == "(전체)" else df[df["Benchmark"] == picked]
    st.dataframe(view, use_container_width=True, hide_index=True)

    st.subheader("F1_strict 비교")
    plot_df = df.dropna(subset=["F1_strict"]).copy()
    fig = px.bar(
        plot_df,
        x="Policy",
        y="F1_strict",
        color="Benchmark",
        barmode="group",
        text=plot_df["F1_strict"].round(3),
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        height=460, xaxis_tickangle=-25,
        margin=dict(l=0, r=0, t=10, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.info(
        "**해석.** (i) 언어 장벽은 영어 intent 패턴 추가만으로 L-DWA +49%, R-DWA +38% 회복 (§7.5). "
        "(ii) 도메인 어휘는 영어 합성 벤치마크에서 R-DWA 0.663 으로 한국어 0.529 보다 오히려 높음 (§7.6). "
        "(iii) Graph·Ontology 부재는 재학습으로도 우회 불가 — Triple-Hybrid 의 본질적 경계 (§7.4)."
    )


def tab_training() -> None:
    st.header("📈 PPO 학습 곡선 (3 seeds)")
    st.caption("cache/ppo_checkpoints/seed_{42,123,999}/history.json 기반")

    smoothing = st.slider("Moving-average window", 1, 200, 50, step=10)

    metrics = ["mean_reward", "entropy", "policy_loss", "value_loss"]
    metric_choice = st.selectbox("지표 선택", metrics, index=0)

    fig = go.Figure()
    for seed, color in [(42, "#C00000"), (123, "#4F86C6"), (999, "#5B9B5E")]:
        hist = load_training_history(seed)
        if not hist:
            continue
        vals = [h.get(metric_choice, 0.0) for h in hist]
        x = list(range(len(vals)))
        fig.add_trace(go.Scatter(
            x=x, y=vals, mode="lines",
            line=dict(color=color, width=0.8), opacity=0.25,
            name=f"seed {seed} (raw)",
            showlegend=False,
        ))
        if smoothing > 1 and len(vals) >= smoothing:
            w = np.ones(smoothing) / smoothing
            s = np.convolve(vals, w, mode="valid")
            fig.add_trace(go.Scatter(
                x=list(range(len(vals) - len(s), len(vals))),
                y=s, mode="lines",
                line=dict(color=color, width=2.5),
                name=f"seed {seed}",
            ))
        else:
            fig.add_trace(go.Scatter(
                x=x, y=vals, mode="lines",
                line=dict(color=color, width=2.5),
                name=f"seed {seed}",
            ))
    fig.update_layout(
        height=500,
        xaxis_title="PPO update step",
        yaxis_title=metric_choice,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.info(
        "세 seed 모두 **mean_reward 0.215 ± 0.002** 로 수렴 — "
        "본 논문 §Ⅴ.3.5 의 재현성 주장을 시각화."
    )


# ---------- main ----------

def main() -> None:
    st.title("🎓 Triple-Hybrid RAG — L-DWA Explorer")
    st.caption(
        "박사학위 논문 \"PPO 기반 L-DWA 를 통한 Triple-Hybrid RAG 성능 최적화\" · "
        "Shin Dong-wook, Hoseo University · 2026 · read-only dashboard (no LLM calls) · "
        "thesis v19"
    )

    agg = load_aggregates()
    samples = load_samples_df()

    tabs = st.tabs([
        "📊 Overview",
        "🔍 쿼리 비교",
        "🎛️ 가중치 시뮬레이터",
        "📈 PPO 학습",
        "📐 Stage-wise Baseline",
        "🌐 Cross-domain",
    ])

    with tabs[0]:
        tab_overview(agg)
    with tabs[1]:
        tab_explorer(samples)
    with tabs[2]:
        tab_simulator(samples)
    with tabs[3]:
        tab_training()
    with tabs[4]:
        tab_stagewise()
    with tabs[5]:
        tab_cross_domain()

    st.divider()
    st.caption(
        "본 저장소: https://github.com/sdw1621/triple-rag-phd · "
        "CORRIGENDUM (선행 저장소): "
        "https://github.com/sdw1621/hybrid-rag-comparsion/blob/main/CORRIGENDUM.md"
    )


if __name__ == "__main__":
    main()
