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


# ---------- landing header + glossary (shown on every page) ----------


def render_landing_header() -> None:
    """Top banner shown above all tabs. Explains what this dashboard is."""
    st.title("🎓 Triple-Hybrid RAG — L-DWA 연구 대시보드")
    st.markdown(
        "**박사학위 논문** — 「근위 정책 최적화 기반 적응형 동적 가중치 학습을 통한 "
        "Triple-Hybrid RAG 프레임워크의 성능 최적화 연구」  \n"
        "**저자** 신동욱 · **지도교수** 문남미 · **소속** 호서대학교 대학원 융합공학과 · **연도** 2026"
    )
    st.caption(
        "이 대시보드는 논문 실험 결과를 탐색하기 위한 **읽기 전용** 페이지입니다. "
        "LLM 을 실시간으로 호출하지 않으므로 접속 시 비용이 발생하지 않고 즉시 반응합니다. "
        "처음 방문하셨다면 아래의 **📖 시작하기** 탭부터 보시면 전체를 이해하기 쉽습니다."
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric(
            label="L-DWA F1_strict (3-seed 평균)",
            value="0.562",
            delta="+6.2% vs R-DWA (0.529)",
            help="본 논문이 제안하는 학습형 정책의 5,000 QA 평균 엄격 F1. "
                 "세 개의 서로 다른 random seed (42, 123, 999) 에서 평균 ± 0.007.",
        )
    with c2:
        st.metric(
            label="Oracle 상한 (이산 격자 argmax)",
            value="0.554",
            delta="L-DWA 가 소폭 초과",
            delta_color="inverse",
            help="66개 가중치 조합 중 질의마다 최고점을 고르는 가상의 상한선 정책. "
                 "L-DWA 가 이 상한을 네 개 F1 지표 모두에서 미세하게 넘습니다. "
                 "연속 정책이 이산 격자 바깥의 가중치를 활용한다는 해석.",
        )
    with c3:
        st.metric(
            label="학습 비용 절감",
            value="≈ $33",
            delta="−85% (vs $288 직접 학습)",
            delta_color="inverse",
            help="330K 엔트리의 오프라인 보상 캐시를 한 번 구축 ($33, 14h) 하면 "
                 "이후 PPO 학습은 LLM 호출 0회로 수행 가능. "
                 "RL-based RAG 튜닝의 진입 비용 장벽을 개인 연구자 수준으로 낮춤.",
        )

    st.divider()


def render_sidebar_glossary() -> None:
    """Sidebar glossary of domain terms — always accessible."""
    with st.sidebar:
        st.markdown("# 📘 용어집")
        st.caption("도메인이 낯선 분을 위한 5분 가이드. 각 항목을 클릭해서 펼치세요.")

        with st.expander("RAG (Retrieval-Augmented Generation)"):
            st.markdown(
                "LLM 이 답을 만들기 전에 **외부 지식을 먼저 검색** 하여 그 결과를 "
                "참고 맥락으로 제공하는 기법. LLM 단독 답변의 약점 (환각, 최신성 부족) "
                "을 완화합니다."
            )

        with st.expander("Triple-Hybrid (세 가지 지식 소스)"):
            st.markdown(
                "- **Vector**: 문서를 임베딩으로 인코딩하고 코사인 유사도로 검색 (FAISS)\n"
                "- **Graph**: 엔티티-관계로 이루어진 지식 그래프에서 BFS 로 경로 탐색 (NetworkX)\n"
                "- **Ontology**: class / property 기반 온톨로지에서 추론기로 답 도출 (Owlready2 + HermiT)\n\n"
                "세 소스는 서로 강점이 다르므로 **함께 쓰면 상보적** 이라는 것이 "
                "Triple-Hybrid 구조의 전제입니다."
            )

        with st.expander("α, β, γ (세 소스 가중치)"):
            st.markdown(
                "세 소스의 상대적 중요도. 합이 1 이 되도록 제약되므로, "
                "**Δ³ 라는 삼각형 안의 한 점** 으로 표현됩니다. "
                "α + β + γ = 1. 예를 들어 (0.6, 0.2, 0.2) 는 Vector 중심, "
                "(0.2, 0.2, 0.6) 은 Ontology 중심을 의미합니다."
            )

        with st.expander("R-DWA (규칙 기반, 선행 JKSCI 2025)"):
            st.markdown(
                "질의를 **simple / multi_hop / conditional** 세 유형으로 규칙 판별하고, "
                "유형별로 미리 정해진 표에서 가중치를 꺼내 쓰는 방식. "
                "학습 데이터가 필요 없다는 장점이 있으나, 유형 내부의 미세한 차이는 "
                "반영하지 못합니다. 본 논문의 **비교 기준선**."
            )

        with st.expander("L-DWA (학습 기반, 본 논문 제안)"):
            st.markdown(
                "질의의 특성을 **18차원 벡터** 로 요약한 뒤, 작은 신경망 "
                "(Actor-Critic, 5,636 파라미터) 이 α·β·γ 를 출력. "
                "**PPO** 알고리즘으로 10,000 에피소드 학습. "
                "R-DWA 와 달리 질의마다 가중치를 동적으로 바꿀 수 있습니다."
            )

        with st.expander("PPO (Proximal Policy Optimization)"):
            st.markdown(
                "OpenAI 가 2017년 제안한 강화학습 알고리즘. "
                "정책 업데이트 폭을 **clip ratio 0.2** 로 제한하여 학습이 발산하지 않도록 안정화. "
                "ChatGPT 의 RLHF 단계에도 쓰이는 사실상의 표준."
            )

        with st.expander("Oracle (상한선 진단용 가상 정책)"):
            st.markdown(
                "각 질의마다 **66개 가중치 조합을 전부 시도** 해 본 뒤 가장 좋은 보상을 "
                "주는 조합을 거꾸로 선택하는 **가상의 상한선 정책**. "
                "정답을 미리 알아야 해서 실제 배포는 불가능하지만, "
                "\"이 상태 공간에서 이론적으로 얼마나 잘할 수 있는가\" 를 측정하는 진단 도구로 사용."
            )

        with st.expander("F1_strict / F1_substring / F1_char"):
            st.markdown(
                "한국어 답변의 정확도를 서로 다른 세 관점에서 본 F1:\n\n"
                "- **strict** (엄격): 조사·구두점 제거 후 토큰 집합 F1\n"
                "- **substring** (부분문자열): gold 항목이 답변에 등장하는 비율\n"
                "- **char** (문자 3-gram): 토큰 경계를 무시한 문자 단위 F1\n\n"
                "동일 응답이어도 세 지표가 다른 값을 내는 경우가 있으며, "
                "이는 한국어 QA 평가의 **형식 민감도** 를 드러냅니다."
            )

        with st.expander("Faithfulness (환각 탐지)"):
            st.markdown(
                "답변의 각 주장이 **검색된 맥락 안에 실제로 등장하는가** 를 확인하는 지표. "
                "모델이 없는 사실을 만들어 내는 **환각 (hallucination)** 을 잡아냅니다. "
                "본 논문은 문장형 응답과 리스트형 응답을 나누어 평가하는 "
                "**2-branch faithfulness** 를 도입하였습니다."
            )

        st.divider()
        st.markdown("## 📚 관련 자료")
        st.markdown(
            "- [본 저장소](https://github.com/sdw1621/triple-rag-phd)\n"
            "- [CORRIGENDUM (선행 저장소)]"
            "(https://github.com/sdw1621/hybrid-rag-comparsion/blob/main/CORRIGENDUM.md)\n"
            "- 박사학위 논문: 신동욱, 호서대학교, 2026"
        )
        st.caption("문의: `sdw19@hanmail.net`")


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


def tab_about() -> None:
    """First tab: orientation for first-time visitors (esp. thesis committee)."""
    st.header("📖 시작하기 — 이 대시보드 둘러보기")

    st.markdown(
        """
### 이 대시보드는 무엇인가요?

박사학위 논문 「**근위 정책 최적화 기반 적응형 동적 가중치 학습을 통한 Triple-Hybrid
RAG 프레임워크의 성능 최적화 연구**」의 실험 결과를, 정적인 표나 그래프가 아니라
**대화형** 으로 탐색할 수 있도록 만든 페이지입니다.

모든 수치는 실험 단계에서 한 번 계산된 뒤 파일에 저장된 결과이며, 이 대시보드가
다시 계산하지는 않습니다. 그래서 접속한다고 해서 **LLM 비용이 발생하지 않고**,
클릭 반응도 즉시 이루어집니다.
"""
    )

    st.subheader("연구를 30초로 정리하면")
    st.markdown(
        """
RAG (Retrieval-Augmented Generation) 시스템은 LLM 이 답을 만들기 전에 외부 지식을
검색해 참고 맥락으로 붙여 주는 구조입니다. 본 연구가 기반으로 삼는 **Triple-Hybrid
RAG** 는 이 검색 단계에서 **세 가지 지식 소스를 동시에 사용**합니다.

- **Vector** 검색 — 문장 임베딩 기반 유사도 검색
- **Graph** 검색 — 지식 그래프 경로 탐색
- **Ontology** 추론 — 클래스·관계로 정의된 지식베이스 추론

문제는 **세 소스를 얼마씩 섞을지** (α·β·γ 가중치) 를 결정하는 일입니다. 선행
JKSCI 2025 논문은 질의 유형에 따라 고정된 표를 쓰는 규칙 기반 방식 (**R-DWA**)
을 제안하였습니다. 본 논문은 이 결정을 **강화학습 (PPO) 으로 대체** 하여, 질의마다
가중치를 학습된 신경망이 계산하는 **L-DWA** 를 제안합니다.
"""
    )

    st.subheader("핵심 결과 세 가지")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            "**① 이산 Oracle 상한 초과**  \n"
            "66-점 이산 격자 위에서 질의마다 최고점을 고르는 Oracle 정책 "
            "(F1<sub>strict</sub> 0.554) 을 L-DWA 의 3-seed 평균 (0.562) 이 "
            "네 개 F1 축 모두에서 소폭 넘습니다. 연속 Dirichlet 평균이 격자 "
            "바깥을 활용한다는 해석.",
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            "**② 조건부 질의에서 +36.7%**  \n"
            "질의 유형별로 보면 conditional (조건부) 유형에서 R-DWA 0.223 대비 "
            "L-DWA 0.304 로 상대 개선이 가장 큽니다. 규칙 기반 방식이 조건 처리에서 "
            "미세 구분이 어려웠던 부분을 학습이 채웁니다."
        )
    with c3:
        st.markdown(
            "**③ 개인 연구자 수준의 학습 비용**  \n"
            "330K 엔트리 오프라인 보상 캐시 ($33, 14시간) 를 한 번 구축하면 "
            "이후 PPO 학습은 **LLM 호출 0회**. 전체 비용을 $288 → $33 (−85%) 로 "
            "낮추어 RL-based RAG 튜닝의 진입 장벽을 해소."
        )

    # ------------------------------------------------------------------
    # 한눈에 보이는 결과 — 유형별 개선 막대 그래프
    # ------------------------------------------------------------------
    st.subheader("📊 한 그림으로 본 전체 결과")
    st.markdown(
        "아래 그래프가 **이 논문의 실험 결과를 한 장에 압축한 그림** 입니다. "
        "가로축은 **질의 유형** (전체/단순/다단계/조건부), 세로축은 **F1<sub>strict</sub> 점수** "
        "(높을수록 좋음). 세 정책 — 규칙 기반 **R-DWA** (파랑), 본 논문 제안 **L-DWA** (빨강), "
        "이론적 상한 **Oracle** (초록) — 의 막대를 나란히 배치했습니다.",
        unsafe_allow_html=True,
    )

    headline_data = pd.DataFrame([
        {"유형": "전체 (5,000)", "R-DWA": 0.529, "L-DWA": 0.562, "Oracle": 0.554},
        {"유형": "단순 (2,000)", "R-DWA": 0.874, "L-DWA": 0.906, "Oracle": 0.901},
        {"유형": "다단계 (1,750)", "R-DWA": 0.354, "L-DWA": 0.365, "Oracle": 0.380},
        {"유형": "조건부 (1,250)", "R-DWA": 0.223, "L-DWA": 0.304, "Oracle": 0.290},
    ])
    headline_long = headline_data.melt(id_vars=["유형"], var_name="정책", value_name="F1_strict")
    fig_headline = px.bar(
        headline_long,
        x="유형",
        y="F1_strict",
        color="정책",
        barmode="group",
        text=headline_long["F1_strict"].round(3),
        color_discrete_map={"R-DWA": "#4F86C6", "L-DWA": "#E8756E", "Oracle": "#8FB573"},
        category_orders={"유형": ["전체 (5,000)", "단순 (2,000)", "다단계 (1,750)", "조건부 (1,250)"]},
    )
    fig_headline.update_traces(textposition="outside")
    fig_headline.update_layout(
        height=420,
        yaxis=dict(title="F1_strict (0 ~ 1, 높을수록 좋음)", range=[0, 1.0]),
        xaxis=dict(title="질의 유형 (괄호 안은 질의 개수)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    st.plotly_chart(fig_headline, use_container_width=True)
    st.caption(
        "※ 각 막대 위의 숫자가 F1_strict 평균값. 빨간 막대 (L-DWA) 가 파란 막대 (R-DWA) 보다 "
        "높은 정도가 이 논문의 개선 폭이며, 초록 막대 (Oracle) 와의 격차가 \"아직 남은 여지\" 입니다. "
        "**조건부** 에서 빨강이 파랑을 가장 크게 앞서고, 초록 (Oracle) 마저 넘는 것이 가장 뚜렷한 결과."
    )

    # ------------------------------------------------------------------
    # Before / After 구체 예시 — "어떻게 하면 어떻게 되는지"
    # ------------------------------------------------------------------
    st.subheader("🎯 \"가중치를 이렇게 바꾸면, 답이 이렇게 달라진다\"")
    st.markdown(
        "수치만 봐서는 감이 오지 않으므로, **논문 Ch.Ⅵ 의 케이스 스터디에서 실제 질의 두 개** "
        "를 뽑아 R-DWA 와 L-DWA 가 각각 어떤 선택을 하고, 그 결과 답변이 어떻게 달라지는지 "
        "나란히 보여 드립니다."
    )

    # Case A — Simple query, 둘 다 정답
    with st.container(border=True):
        st.markdown("#### 예시 A — 단순 질의 (CS-1)")
        st.markdown(
            "**질의**: *\"최재원 교수의 소속 학과는?\"*  \n"
            "**정답 (gold)**: `바이오의공학과`"
        )
        colA, colB = st.columns(2)
        with colA:
            st.markdown(
                "**🧾 R-DWA 가 고른 것**  \n"
                "가중치: (α=0.6, β=0.2, γ=0.2) — simple 유형의 기본값  \n\n"
                "답변 → `바이오의공학과`  \n"
                "점수 → **F1_strict = 1.00**, EM = 1.00  \n\n"
                "*규칙 표의 기본값만으로 충분한 쉬운 질의*"
            )
        with colB:
            st.markdown(
                "**🤖 L-DWA 가 고른 것**  \n"
                "가중치: (α=0.5, β=0.4, γ=0.1) — 학습된 연속 정책이 자체 판단  \n\n"
                "답변 → `바이오의공학과`  \n"
                "점수 → **F1_strict = 1.00**, EM = 1.00  \n\n"
                "*L-DWA 도 같은 답. 단순 질의에서는 두 정책 모두 완벽*"
            )
        st.info(
            "📌 **교훈.** 쉬운 질의에서는 R-DWA 도 충분히 잘 합니다. "
            "L-DWA 의 진짜 강점은 다음 예시처럼 **조건 처리가 필요한 질의** 에서 드러납니다."
        )

    # Case B — Conditional query aggregate
    with st.container(border=True):
        st.markdown("#### 예시 B — 조건부 질의 집계 (1,250개)")
        st.markdown(
            "**질의 예**: *\"기계공학과 소속 55세 이하 교수는?\"* 같은 **나이·소속 제약** 이 "
            "포함된 질의. 단순 이름 매칭으로는 풀 수 없고, **조건을 논리적으로 만족하는 답** 을 "
            "찾아야 합니다. 이런 질의가 벤치마크에 1,250 개 있습니다."
        )
        colA, colB = st.columns(2)
        with colA:
            st.markdown(
                "**🧾 R-DWA 의 기본 가중치**  \n"
                "유형: conditional → (α=0.2, β=0.2, γ=0.6)  \n"
                "→ 1,250 개 질의에 **항상 같은 가중치** 적용  \n\n"
                "→ F1_strict 평균 **0.223**  \n"
                "*조건 처리에 Ontology (γ) 를 높게 줬지만, 질의마다 필요한 정도가 다른데 고정값만 씀*"
            )
        with colB:
            st.markdown(
                "**🤖 L-DWA 의 학습된 연속 정책**  \n"
                "질의마다 다른 α·β·γ 출력  \n"
                "→ 수치 제약·열거형 제약·배타 조건을 각각 다르게 취급  \n\n"
                "→ F1_strict 평균 **0.304** (+36.7% vs R-DWA)  \n"
                "*Oracle 의 0.290 조차 넘어섬. 본 논문의 가장 뚜렷한 per-type 증거*"
            )
        st.info(
            "📌 **교훈.** 같은 \"조건부 질의\" 라고 해도 내부 성격이 다양한데, "
            "규칙 표는 이들을 하나로 묶어 처리합니다. **학습된 정책이 이 내부 차이를 잡아내어** "
            "한 유형에서 +36.7% 의 상대 개선을 내는 것이 본 논문의 핵심 성과입니다."
        )

    # Case C — Oracle exceedance (the headline claim)
    with st.container(border=True):
        st.markdown("#### 예시 C — 전체 평균에서 Oracle 넘기 (가장 강력한 주장)")
        st.markdown(
            "**Oracle** 은 \"66개 가중치 조합 중 정답에 가장 가까운 것\" 을 항상 고르는 전지전능한 "
            "가상 심판입니다. 원칙적으로 이산 격자 안에서는 이보다 더 잘할 수 없습니다."
        )
        cc1, cc2, cc3 = st.columns(3)
        with cc1:
            st.metric("R-DWA", "0.5293", "출발점", delta_color="off")
        with cc2:
            st.metric("Oracle (이산 상한)", "0.5540", "+4.7% vs R-DWA", delta_color="off")
        with cc3:
            st.metric("L-DWA (본 논문)", "0.5620", "Oracle 초과 +0.8%", delta_color="normal")
        st.info(
            "📌 **교훈.** Oracle 은 66개 격자 안의 답만 볼 수 있는 반면, L-DWA 는 "
            "Dirichlet 분포의 평균값으로 **격자 바깥의 연속 지점** 도 쓸 수 있습니다. "
            "이 차이가 +0.8%p 로 드러납니다. 개선 폭이 큰 것은 아니지만, \"이산 상한을 "
            "넘는다\" 는 관찰은 **양적 개선이 아닌 질적 의미** 를 가집니다."
        )

    st.markdown(
        "더 많은 사례가 궁금하시면 **🔍 쿼리 비교** 탭에서 동일 질의에 대해 다섯 정책이 내린 "
        "결정을 직접 비교할 수 있고, **🎛️ 가중치 탐색기** 탭에서는 여러분이 직접 가중치를 "
        "움직여 어떤 조합이 어떤 보상을 주는지 실시간 확인할 수 있습니다."
    )

    st.subheader("탭 구성 — 논문을 따라 읽어 내려가는 순서")
    st.markdown(
        """
이 대시보드의 탭은 논문의 장 (章) 구조를 따라 **이야기 순서** 로 배치되어 있습니다.
처음 방문하셨다면 위에서 아래로 한 번 훑어 보시면 논문 전체가 어떻게 전개되는지
감이 잡힙니다.

**Part 1 — 왜 이 문제인가**

- 🧩 **문제** (Ch.1 · Ch.3) — Triple-Hybrid RAG 의 구조, α · β · γ 가 무엇인지,
  왜 질의마다 최적 가중치가 다른지를 예시와 함께 설명.

**Part 2 — 방법론**

- 📜 **R-DWA (기존)** (Ch.4) — 선행 JKSCI 2025 의 2단계 규칙 기반 방법과
  그 네 가지 한계.
- 🧠 **L-DWA (제안)** (Ch.5) — 1-step MDP 공식화, 5,636 파라미터 Actor-Critic,
  PPO 학습, 그리고 비용 문제를 해결한 오프라인 보상 캐시.

**Part 3 — 결과**

- 📊 **결과 요약** (Ch.6 §3-4) — 다섯 정책의 전체 F1 / EM / Faithfulness 비교.
- 🔍 **쿼리 비교** — 동일 질의에 대해 정책별 답이 어떻게 달라지는지 샘플 단위 확인.
- 🎛️ **가중치 탐색기** — 여러분이 직접 슬라이더를 움직여 330K 캐시에서 즉시 보상 조회.
- 📈 **PPO 학습** — 세 개 seed 의 수렴 곡선, 재현성 확인.

**Part 4 — 검증**

- 📐 **재현성 복구** (Ch.6 §3.1) — 선행 논문 F1 0.86 이 현 저장소에서 왜 재현되지
  않았고, 세 단계로 어떻게 복구되는지 (**CORRIGENDUM** 배경).
- 🌐 **교차 도메인** (Ch.6 §7) — 이 방법론이 HotpotQA · MuSiQue · PubMedQA 에서도
  통하는지, 실패 원인의 세 축 분해.

**Part 5 — 맺음말**

- 🔭 **결론** (Ch.7) — 네 가지 기여, 솔직하게 보고하는 한계, 다음 연구자에게 남기는 방향.

화면 **왼쪽 사이드바** 에는 RAG · Triple-Hybrid · α·β·γ · PPO · Oracle ·
Faithfulness 등의 **용어집** 이 접혀 있습니다. 낯선 용어가 나오면 사이드바를
함께 펼쳐 놓고 보시기를 권장드립니다.
"""
    )

    st.subheader("무엇이 '이 논문의 기여' 인가요?")
    st.markdown(
        """
- **방법론**: RAG 맥락에서 세 지식 소스 가중치 결정을 1-step MDP 로 공식화하고
  PPO 로 학습한 최초 사례.
- **엔지니어링**: 오프라인 보상 캐시 구조를 도입하여 RL 기반 RAG 튜닝의 비용을
  개인 연구자 수준으로 낮춤.
- **재현성**: 선행 JKSCI 2025 논문의 평가 코드에서 세 가지 결함 (`normalize_korean`
  의 구두점 미처리, 프롬프트-gold 형식 불일치, Faithfulness 단일 분기) 을 식별하여
  선행 저장소에 **CORRIGENDUM** 으로 공개 정정. 📐 Stage-wise Baseline 탭에서
  세부 복구 과정을 볼 수 있습니다.
- **실증**: 5,000 QA 위에서 L-DWA 가 R-DWA 대비 일관 개선을 보이고, 특히 이산
  Oracle 상한을 넘는 지점을 관찰.
- **한계의 정직한 보고**: 한국어로 학습된 L-DWA 를 영어 벤치마크에 그대로 적용하면
  성능이 떨어지며, 그 원인을 언어·어휘·아키텍처 세 축으로 분해하여 각각 독립
  실험으로 확인 (🌐 Cross-domain 탭).
"""
    )

    st.info(
        "💡 **TIP** — 특정 숫자나 그래프의 의미가 궁금하시면 위젯 옆의 물음표 아이콘 "
        "(❓) 에 마우스를 올려 보세요. 각 요소에 대한 설명이 툴팁으로 나타납니다."
    )


def tab_problem() -> None:
    """Ch.1 + Ch.3 — problem motivation, Triple-Hybrid structure, α·β·γ."""
    st.header("🧩 문제 — 왜 가중치가 어려운가")

    st.markdown(
        """
### RAG 의 기본 아이디어부터

대형 언어모델 (LLM) 은 사전 학습 데이터에 없는 최신 정보나 내부 문서에 대한
질문에 약합니다. 그래서 **질문에 답하기 전에 관련 문서를 먼저 검색해서 LLM 에
참고 맥락으로 붙여 주는 방식** 이 널리 쓰입니다. 이를 **RAG (Retrieval-Augmented
Generation)** 라고 부릅니다. 일반적인 RAG 는 문서를 벡터로 바꾸어 저장하고,
질의도 벡터로 바꾸어 유사도가 높은 상위 k 개의 문서 조각을 가져옵니다.
"""
    )

    st.subheader("왜 세 개의 소스를 써야 하는가 — Triple-Hybrid")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            "**① Vector 검색**  \n"
            "의미 기반 유사도 검색. *\"딥러닝 연구 교수\"* 처럼 어휘가 다양하게 변해도 "
            "잘 찾습니다. 하지만 정확한 관계 (A 가 B 에 소속) 를 따지는 데는 약합니다."
        )
    with c2:
        st.markdown(
            "**② Graph 검색**  \n"
            "엔티티-관계로 된 지식 그래프에서 경로를 따라 탐색. *\"컴공과 소속 교수들\"* "
            "같은 소속 질의, *\"A 와 B 가 공동 지도한 학생\"* 같은 멀티홉 질의에 강합니다."
        )
    with c3:
        st.markdown(
            "**③ Ontology 추론**  \n"
            "클래스·속성 기반 온톨로지에서 규칙 추론. *\"55세 이하\"*, *\"전임 교수만\"* 같은 "
            "**조건부 제약** 을 논리적으로 만족하는 답을 찾을 때 강합니다."
        )

    st.markdown(
        """
세 소스는 서로 강점이 다릅니다. **어떤 질의는 한 소스만 봐도 충분** 하고,
**다른 질의는 세 소스를 적절히 섞어야** 올바른 답이 나옵니다. 이 "섞는 비율" 이
바로 본 논문의 핵심 질문입니다.
"""
    )

    st.subheader("α · β · γ 가 무엇인가")
    st.markdown(
        r"""
세 소스의 기여도를 $\alpha$, $\beta$, $\gamma$ 로 표기합니다. 각각 Vector, Graph,
Ontology 의 가중치이며 **합이 1** 이 되도록 제약됩니다 ($\alpha + \beta + \gamma = 1$).
이 제약 때문에 $(\alpha, \beta, \gamma)$ 는 **삼각형 $\Delta^3$ 안의 한 점** 으로
표현됩니다. 세 꼭짓점은 각각 한 소스만 100% 쓰는 극단에 대응합니다.

예를 들어:
- $(0.6, 0.2, 0.2)$ — Vector 가 주도, Graph/Ontology 는 보조
- $(0.2, 0.2, 0.6)$ — Ontology 중심 (조건부 추론이 필요한 질의)
- $(\tfrac{1}{3}, \tfrac{1}{3}, \tfrac{1}{3})$ — 완전 균등 (baseline fallback)
"""
    )

    # Mini illustration of Δ³ simplex
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[0, 1, 0.5, 0], y=[0, 0, np.sqrt(3)/2, 0],
        mode="lines",
        line=dict(color="black", width=1.5),
        showlegend=False, hoverinfo="skip",
    ))
    example_points = [
        (0.6, 0.2, 0.2, "Vector 중심"),
        (0.2, 0.6, 0.2, "Graph 중심"),
        (0.2, 0.2, 0.6, "Ontology 중심"),
        (1/3, 1/3, 1/3, "균등"),
    ]
    for a, b, g, label in example_points:
        x = b + 0.5 * g
        y = (np.sqrt(3)/2) * g
        fig.add_trace(go.Scatter(
            x=[x], y=[y],
            mode="markers+text",
            marker=dict(size=16, color="#E8756E"),
            text=[f" {label}<br> ({a:.2f}, {b:.2f}, {g:.2f})"],
            textposition="middle right",
            showlegend=False,
            hoverinfo="skip",
        ))
    for vx, vy, label in [
        (-0.02, -0.05, "Vector (α=1)"),
        (1.02, -0.05, "Graph (β=1)"),
        (0.5, np.sqrt(3)/2 + 0.04, "Ontology (γ=1)"),
    ]:
        fig.add_annotation(x=vx, y=vy, text=label, showarrow=False, font=dict(size=12))
    fig.update_layout(
        xaxis=dict(visible=False, range=[-0.25, 1.4]),
        yaxis=dict(visible=False, range=[-0.12, 1.05], scaleanchor="x"),
        height=360, margin=dict(l=0, r=0, t=10, b=10),
        title="그림. Δ³ 삼각형 위의 가중치 점 네 개 예시",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("핵심 어려움 — 질의마다 최적 점이 다르다")
    st.markdown(
        """
만약 **모든 질의에 공통으로 좋은 하나의 가중치** 가 있다면 이 문제는 훨씬 쉬울
것입니다. 그러나 실험적으로 그렇지 않습니다. 아래는 같은 합성 대학 벤치마크에서
뽑은 세 가지 질의의 이상적 가중치입니다.
"""
    )
    example_queries = pd.DataFrame([
        {"질의": "최재원 교수의 소속 학과는?", "유형": "simple",
         "이상적 α": 0.6, "이상적 β": 0.2, "이상적 γ": 0.2,
         "왜": "이름 매칭만으로 충분 → Vector"},
        {"질의": "배가 포함된 프로젝트 참여 학과는?", "유형": "multi_hop",
         "이상적 α": 0.2, "이상적 β": 0.6, "이상적 γ": 0.2,
         "왜": "엔티티 → 프로젝트 → 학과 경로 탐색 → Graph"},
        {"질의": "기계공학과 소속 55세 이하 교수는?", "유형": "conditional",
         "이상적 α": 0.1, "이상적 β": 0.1, "이상적 γ": 0.8,
         "왜": "연령 · 소속 제약 추론 → Ontology"},
    ])
    st.dataframe(example_queries, use_container_width=True, hide_index=True)

    st.markdown(
        """
요약하면, **질의마다 다른 점을 골라야** 하고, 그것도 **질의의 특성을 보고
자동으로 결정** 해야 합니다. 다음 탭 (📜 R-DWA) 에서 선행 JKSCI 2025 논문이 이
문제를 어떻게 해결했는지, 그리고 어떤 한계가 있었는지부터 살펴봅니다.
"""
    )

    st.info(
        "💡 **관련 논문 장** — Ch.Ⅰ 서론 §1-2, Ch.Ⅲ 아키텍처 §1-3. "
        "용어가 낯설면 왼쪽 사이드바의 📘 용어집을 함께 펼쳐 보세요."
    )


def tab_rdwa() -> None:
    """Ch.4 — R-DWA rule-based approach + 4 limitations."""
    st.header("📜 R-DWA — 선행 연구의 규칙 기반 접근")

    st.markdown(
        """
선행 JKSCI 2025 논문 (Shin & Moon, 2025) 이 제안한 **Rule-based Dynamic Weighting
Algorithm (R-DWA)** 는 가중치를 **두 단계의 규칙** 으로 결정합니다. 별도 학습이
필요 없어 즉시 배포 가능하다는 장점이 있습니다.
"""
    )

    st.subheader("2단계 결정 구조")
    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown(
            """
**Stage 1 — 질의 유형 판별.** 한국어 정규식 기반 분석기 (`QueryAnalyzer`) 가
질의를 세 유형 중 하나로 분류합니다:

- **simple** — 단일 엔티티 속성 질의
- **multi_hop** — 여러 엔티티를 잇는 관계 질의
- **conditional** — 조건 (나이, 소속 등) 을 포함한 질의

**Stage 2 — 유형별 기본 가중치.** 유형에 따라 미리 정해진 표에서 $(\\alpha, \\beta, \\gamma)$ 를 꺼내 옵니다:
"""
        )
    with c2:
        base_weights = pd.DataFrame([
            {"유형": "simple", "α (Vector)": 0.6, "β (Graph)": 0.2, "γ (Ontology)": 0.2},
            {"유형": "multi_hop", "α (Vector)": 0.2, "β (Graph)": 0.6, "γ (Ontology)": 0.2},
            {"유형": "conditional", "α (Vector)": 0.2, "β (Graph)": 0.2, "γ (Ontology)": 0.6},
        ])
        st.dataframe(base_weights, use_container_width=True, hide_index=True)
        st.caption("Table 4-1. R-DWA 기본 가중치")

    st.markdown(
        r"""
**Stage 3 — 밀도 기반 미세 조정.** 질의에서 추출한 **관계 밀도** $s_r$, **제약 밀도** $s_c$
로 기본 가중치를 연속 조정합니다 ($\lambda = 0.3$, grid search 로 선정):

$$
\alpha' = \alpha_{base} \times (1 - \lambda \cdot (s_r + s_c) / 2)
$$
$$
\beta' = \beta_{base} + \lambda \cdot s_r \cdot (1 - \beta_{base})
$$
$$
\gamma' = \gamma_{base} + \lambda \cdot s_c \cdot (1 - \gamma_{base})
$$

마지막으로 합이 1 이 되도록 정규화합니다.
"""
    )

    st.subheader("구체 예시 — 조건부 질의에 R-DWA 가 결정하는 가중치")
    st.markdown(
        "질의: **\"기계공학과 소속 55세 이하 교수는?\"**"
    )
    calc_steps = pd.DataFrame([
        {"단계": "(1) QueryAnalyzer", "계산": "type=conditional, (s_r, s_c) = (0.20, 0.20)", "값": ""},
        {"단계": "(2) 기본 가중치", "계산": "conditional →", "값": "α=0.20, β=0.20, γ=0.60"},
        {"단계": "(3a) α'", "계산": "0.20 × (1 − 0.3 × (0.20+0.20)/2)", "값": "0.188"},
        {"단계": "(3b) β'", "계산": "0.20 + 0.3 × 0.20 × 0.80", "값": "0.248"},
        {"단계": "(3c) γ'", "계산": "0.60 + 0.3 × 0.20 × 0.40", "값": "0.624"},
        {"단계": "(4) 정규화", "계산": "합계 1.060 으로 나눔", "값": "(0.178, 0.234, 0.588)"},
    ])
    st.dataframe(calc_steps, use_container_width=True, hide_index=True)

    st.subheader("R-DWA 가 부딪힌 네 가지 한계")
    st.markdown(
        """
실험을 해 보니 규칙 기반 접근은 네 가지 구조적 한계에 부딪혔습니다. 이것이
본 논문이 학습 기반 접근으로 가는 **동기** 가 됩니다.

**① 도메인 편향된 기본 가중치.** Table 4-1 의 대각선 배치 (simple→α, multi_hop→β,
conditional→γ) 는 직관에 기반한 사전 분포였지만, Oracle 분석 결과 실제 보상
지형은 Ontology (γ) 쪽으로 훨씬 쏠려 있었습니다. Simple 질의조차 Ontology
비중이 높은 가중치에서 더 좋은 점수가 나오는 경우가 많았습니다.

**② 단일 $\\lambda$ 하이퍼파라미터.** grid search 로 고른 $\\lambda = 0.3$ 은 전체 질의
분포에 대한 평균적 최적값일 뿐, 개별 질의의 길이 · 엔티티 수 · 부정어 포함 여부
같은 특성은 반영하지 못합니다.

**③ 유형 내 미세 구분 부재.** 같은 conditional 이라도 수치 제약 (*40세 이하*),
열거형 제약 (*A 또는 B*), 배타 조건 (*~를 제외한*) 은 각기 다른 처리를 필요로
하지만, R-DWA 는 이들을 모두 같은 $(0.2, 0.2, 0.6)$ 으로 취급합니다.

**④ 도메인 이전 불가.** Table 4-1 과 $\\lambda$ 는 합성 대학 도메인에서 튜닝된 값이라,
재튜닝 없이 다른 도메인으로 옮기면 성능이 바로 떨어집니다. HotpotQA 에서 R-DWA
F1<sub>strict</sub> 는 0.097 로, 같은 조건의 Vector-only 0.102 에도 미치지 못합니다.
""",
        unsafe_allow_html=True,
    )

    st.info(
        "💡 **관련 논문 장** — Ch.Ⅳ R-DWA 전체. 재현 수치는 Ch.Ⅵ §5 를 참조하세요. "
        "네 가지 한계를 정면으로 다루는 방법은 다음 탭 (🧠 L-DWA) 에서 이어집니다."
    )


def tab_ldwa() -> None:
    """Ch.5 — L-DWA learning-based approach (MDP, PPO, offline cache)."""
    st.header("🧠 L-DWA — 본 논문의 학습 기반 접근")

    st.markdown(
        """
R-DWA 의 네 가지 한계를 정면으로 다루는 가장 자연스러운 방법은, **규칙 표를
작은 신경망으로 대체** 하는 것입니다. 질의의 특성을 입력으로 받아 $(\\alpha, \\beta, \\gamma)$
를 출력하는 정책을 **데이터로부터 학습** 하면, 유형 내 미세 구분도, 도메인
편향의 자동 보정도 가능해집니다.
"""
    )

    st.subheader("① 문제를 MDP 로 공식화")
    st.markdown(
        r"""
본 논문은 가중치 결정을 **1-step Markov Decision Process** 로 정의합니다.

- **State** (18차원): 질의 메타 (길이·엔티티 수·제약 수), intent logits (3차원, 향후 BERT
  확장 자리), 세 소스의 밀도 신호 ($s_e, s_r, s_c$), retrieval 점수의 통계량 등.
- **Action** (3-simplex $\Delta^3$): 출력 $(\alpha, \beta, \gamma)$ 로 합이 1 인 연속 벡터.
  Dirichlet 분포에서 샘플링.
- **Reward** (scalar):

$$
R = 0.5 \cdot F_1^{strict} + 0.3 \cdot EM + 0.2 \cdot Faithfulness - 0.1 \cdot \max(0, \text{latency} - 5)
$$

- **에피소드 길이**: 1. 가중치를 한 번 내면 retrieval · generation 이 일어나고
  즉시 보상이 결정되므로 다단계 transition 이 없습니다.
"""
    )

    st.subheader("② 작은 Actor-Critic 네트워크")
    arch = pd.DataFrame([
        {"레이어": "Input", "크기": "18", "활성화": "—"},
        {"레이어": "FC1 (shared)", "크기": "18 → 64", "활성화": "Tanh"},
        {"레이어": "FC2 (shared)", "크기": "64 → 64", "활성화": "Tanh"},
        {"레이어": "Actor head", "크기": "64 → 3 (Dirichlet α_i)", "활성화": "Softplus"},
        {"레이어": "Critic head", "크기": "64 → 1", "활성화": "—"},
    ])
    st.dataframe(arch, use_container_width=True, hide_index=True)
    st.caption("총 파라미터 수 = **5,636** . 스마트폰 수준의 경량 정책.")

    st.subheader("③ PPO 로 학습")
    st.markdown(
        """
Actor 는 Dirichlet 분포에서 $(\\alpha, \\beta, \\gamma)$ 를 샘플링, Critic 은 상태 가치를 예측합니다.
학습은 OpenAI 의 **Proximal Policy Optimization (PPO)** 로 진행합니다. clip
ratio 0.2, entropy coef 0.01, 10,000 에피소드. 세 개의 서로 다른 random seed
(42, 123, 999) 로 각각 학습한 뒤 그 평균을 보고합니다.
"""
    )

    st.subheader("④ 그런데 학습 비용은? — 오프라인 보상 캐시")
    st.markdown(
        """
강화학습의 큰 난관은 **매 에피소드마다 LLM 을 호출해야 한다** 는 점입니다.
10,000 에피소드 × 3 seed × 평균 32-rollout 을 naive 하게 하면 1백만 번이 넘는
LLM 호출이 필요하고, gpt-4o-mini 기준으로도 약 $288 가 듭니다. 개인 연구자에게
부담스러운 수준입니다.

본 논문은 이 문제를 **"보상을 미리 전부 계산해서 파일에 담아 두는"** 단순한
아이디어로 우회합니다:

1. 5,000 질의 × 66 이산 가중치 조합 = **330,000 엔트리** 의 보상을 한 번 계산.
2. 계산 결과 $(\\alpha, \\beta, \\gamma) \\to (F_1, EM, Faith, \\text{latency})$ 를 SQLite
   에 저장 (16.8 MB).
3. 이후 PPO 학습은 이 캐시를 **조회** 하므로 LLM 호출 0 회.

캐시 구축 자체는 한 번의 **$33 · 14시간** 투자. 이후 학습 · 재학습은 공짜가
됩니다. 전체 비용이 $288 → $33 로 **85% 절감** 되고, 이것이 본 연구를 개인
연구자 수준에서 반복 가능하게 만들어 준 핵심 엔지니어링 선택입니다.
"""
    )

    st.info(
        "💡 **관련 논문 장** — Ch.Ⅴ L-DWA §1-5. "
        "다음 탭 (📊 결과 요약) 부터는 이 학습된 정책이 실제로 얼마나 잘 작동하는지 살펴봅니다."
    )


def tab_conclusion() -> None:
    """Ch.7 — Conclusion: contributions, limitations, future work."""
    st.header("🔭 결론 — 기여와 한계")

    st.markdown(
        """
본 논문이 무엇을 주장하고, 무엇을 하지 못했고, 다음 연구자가 무엇을 이어 가면
좋을지 한 페이지로 정리합니다.
"""
    )

    st.subheader("이 논문의 네 가지 기여")
    st.markdown(
        """
- **방법론** — Triple-Hybrid RAG 의 가중치 결정을 1-step MDP 로 공식화하고
  PPO 로 학습한 최초 사례. Self-RAG, Adaptive-RAG 같은 기존 RAG-RL 연구가
  retrieval/generation 의 self-reflection 이나 router 에 집중한 반면, 본
  논문은 **복수 소스의 연속 가중치 결정** 자체를 정책의 출력으로 삼습니다.

- **엔지니어링** — 오프라인 보상 캐시 (330K 엔트리, 16.8 MB) 로 RL 기반 RAG
  튜닝의 학습 비용을 **$288 → $33 (−85%)** 로 낮추었습니다. 개인 연구자 수준에서
  이 문제를 반복 실험할 수 있게 해 준 실용적 장치.

- **재현성** — 선행 JKSCI 2025 논문의 평가 코드에서 세 가지 결함 (`normalize_korean`
  의 구두점 미처리, prompt-gold 형식 불일치, Faithfulness 단일 분기) 을 식별하여
  선행 저장소에 **CORRIGENDUM** 으로 공개 정정하였습니다. Stage-wise 복구 궤적
  (S0 → S1 → S2 → S3) 은 📐 재현성 탭에서 확인할 수 있습니다.

- **실증** — 5,000 QA 위에서 L-DWA 3-seed 평균이 R-DWA 대비 F1<sub>strict</sub>
  +6.2%, Faithfulness +6.6% 개선. 특히 66-점 이산 격자 argmax 로 정의된
  **Oracle 상한** (F1<sub>strict</sub> 0.554) 을 네 개 F1 지표 모두에서 미세하게
  초과하며 (0.562), 연속 정책이 이산 격자 바깥의 가중치를 활용한다는 관찰을
  처음으로 제시합니다. 유형별로는 conditional 에서 +36.7% 로 개선 폭이 가장 큽니다.
""",
        unsafe_allow_html=True,
    )

    st.subheader("솔직하게 보고하는 한계")
    st.markdown(
        """
- **한국어 단일 도메인 학습.** L-DWA 의 end-to-end 검증은 한국어 합성 대학 도메인
  하나에서만 수행하였습니다. 교차 도메인 실험 (🌐 탭) 은 transfer feasibility
  와 원인 분해까지만 제공합니다.

- **State 품질의 결정성.** 영어 합성 벤치마크에서 L-DWA 가 R-DWA 를 넘지 못한
  결과는, state 가 질의 간에 구분되지 않으면 학습형 정책의 이점이 사라진다는
  사실을 분명히 보여 줍니다. 다국어 state 설계가 선행 조건.

- **Reward 식의 EM 항.** list prompt 도입 전에는 EM 이 구조적으로 0 이었고,
  도입 후에도 0.3·EM 항이 reward 에 기여하는 비중은 작습니다. 향후 reward
  재설계가 필요한 부분.

- **BERT intent classifier 미학습.** 구현만 제시되어 있고 intent_logits 는
  (0, 0, 0) 으로 고정. 특히 conditional 유형에서 추가 개선 여지.

- **교차 도메인 naive transfer 실패.** Vector-only 기준선 대비 30~35% 하락.
  세 축으로 분해한 원인과 후속 실험은 🌐 탭에서 확인할 수 있습니다.
"""
    )

    st.subheader("다음 연구자에게 남기는 방향")
    st.markdown(
        """
- **다국어 state feature** — multilingual BERT 기반 intent encoder 또는
  retrieval 점수만으로 구성된 domain-invariant state.
- **도메인별 fine-tuning 레시피 체계화** — 작은 per-domain cache + 추가 PPO 의
  비용-효과 비교 (HotpotQA, MuSiQue, PubMedQA 에서).
- **Reward 재설계** — EM 가중치 0.3 을 F1 (list-aware) 또는 Faithfulness 로 재분배.
- **Joint multi-domain training** — 복수 도메인 캐시 결합 학습으로 도메인별
  암묵적 signature 형성 여부 검증.
- **선행 저장소와의 평가기 통일** — `hybrid-rag-comparsion` 에 본 논문의 버그
  수정을 적용하여 F1 정의를 양 프로젝트에서 통일.
- **더 큰 Actor-Critic** — 현 5,636 파라미터 네트워크의 표현력 한계와 스케일링
  효과 조사.
"""
    )

    st.subheader("연구 과정에서 얻은 두 가지 교훈")
    st.markdown(
        """
이 연구를 정리하면서 결과 자체보다 과정에서 배운 두 가지가 있습니다.

하나는 **평가 코드를 다시 들여다본 경험** 입니다. 마무리 단계에서 F1 수치가
기대보다 낮다는 인상을 받아 `normalize_korean` 을 역추적한 끝에 쉼표와 마침표가
토큰화 이전에 제거되지 않는 작은 버그를 찾았습니다. 이를 바로잡고 나서야
L-DWA 의 Oracle 대비 위치가 제대로 드러났는데, 숫자가 오르내린 것보다 "약간
낫다" 수준의 결과가 "이산 상한을 넘는다" 로 해석이 달라졌다는 점이 더
인상적이었습니다.

다른 하나는 **기대와 다르게 나온 결과를 어떻게 다룰지** 에 관한 것입니다. 교차
도메인 실험은 처음에는 간단히 "전이가 되는지" 만 확인할 생각이었고, L-DWA 가
무난히 넘어 줄 것으로 기대하였습니다. 막상 결과가 30~35% 저하로 나오자 한
문단으로 "한계" 라고 쓰고 넘어가고 싶은 유혹이 있었습니다. 그러나 원인을 언어 ·
도메인 · 아키텍처로 나누어 각각 별도의 후속 실험으로 확인해 보는 편이 오히려 더
명확한 결론을 내려주었습니다.
"""
    )

    st.info(
        "💡 **관련 논문 장** — Ch.Ⅶ 결론 전체. "
        "모든 코드 · 데이터 · 체크포인트는 [GitHub 저장소]"
        "(https://github.com/sdw1621/triple-rag-phd) 에 공개되어 있습니다."
    )


def tab_overview(agg: pd.DataFrame) -> None:
    st.header("📊 결과 요약 — 다섯 정책의 전체 성능 비교")

    st.markdown(
        "이 탭은 **5,000 개의 질의에 대해 다섯 가지 정책을 돌린 뒤 얻은 평균 점수** 를 "
        "한 화면에 모아 보여줍니다. 맨 위의 세 metric 카드로 핵심 결과를 먼저 확인하고, "
        "아래 표에서 정책별 · 지표별 수치를 비교하시면 됩니다. "
        "Simple / Multi-hop / Conditional 유형별 막대 그래프는 **어떤 유형에서 L-DWA 가 "
        "가장 큰 개선을 보이는지** 를 시각적으로 드러냅니다."
    )
    st.caption("※ 데이터: 5,000 QA × 5 정책 · list-prompt · post-CORRIGENDUM 평가기 (수정 버그 반영)")

    overall = agg[agg["scope"] == "overall"].set_index("policy")
    if overall.empty:
        st.warning(
            "⚠️ 집계 결과 파일을 아직 찾지 못했습니다. 저장소 최신 상태로 앱을 재배포하면 "
            "해결됩니다. (`results/rerun_*_list.json` 5개 파일 필요)"
        )
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

    # ------------------------------------------------------------------
    # 표 읽는 법 — 정책 (행) · 지표 (열) 친절 해설
    # ------------------------------------------------------------------
    st.subheader("📖 이 표를 읽는 법")
    st.markdown(
        "아래 표를 보면 **5개의 행** 과 **여러 개의 열** 이 있습니다. "
        "행은 **서로 다른 다섯 가지 정책** 이고, 열은 **그 정책을 평가한 여러 가지 지표** 입니다. "
        "표 자체를 보여드리기 전에, 각 이름이 무슨 뜻인지부터 하나씩 풀어 두겠습니다."
    )

    # --- 행(정책) 해설 ---
    st.markdown("##### 🎭 다섯 가지 정책 (표의 행)")
    st.markdown(
        "이 논문의 표에 등장하는 \"정책\" 은 **\"α · β · γ 가중치를 어떻게 정할 것인가\" 를 "
        "담당하는 다섯 가지 방식** 입니다. 성격이 다른 다섯 명의 \"심판\" 이라고 생각해도 됩니다."
    )
    policy_cards = [
        ("🧾 R-DWA",
         "**선행 JKSCI 2025 논문의 방법 — 이 연구의 비교 기준선.**  \n"
         "미리 정해 둔 매뉴얼을 보고 가중치를 고르는 사서에 비유할 수 있습니다. "
         "질의를 세 유형 중 하나로 분류한 뒤, 유형별 기본 가중치 표에서 값을 꺼내 씁니다. "
         "학습이 필요 없어 즉시 배포 가능하지만, 유형 내 미세 차이는 반영하지 못합니다."),
        ("👁️ Oracle",
         "**이상적 상한선을 보여 주는 가상 정책 — 실제 배포 불가.**  \n"
         "정답을 미리 알고 있는 전지전능한 심판에 비유할 수 있습니다. "
         "각 질의마다 66개 가중치 조합을 모두 시험해 보고 가장 좋은 점수를 주는 조합을 "
         "거꾸로 선택합니다. 현실에서는 정답을 모르기 때문에 쓸 수 없지만, "
         "\"이 상태 공간에서 이론적으로 얼마나 잘할 수 있는가\" 의 **천장** 을 알려줍니다."),
        ("🤖 L-DWA (seed 42 / 123 / 999)",
         "**본 논문이 제안하는 학습 기반 정책 — 핵심 기여.**  \n"
         "수많은 질의를 보며 스스로 좋은 가중치를 터득한 작은 AI 라고 생각하면 됩니다. "
         "5,636 개의 파라미터를 가진 신경망이 PPO 알고리즘으로 10,000 번 연습합니다. "
         "**seed 42 · 123 · 999** 세 개는 같은 방법을 서로 다른 난수 시작값으로 세 번 학습한 "
         "결과입니다. 세 결과가 비슷하게 나오면 \"우연히 잘된 게 아니라 재현 가능한 결과\" 라는 뜻이 됩니다."),
    ]
    for title, desc in policy_cards:
        with st.container(border=True):
            st.markdown(f"**{title}**  \n{desc}")

    # --- 열(지표) 해설 ---
    st.markdown("##### 📏 아홉 개의 열 (각 지표가 무엇을 재는가)")
    st.markdown(
        "표의 열은 **같은 정책의 답변을 여러 각도에서 채점한 점수들** 입니다. "
        "하나의 점수만 보면 오해하기 쉬워서, 본 논문은 일부러 여러 지표를 함께 보여 줍니다. "
        "각 열을 클릭해서 펼쳐 보세요."
    )

    with st.expander("**policy** — 어떤 정책인지 (각 행의 이름표)"):
        st.markdown(
            "그 행의 점수가 **어느 정책의 결과인지** 를 알려 주는 이름표입니다. "
            "위의 다섯 정책 중 하나가 들어갑니다."
        )
    with st.expander("**scope** — 점수를 낸 범위"):
        st.markdown(
            "그 점수를 **전체 질의 기준** 으로 낸 것인지, **특정 유형만** 따로 본 것인지를 "
            "알려 줍니다.  \n\n"
            "- `overall` — 5,000 개 질의 전체 평균\n"
            "- `simple` — 단순 유형 질의 2,000 개 평균\n"
            "- `multi_hop` — 다단계 관계 질의 1,750 개 평균\n"
            "- `conditional` — 조건부 질의 1,250 개 평균\n\n"
            "이 표에서는 `overall` 행만 보여 주고 있습니다. 유형별 비교는 아래 **Per-type F1_strict** 그래프를 참고하세요."
        )
    with st.expander("**n** — 몇 개 질의로 평가했는가"):
        st.markdown(
            "그 행의 점수가 **몇 개의 질의** 를 기반으로 계산되었는지를 나타냅니다. "
            "여기서는 모두 `5000` 으로, 한국어 합성 대학 벤치마크 전체를 사용한 결과입니다. "
            "점수 표준편차의 신뢰도를 가늠할 때 함께 보는 값이기도 합니다."
        )
    with st.expander("**F1_strict** — 엄격 채점 F1 (가장 주된 지표)"):
        st.markdown(
            "**0 부터 1 사이의 점수, 높을수록 좋음.**  \n\n"
            "답변과 정답을 \"단어 단위로 꼼꼼히\" 비교합니다. 조사 (은/는/이/가) 와 "
            "구두점을 제거한 뒤, 두 쪽의 단어들이 얼마나 겹치는지를 재는 F1 점수입니다.  \n\n"
            "🎒 **비유**  \n"
            "\"정답: 홍성민, 황성민, 전성민\" 에 대해,  \n"
            "- 답변이 \"홍성민, 황성민, 전성민\" 이면 → **F1_strict = 1.00** (완벽)\n"
            "- 답변이 \"홍성민 교수, 황성민 교수가 담당합니다\" 이면 → 전성민이 빠졌고 \"교수\", \"담당합니다\" 같은 여분 단어가 들어가서 **F1 이 내려감**\n"
            "- 답변이 \"박민수\" 이면 → 한 명도 안 맞음, **F1_strict = 0**  \n\n"
            "가장 엄격한 채점 방식이라 점수가 다른 지표들보다 낮게 나옵니다. "
            "본 논문은 이 값을 **주된 비교 기준** 으로 삼습니다."
        )
    with st.expander("**F1_substring** — 부분 일치 채점 F1"):
        st.markdown(
            "**0 부터 1 사이의 점수, 높을수록 좋음.**  \n\n"
            "정답을 쉼표로 나눈 각 \"항목\" 이 답변 문자열 어딘가에 **부분문자열로 등장하기만 하면** "
            "맞은 것으로 칩니다. 더 관대한 채점 방식입니다.  \n\n"
            "🎒 **비유**  \n"
            "\"정답: 홍성민, 황성민, 전성민\" 에 대해,  \n"
            "- \"홍성민 교수, 황성민 교수, 전성민 교수가 담당합니다\" → 세 이름이 모두 답변 안에 들어있음 → **F1_substring = 1.00**  \n\n"
            "선행 JKSCI 2025 논문의 F1 0.86 은 이 관대한 계산에 가까웠던 것으로 추정됩니다. "
            "본 논문은 F1_strict (엄격) 와 F1_substring (관대) 을 **함께** 보고해서 평가 기준의 영향을 드러냅니다."
        )
    with st.expander("**F1_char** — 글자 단위 채점 F1"):
        st.markdown(
            "**0 부터 1 사이의 점수, 높을수록 좋음.**  \n\n"
            "답변과 정답을 **연속된 3글자 조각** 으로 잘게 잘라서 비교합니다. 조사나 띄어쓰기, "
            "어순에 영향을 덜 받는 \"형식에 강건한\" 지표입니다.  \n\n"
            "🎒 **비유**  \n"
            "\"홍성민\" 과 \"홍성민은\" 은 토큰으로 보면 다르지만, 글자 단위로 보면 "
            "공통된 3글자 조각 `홍성민` 이 있어 매칭됩니다. "
            "세 F1 의 중간 관점으로 병기하여, 앞 두 지표의 격차가 어디에서 오는지 살피는 데 씁니다."
        )
    with st.expander("**EM** (Exact Match) — 완전 일치"):
        st.markdown(
            "**0 또는 1 에 가까운 점수 (평균값은 0~1 사이).**  \n\n"
            "답변이 정답과 **토큰 단위로 완전히 똑같을 때만** 1, 그렇지 않으면 0 입니다. 매우 엄격한 지표.  \n\n"
            "🎒 **비유**  \n"
            "문장형 답변 \"홍성민 교수, 황성민 교수가 담당합니다\" 는 정답 \"홍성민, 황성민\" 과 "
            "한 글자도 다르지 않아야 1점. 여분 단어가 조금이라도 있으면 0점.  \n\n"
            "그래서 LLM 이 자연어 문장으로 답하던 원래 방식으로는 EM 이 구조적으로 거의 항상 0 이었습니다. "
            "본 논문은 답변 형식을 list 형태로 유도하는 **list prompt** 를 도입하여 EM 이 약 0.39까지 회복됨을 보였습니다."
        )
    with st.expander("**Faithfulness** — 충실도 (환각 탐지)"):
        st.markdown(
            "**0 부터 1 사이의 점수, 높을수록 좋음.**  \n\n"
            "답변의 각 주장이 **검색된 문서 안에 실제로 등장하는지** 를 확인합니다. "
            "모델이 없는 사실을 지어내는 **환각 (hallucination)** 을 잡아내는 지표입니다.  \n\n"
            "🎒 **비유**  \n"
            "참고 자료: \"홍성민 교수가 강의를 담당한다\"  \n"
            "- 답변 \"홍성민, 황성민\" 에서 **황성민 은 자료에 없음** → 환각. Faithfulness 가 떨어짐.  \n\n"
            "F1 이 \"정답과 얼마나 같은가\" 를 본다면, Faithfulness 는 \"지어냄이 없는가\" 를 봅니다. 둘은 다른 축입니다."
        )
    with st.expander("**Latency** — 응답 시간 (초)"):
        st.markdown(
            "**초 단위, 낮을수록 빠름.**  \n\n"
            "질의 하나를 받고 답하기까지 걸린 시간. 본 연구는 retrieval + LLM generation 까지의 "
            "전체 시간을 재며, 대체로 **0.7 초 ~ 0.9 초** 범위에 들어옵니다. 사용자 체감으로는 \"즉시\" 에 가까운 수준.  \n\n"
            "보상 함수에는 latency 가 5초를 넘길 때만 페널티가 붙도록 설계되어 있어, "
            "현 수치 (1초 미만) 에서는 reward 에 직접 영향을 주지 않습니다."
        )

    st.markdown("##### 📋 정책별 지표 표 (list prompt)")
    st.caption(
        "이제 위 해설을 참고하며 아래 표를 읽어 보세요. 눈에 띄는 숫자는 표 아래에서 해석해 드립니다."
    )
    st.dataframe(
        overall.round(4),
        use_container_width=True,
    )

    # --- 숫자 해석 ---
    st.markdown("##### 🧭 위 숫자가 말해 주는 것")
    st.markdown(
        """
표의 숫자를 그대로 읽으면 다음과 같은 그림이 그려집니다.

- **R-DWA (선행 연구 기준선)** — F1_strict **0.5293**. 5,000 개 질의 전체에서
  엄격 채점 기준으로 약 **절반 정도 맞춘다** 는 뜻입니다. 반면 F1_substring
  **0.4821**, F1_char **0.4686** 으로 비슷한 수준을 보여, 관대하게 봐도 극적으로
  점수가 오르지는 않습니다.

- **Oracle (이론적 상한선)** — F1_strict **0.5540**. 정답을 미리 알고 66개 가중치
  조합 중 가장 좋은 것을 골랐을 때의 천장입니다. R-DWA 대비 +0.025 정도 위에
  있어, 이 상태 공간에서 규칙 기반만으로는 이 이상 올라가기 어렵다는 사실을
  알려 줍니다.

- **L-DWA (본 논문 제안, 3 seeds)** — F1_strict **0.5663 / 0.5539 / 0.5656**.
  세 시드 평균이 **0.562** 로, **Oracle 의 0.554 를 미세하게 넘습니다**. 이 관찰이
  바로 이 논문의 가장 중요한 주장입니다. "학습된 연속 가중치가 66-점 이산 격자
  바깥의 지점을 활용한다" 는 해석이 가능해집니다. 그리고 세 시드의 편차가
  1% 이내라는 점은 이 결과가 **재현 가능** 함을 뜻합니다.

- **EM** 은 모든 정책에서 **약 0.39** 로 서로 비슷합니다. 본 논문이 도입한
  list prompt 덕분에 구조적 0 이었던 EM 이 이 수준으로 회복된 것 자체가 중요한
  변화입니다.

- **Faithfulness** 는 R-DWA **0.544** → L-DWA **0.585** 로 약 7% 오릅니다. "정책을
  바꾼 결과 환각이 더 줄었다" 는 뜻.

- **Latency** 는 다섯 정책 모두 **0.7 초 ~ 0.8 초** 범위로 거의 차이가 없습니다.
  학습된 정책이라고 해서 응답이 느려지는 문제는 없다는 것을 확인해 줍니다.

📌 **한 줄 요약.** R-DWA 는 0.53, Oracle 상한은 0.55, L-DWA 는 Oracle 을 넘는
0.56 에 도달했으며 세 시드 모두 비슷하게 재현됩니다. 개선 폭은 크지 않지만
**상한을 넘는다는 점에서 질적으로 다른 의미** 를 가집니다.
"""
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
    st.header("🔍 쿼리 비교 — 동일 질의에 대한 정책별 답")

    st.markdown(
        "이 탭은 **같은 질의에 다섯 정책이 각각 무엇을 답했는지** 를 나란히 놓고 보여줍니다. "
        "먼저 아래 **관심 영역** 에서 빠른 필터를 고른 다음, 표에서 흥미로운 질의를 클릭하고 "
        "🔬 **자세히 보기** 에서 답변·가중치·점수를 비교하세요."
    )
    st.caption(
        "※ 각 정책별로 50 개씩 저장된 샘플을 사용합니다. 전체 5,000 QA 평균은 "
        "📊 결과 요약 탭을 참고하세요. 표의 각 열 이름의 의미는 📊 결과 요약 탭의 열 해설 참조."
    )

    if samples_df.empty:
        st.warning(
            "⚠️ 쿼리 샘플 파일 (`results/rerun_*_list.json`) 을 불러오지 못했습니다. "
            "저장소 최신 상태로 앱을 재배포하면 해결됩니다."
        )
        return

    # ------------------------------------------------------------------
    # 관심 영역 프리셋 — 사용자가 "무엇을 보고 싶은지" 빠르게 고르도록
    # ------------------------------------------------------------------
    st.markdown("##### 🔎 관심 영역 빠르게 고르기")
    preset = st.radio(
        "어떤 질의를 먼저 볼까요?",
        options=[
            "🎯 L-DWA 가 R-DWA 를 가장 많이 이긴 질의",
            "🔬 Oracle 도 어려워한 질의 (남은 개선 여지)",
            "💯 모든 정책이 정답 (단순 질의의 전형)",
            "🐛 모든 정책이 실패 (retrieval 한계)",
            "📂 전체 보기 (직접 필터링)",
        ],
        horizontal=False,
        key="explorer_preset",
        help="프리셋 하나를 고르면 아래 표가 그에 맞게 정렬·필터링됩니다.",
    )

    col_filter1, col_filter2 = st.columns(2)
    with col_filter1:
        type_filter = st.multiselect(
            "유형으로 걸러내기",
            sorted(samples_df["type"].unique()),
            default=sorted(samples_df["type"].unique()),
            help="simple / multi_hop / conditional 중 관심 있는 유형만 선택.",
        )
    with col_filter2:
        metric_filter = st.selectbox(
            "어떤 기준으로 정렬?",
            options=["qid", "f1_strict", "f1_substring", "faithfulness", "latency"],
            index=0,
            format_func=lambda x: {
                "qid": "질의 번호 (qid)",
                "f1_strict": "F1_strict (엄격 정확도)",
                "f1_substring": "F1_substring (부분 일치)",
                "faithfulness": "Faithfulness (환각 없음)",
                "latency": "Latency (응답 속도)",
            }.get(x, x),
        )
    order_desc = st.checkbox(
        "내림차순 정렬 (높은 값부터)", value=True, key="explorer_order_desc"
    )

    filtered = samples_df[samples_df["type"].isin(type_filter)].copy()

    # Pivot to one row per qid with columns per policy (F1_strict values)
    pivot_metric = filtered.pivot_table(
        index=["qid", "type", "gold"],
        columns="policy",
        values="f1_strict",
        aggfunc="first",
    ).reset_index()

    # Apply preset filtering / sorting
    pol_cols = [c for c in pivot_metric.columns if c in POLICY_FILES]

    def _get_seed42(row):
        return row.get("L-DWA (seed 42)", np.nan)

    def _get_rdwa(row):
        return row.get("R-DWA", np.nan)

    def _get_oracle(row):
        return row.get("Oracle", np.nan)

    if preset == "🎯 L-DWA 가 R-DWA 를 가장 많이 이긴 질의":
        if "L-DWA (seed 42)" in pivot_metric.columns and "R-DWA" in pivot_metric.columns:
            pivot_metric["diff_ldwa_rdwa"] = pivot_metric["L-DWA (seed 42)"] - pivot_metric["R-DWA"]
            pivot_metric = pivot_metric.sort_values("diff_ldwa_rdwa", ascending=False)
    elif preset == "🔬 Oracle 도 어려워한 질의 (남은 개선 여지)":
        if "Oracle" in pivot_metric.columns:
            pivot_metric = pivot_metric.sort_values("Oracle", ascending=True)
    elif preset == "💯 모든 정책이 정답 (단순 질의의 전형)":
        if pol_cols:
            pivot_metric["min_f1"] = pivot_metric[pol_cols].min(axis=1)
            pivot_metric = pivot_metric[pivot_metric["min_f1"] >= 0.9]
            pivot_metric = pivot_metric.sort_values("min_f1", ascending=False)
    elif preset == "🐛 모든 정책이 실패 (retrieval 한계)":
        if pol_cols:
            pivot_metric["max_f1"] = pivot_metric[pol_cols].max(axis=1)
            pivot_metric = pivot_metric[pivot_metric["max_f1"] <= 0.1]
    else:
        # 📂 전체 보기 — user's own sort
        sort_col = metric_filter if metric_filter in filtered.columns else "qid"
        if sort_col in pivot_metric.columns:
            pivot_metric = pivot_metric.sort_values(sort_col, ascending=not order_desc)
        elif pol_cols:
            pivot_metric = pivot_metric.sort_values(pol_cols[0], ascending=not order_desc)

    # Friendly column rename for display
    display_df = pivot_metric.rename(columns={
        "qid": "번호",
        "type": "유형",
        "gold": "정답",
    }).drop(columns=[c for c in ["diff_ldwa_rdwa", "min_f1", "max_f1"] if c in pivot_metric.columns])

    st.markdown("##### 📋 매칭된 질의 상위 30개")
    st.caption(
        "숫자는 각 정책의 **F1_strict** (0~1, 높을수록 좋음). 행을 보고 흥미로운 질의의 **번호** 를 기억한 뒤 "
        "아래 🔬 **자세히 보기** 에서 해당 번호를 선택하세요."
    )
    st.dataframe(display_df.round(3).head(30), use_container_width=True, hide_index=True)

    st.divider()

    # ------------------------------------------------------------------
    # Deep dive
    # ------------------------------------------------------------------
    available_qids = (
        pivot_metric["qid"].tolist() if not pivot_metric.empty
        else sorted(filtered["qid"].unique(), key=lambda x: int(x))
    )

    if not available_qids:
        st.info("현재 필터로 매칭된 질의가 없습니다. 프리셋이나 유형 필터를 바꿔 보세요.")
        return

    st.markdown("##### 🔬 자세히 보기 — 질의 하나를 골라 정책별 답을 비교")
    picked = st.selectbox(
        "질의 번호 (qid) 선택",
        available_qids,
        help="위 표의 '번호' 열에서 본 번호를 여기에서 고르면, 그 질의에 대해 다섯 정책이 "
             "어떤 가중치를 썼고 어떤 답을 냈는지 아래에 펼쳐집니다.",
    )

    if picked:
        sub = filtered[filtered["qid"] == picked].set_index("policy")
        if sub.empty:
            st.warning("이 번호에 해당하는 샘플이 없습니다.")
            return

        st.markdown(f"**질의 정답 (gold)**: `{sub.iloc[0]['gold']}`")
        st.markdown(f"**유형**: {sub.iloc[0]['type']}")

        st.caption(
            "각 정책 카드의 F1 / EM / Faith 는 0~1 사이 점수 (높을수록 좋음). "
            "Weights (α, β, γ) 는 그 정책이 이 질의에 대해 선택한 가중치. 합이 1."
        )

        for policy in sub.index:
            row = sub.loc[policy]
            f1v = row["f1_strict"]
            # color-hint expander title
            score_icon = "🟢" if f1v >= 0.9 else ("🟡" if f1v >= 0.3 else "🔴")
            with st.expander(
                f"{score_icon} **{policy}** — F1={row['f1_strict']:.3f}, "
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
    st.header("🎛️ 가중치 시뮬레이터 — 직접 움직여 보기")

    st.markdown(
        "이 탭은 여러분이 **직접 α·β·γ 슬라이더를 움직여** 보면서, "
        "특정 질의에 대해 어느 가중치 조합이 가장 높은 보상을 주는지 체험할 수 있는 "
        "실험 공간입니다. 왼쪽에서 슬라이더로 가중치를 조정하면, 오른쪽 삼각형 위에 "
        "66개 가중치 조합 각각의 보상이 색으로 표시됩니다. Oracle (argmax) 은 "
        "이 중 가장 좋은 점을 자동으로 찾아 알려 주는 가상의 상한선입니다."
    )
    st.caption(
        "※ 모든 응답·점수는 사전에 계산된 **330,000 개 엔트리의 오프라인 캐시** 에서 "
        "즉시 조회됩니다. 실시간 LLM 호출은 발생하지 않습니다."
    )

    conn = get_cache_conn()
    if conn is None:
        st.warning(
            "⚠️ 이 탭은 16.8 MB 의 보상 캐시 파일 (`cache/university.sqlite`) 이 있어야 "
            "동작합니다. 지금은 파일이 감지되지 않아 비활성화되어 있습니다. "
            "배포 환경이라면 저장소의 최신 `main` 브랜치로 앱을 재배포하면 해결됩니다."
        )
        return

    if samples_df.empty:
        st.warning(
            "⚠️ 쿼리 샘플 데이터 (`results/rerun_*_list.json`) 를 불러오지 못했습니다. "
            "저장소 최신 상태로 앱을 재배포하면 해결됩니다."
        )
        return

    qid_options = sorted(samples_df["qid"].unique(), key=lambda x: int(x))
    picked = st.selectbox(
        "쿼리 선택",
        qid_options,
        key="sim_qid",
        help="저장된 50개 샘플 중 하나를 고르세요. 각 qid 에 대해 66개 가중치 조합 전체의 "
             "보상이 오른쪽 삼각형에 색으로 표시됩니다.",
    )

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

    # ---- preset quick-pick buttons ----
    st.markdown("##### 🎯 프리셋으로 빠르게 가중치 고르기")
    st.caption(
        "버튼을 누르면 아래 슬라이더가 해당 조합으로 바뀝니다. 직접 조정하고 싶으면 슬라이더를 움직이시면 됩니다."
    )

    # Initialize session state for sliders
    if "sim_alpha" not in st.session_state:
        st.session_state["sim_alpha"] = 0.3
    if "sim_beta" not in st.session_state:
        st.session_state["sim_beta"] = 0.3

    # Oracle for this qid
    oracle_best = df.iloc[0]
    o_a, o_b, o_g = float(oracle_best["alpha"]), float(oracle_best["beta"]), float(oracle_best["gamma"])

    p_cols = st.columns(5)
    if p_cols[0].button("Vector 중심\n(α=0.6, β=0.2, γ=0.2)", key="p_vec"):
        st.session_state["sim_alpha"] = 0.6
        st.session_state["sim_beta"] = 0.2
    if p_cols[1].button("Graph 중심\n(α=0.2, β=0.6, γ=0.2)", key="p_gra"):
        st.session_state["sim_alpha"] = 0.2
        st.session_state["sim_beta"] = 0.6
    if p_cols[2].button("Ontology 중심\n(α=0.2, β=0.2, γ=0.6)", key="p_ont"):
        st.session_state["sim_alpha"] = 0.2
        st.session_state["sim_beta"] = 0.2
    if p_cols[3].button("균등\n(α=β=γ=0.33)", key="p_uni"):
        st.session_state["sim_alpha"] = 0.3
        st.session_state["sim_beta"] = 0.3
    if p_cols[4].button(
        f"Oracle 최적\n(α={o_a:.1f}, β={o_b:.1f}, γ={o_g:.1f})",
        key="p_ora",
        help="이 질의에서 66개 조합 중 가장 높은 보상을 주는 가중치",
    ):
        st.session_state["sim_alpha"] = o_a
        st.session_state["sim_beta"] = o_b

    st.divider()

    colL, colR = st.columns([2, 3])
    with colL:
        st.markdown("##### 🎚️ 슬라이더로 직접 조정")
        alpha = st.slider(
            "α (Vector) — 문장 임베딩 유사도 검색의 비중",
            0.0, 1.0, value=st.session_state["sim_alpha"], step=0.1, key="slider_a",
            help="높을수록 의미 기반 유사도 검색 결과가 더 많이 섞입니다.",
        )
        beta = st.slider(
            "β (Graph) — 지식 그래프 경로 탐색의 비중",
            0.0, 1.0 - alpha, value=min(st.session_state["sim_beta"], 1.0 - alpha), step=0.1, key="slider_b",
            help="높을수록 엔티티-관계 그래프에서 찾은 결과가 더 많이 섞입니다.",
        )
        gamma = round(1.0 - alpha - beta, 2)
        st.markdown(f"**γ (Ontology, 자동 계산)** = `{gamma:.2f}` — 제약 추론의 비중")

        ai, bi, gi = round(alpha * 10), round(beta * 10), round(gamma * 10)
        matched = df[
            (df["a"] == ai) & (df["b"] == bi) & (df["g"] == gi)
        ]

        st.markdown("##### 📐 이 가중치를 썼을 때의 점수")
        if not matched.empty:
            r = matched.iloc[0]
            mc1, mc2 = st.columns(2)
            with mc1:
                st.metric("F1 (이 가중치)", f"{r['f1']:.3f}",
                          help="0~1, 높을수록 좋음")
                st.metric("Faithfulness", f"{r['faith']:.3f}",
                          help="0~1, 높을수록 환각 없음")
            with mc2:
                st.metric("EM", f"{r['em']:.3f}",
                          help="완전 일치. 1 또는 0")
                st.metric("Reward (총 보상)", f"{r['reward']:.3f}",
                          help="0.5·F1 + 0.3·EM + 0.2·Faith − latency penalty. PPO 가 최대화하는 값")

            # Rank within 66 grid
            rank = (df["reward"].rank(ascending=False, method="min").loc[matched.index[0]])
            total = len(df)
            st.caption(
                f"💡 이 조합은 66개 격자 중 **{int(rank)}위** "
                f"(Oracle 최적 대비 {r['reward'] - oracle_best['reward']:+.3f})."
            )
        else:
            st.warning("해당 가중치가 격자에 없습니다. 슬라이더를 0.1 단위로 맞춰 주세요.")

        st.divider()
        st.markdown("##### 👁️ 참고 — Oracle 최적 가중치")
        st.markdown(
            f"이 질의에 대해 66개 조합 중 보상이 가장 높은 조합은  \n"
            f"**α = {o_a:.1f}, β = {o_b:.1f}, γ = {o_g:.1f}** 이고,  \n"
            f"이때 Reward = **{oracle_best['reward']:.3f}** 입니다."
        )

    with colR:
        st.markdown("##### 🔺 66-그리드 전체의 보상 지도")
        st.caption(
            "Δ³ 삼각형 위의 각 점은 α·β·γ 한 조합. 색이 진할수록 (밝은 노랑) 보상이 높음. "
            "점에 마우스를 올리면 해당 가중치와 보상이 표시됩니다."
        )
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
    st.header("📐 Stage-wise Baseline — 선행 논문 F1 0.86 의 재현 추적")

    st.markdown(
        "이 탭은 선행 JKSCI 2025 논문이 보고한 **R-DWA F1 0.86** 이 현 저장소의 "
        "코드·데이터로는 처음에 재현되지 않았던 문제를, 평가 코드의 세 가지 결함을 "
        "**한 단계씩 고쳐 가며** 추적한 기록입니다. S0 (원 보고치 · 재현 불가) 에서 "
        "S1 (구두점 처리 버그 수정) · S2 (gold 형식에 맞춘 프롬프트) · S3 (Faithfulness "
        "2-branch 도입) 를 거치면서 F1 이 어떻게 움직였는지 볼 수 있습니다. "
        "발견된 결함들은 선행 저장소에 **CORRIGENDUM** 으로 공개 정정하였습니다."
    )
    st.caption(
        "※ 본 논문의 L-DWA 개선 (+6.2%) 은 이 stage-wise 복구가 끝난 S3 위에서 측정된 "
        "순수 학습 기여분입니다. 즉 **평가 인프라 정비와 학습 개선을 분리** 하여 보고합니다."
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
    st.header("🌐 Cross-domain — 다른 데이터셋에서도 통하는가?")

    st.markdown(
        "이 탭은 **한국어 합성 대학 데이터로 학습한 L-DWA 를 영어 벤치마크에 그대로 "
        "적용하면 어떤 일이 일어나는지** 보여 줍니다. 세 벤치마크 (HotpotQA · "
        "MuSiQue · PubMedQA) 에서 naive transfer 는 Vector-only 기준선보다 약 30~35% "
        "떨어진 결과를 냈습니다. 본 논문은 이 실패 원인을 그대로 한계로 치부하지 않고, "
        "**언어 장벽 · 도메인 어휘 · Graph/Ontology 부재** 의 세 축으로 분해한 뒤 "
        "각각을 독립 실험으로 확인하였습니다. 영어 intent 패턴 추가 (+38~49% 회복), "
        "영어 합성 벤치마크 구축 (R-DWA 0.663 으로 한국어보다 높음), 영어 코퍼스에서의 "
        "PPO 재학습 결과를 아래 표와 그래프에서 함께 확인할 수 있습니다."
    )
    st.caption(
        "※ 본 논문 Ch.Ⅵ §7 과 일치. 교차 도메인 실패의 해석은 "
        "\"방법론의 한계\" 가 아니라 \"학습 조건의 정의\" 에 관한 실증입니다."
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
    st.header("📈 PPO 학습 — 세 개 seed 의 수렴 곡선")

    st.markdown(
        "이 탭은 PPO 알고리즘이 **세 개의 서로 다른 random seed** (42, 123, 999) "
        "에서 어떻게 학습되어 가는지 보여 줍니다. 학습이 재현 가능하다면 세 곡선이 "
        "**비슷한 지점으로 수렴** 해야 하며, 본 논문의 경우 세 seed 모두 "
        "mean_reward ≈ 0.215 ± 0.002 에서 멈추는 것을 확인할 수 있습니다. "
        "아래 슬라이더로 이동평균 창을 조정해 노이즈를 걷어 내면 전반적인 추세가 "
        "더 잘 보입니다."
    )
    st.caption(
        "※ 세 seed 의 표준편차가 1% 미만이라는 사실이 본 논문의 주장 "
        "(학습 자체의 재현성) 을 뒷받침합니다."
    )

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
    # Landing header — always visible above tabs
    render_landing_header()

    # Sidebar glossary — always accessible
    render_sidebar_glossary()

    agg = load_aggregates()
    samples = load_samples_df()

    # Tabs follow the thesis chapter arc: 한눈에 보기 → 문제 → 방법 → 결과 → 검증 → 결론.
    # Each tab narrates a section of the paper and embeds the relevant data / figures.
    tabs = st.tabs([
        "📖 한눈에 보기",
        "🧩 문제",
        "📜 R-DWA (기존)",
        "🧠 L-DWA (제안)",
        "📊 결과 요약",
        "🔍 쿼리 비교",
        "🎛️ 가중치 탐색기",
        "📈 PPO 학습",
        "📐 재현성 복구",
        "🌐 교차 도메인",
        "🔭 결론",
    ])

    with tabs[0]:
        tab_about()
    with tabs[1]:
        tab_problem()
    with tabs[2]:
        tab_rdwa()
    with tabs[3]:
        tab_ldwa()
    with tabs[4]:
        tab_overview(agg)
    with tabs[5]:
        tab_explorer(samples)
    with tabs[6]:
        tab_simulator(samples)
    with tabs[7]:
        tab_training()
    with tabs[8]:
        tab_stagewise()
    with tabs[9]:
        tab_cross_domain()
    with tabs[10]:
        tab_conclusion()

    st.divider()
    st.caption(
        "박사학위 논문 「PPO 기반 L-DWA 를 통한 Triple-Hybrid RAG 성능 최적화」 · "
        "Shin Dong-wook (신동욱), Hoseo University · 2026 · "
        "read-only dashboard (no LLM calls)  \n"
        "저장소: https://github.com/sdw1621/triple-rag-phd · "
        "선행 CORRIGENDUM: "
        "https://github.com/sdw1621/hybrid-rag-comparsion/blob/main/CORRIGENDUM.md"
    )


if __name__ == "__main__":
    main()
