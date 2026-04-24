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

    st.subheader("탐색 순서 추천")
    st.markdown(
        """
처음 방문하셨다면 아래 순서로 탭을 둘러보시면 이해가 빠릅니다.

1. **📊 결과 요약** — 다섯 정책의 전체 F1/EM/Faithfulness 를 한 표로 비교
2. **🔍 쿼리 비교** — 동일 질의에 대해 정책별 답이 어떻게 달라지는지 샘플로 확인
3. **🎛️ 가중치 시뮬레이터** — 여러분이 직접 슬라이더로 α·β·γ 를 움직여 보면서
   330K 캐시에서 즉시 보상 조회
4. **📈 PPO 학습** — 세 개 seed 의 학습 곡선 overlay, 재현성 확인
5. **📐 Stage-wise Baseline** — 선행 JKSCI 논문의 F1 0.86 이 현 저장소에서 왜
   재현되지 않았고, 어떤 세 단계로 복구되는지 (**CORRIGENDUM** 배경)
6. **🌐 Cross-domain** — 이 방법론이 다른 데이터셋 (HotpotQA 등) 에서도 통하는지

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
    st.header("🔍 쿼리 비교 — 동일 질의에 대한 정책별 답")

    st.markdown(
        "이 탭은 **같은 질의에 다섯 정책이 각각 무엇을 답했는지** 를 나란히 놓고 "
        "보여줍니다. gold 정답, 정책별 답변, 해당 답의 F1 · EM · Faithfulness 점수, "
        "그리고 그 정책이 선택한 α·β·γ 가중치가 모두 표시됩니다. 표를 정렬하거나 "
        "유형으로 걸러서 흥미로운 질의를 고른 뒤, 아래 **🔬 Deep dive** 영역에서 "
        "그 질의의 답을 펼쳐 볼 수 있습니다."
    )
    st.caption(
        "※ 각 정책별로 50 개씩 저장된 샘플을 사용합니다. 전체 5,000 QA 에 대한 점수는 "
        "📊 결과 요약 탭을 참고하세요."
    )

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

    tabs = st.tabs([
        "📖 시작하기",
        "📊 결과 요약",
        "🔍 쿼리 비교",
        "🎛️ 가중치 시뮬레이터",
        "📈 PPO 학습",
        "📐 Stage-wise Baseline",
        "🌐 Cross-domain",
    ])

    with tabs[0]:
        tab_about()
    with tabs[1]:
        tab_overview(agg)
    with tabs[2]:
        tab_explorer(samples)
    with tabs[3]:
        tab_simulator(samples)
    with tabs[4]:
        tab_training()
    with tabs[5]:
        tab_stagewise()
    with tabs[6]:
        tab_cross_domain()

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
