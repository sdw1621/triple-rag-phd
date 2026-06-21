import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import qrcode
import base64
from io import BytesIO
from pathlib import Path

_HERE = Path(__file__).parent

st.set_page_config(
    page_title="Triple-Hybrid RAG 시뮬레이션",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="auto",
)


st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;500;700;900&display=swap');
html, body, [class*="css"] { font-family: 'Noto Sans KR', sans-serif; }

.hero {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    padding: 2.5rem 2rem; border-radius: 16px; margin-bottom: 1.5rem; color: white;
}
.hero h1 { font-size: 1.85rem; font-weight: 900; margin: 0 0 0.5rem 0; letter-spacing: -0.5px; }
.hero p  { font-size: 1.05rem; color: #b0d4ec; margin: 0; line-height: 1.6; }

.metric-card {
    background: linear-gradient(145deg, #1a2a3e, #0f1e2e);
    border: 1px solid #2d4a6b; border-radius: 12px;
    padding: 1rem 1.2rem; text-align: center;
}
.metric-card .val   { font-size: 1.9rem; font-weight: 900; color: #4fc3f7; }
.metric-card .lbl   { font-size: 0.9rem; color: #78909c; margin-top: 4px; }
.metric-card .delta { font-size: 0.95rem; color: #66bb6a; font-weight: 700; margin-top: 2px; }

.step-box {
    background: #0d1b2a; border-left: 4px solid #4fc3f7;
    border-radius: 8px; padding: 0.9rem 1.1rem; margin-bottom: 0.8rem;
}
.step-box .step-title { color: #4fc3f7; font-weight: 700; font-size: 1rem; margin-bottom: 5px; }
.step-box .step-body  { color: #cfd8dc; font-size: 1rem; line-height: 1.7; }

.reason-box {
    background: #001020; border-left: 4px solid #ffd54f;
    border-radius: 8px; padding: 0.75rem 1rem; margin-top: 8px; font-size: 0.95rem;
}
.reason-box .r-label { color: #ffd54f; font-weight: 700; margin-bottom: 3px; }
.reason-box .r-body  { color: #ffe082; line-height: 1.6; }
.reason-box .r-kw    { background:#1a1000; border:1px solid #ffd54f; border-radius:4px;
                        padding:1px 6px; font-family:monospace; font-size:0.9rem; color:#ffd54f; }

.rdwa-box {
    background: #1a1200; border-left: 4px solid #ffb74d;
    border-radius: 8px; padding: 0.9rem 1.1rem; margin-bottom: 0.5rem;
}
.rdwa-box .rdwa-title { color: #ffb74d; font-weight: 700; font-size: 1rem; margin-bottom: 5px; }
.rdwa-box .rdwa-body  { color: #ffe0b2; font-size: 0.95rem; line-height: 1.7; font-family: monospace; }

.adwa-box {
    background: #001a09; border-left: 4px solid #66bb6a;
    border-radius: 8px; padding: 0.9rem 1.1rem; margin-bottom: 0.8rem;
}
.adwa-box .adwa-title { color: #66bb6a; font-weight: 700; font-size: 1rem; margin-bottom: 5px; }
.adwa-box .adwa-body  { color: #c8e6c9; font-size: 0.95rem; line-height: 1.7; }

.tag { display:inline-block; padding:2px 10px; border-radius:999px; font-size:0.88rem; font-weight:700; margin-right:4px; }
.tag-simple      { background:#1565c0; color:white; }
.tag-multihop    { background:#4a148c; color:white; }
.tag-conditional { background:#b71c1c; color:white; }

.weight-bar-wrap { display:flex; align-items:center; gap:8px; margin:5px 0; }
.weight-bar-wrap .bar-lbl { width:80px; font-size:0.92rem; color:#90a4ae; }
.weight-bar-bg   { flex:1; background:#1a2634; border-radius:4px; height:14px; }
.weight-bar-fill { height:14px; border-radius:4px; }

.info-box {
    background:#0a1929; border:1px solid #1565c0; border-radius:10px;
    padding:1rem 1.2rem; font-size:1rem; color:#b0bec5; line-height:1.7;
}

.stage-badge {
    display:inline-block; background:#1565c0; color:white;
    font-size:0.85rem; font-weight:700; padding:2px 8px; border-radius:999px; margin-right:6px;
}

/* ── 용어 카드 뉴스 ── */
.explain-box {
    background: #0a1a2b; border: 1px solid #1e3a55; border-radius: 10px;
    padding: 1.2rem 1.4rem; margin-top: 4px;
}
.explain-box .eb-title  { font-size: 1rem; font-weight: 700; color: #4fc3f7; margin-bottom: 6px; }
.explain-box .eb-simple { font-size: 1rem; color: #e0f2fe; line-height: 1.75; margin-top: 8px; }
.explain-box .eb-analogy { font-size: 0.95rem; color: #80cbc4; margin-top:8px; padding:8px 10px;
                            background:#071520; border-radius:6px; }
.explain-box .eb-formula { font-family:monospace; font-size:0.92rem; color:#ce93d8; margin-top:8px;
                             padding:8px 10px; background:#100820; border-radius:6px; }
.tc-cat { font-size:0.8rem; font-weight:700; border-radius:999px; padding:2px 8px;
           display:inline-block; margin-bottom:4px; }
.cat-core  { background:#1565c0; color:white; }
.cat-rl    { background:#4a148c; color:white; }
.cat-eval  { background:#1b5e20; color:white; }
.cat-infra { background:#bf360c; color:white; }

/* ── 연구 배경 타임라인 ── */
.bg-section {
    background: linear-gradient(135deg, #0a1929 0%, #0d1b2a 100%);
    border: 1px solid #1e3a55; border-radius: 14px;
    padding: 1.4rem 1.6rem; margin-bottom: 1.2rem;
}
.bg-section .bg-title {
    font-size: 0.95rem; font-weight: 700; letter-spacing: 2px;
    color: #607d8b; text-transform: uppercase; margin-bottom: 1rem;
}
.timeline-col {
    background: #0f1e2e; border-radius: 10px;
    padding: 1rem 1.1rem; height: 100%;
    border-top: 3px solid #1565c0;
}
.timeline-col.rdwa  { border-top-color: #ffb74d; }
.timeline-col.limit { border-top-color: #ef5350; }
.timeline-col.adwa  { border-top-color: #66bb6a; }
.timeline-col .tc-head {
    font-size: 0.9rem; font-weight: 700; letter-spacing: 1px;
    text-transform: uppercase; margin-bottom: 6px;
}
.timeline-col .tc-title {
    font-size: 1rem; font-weight: 900; margin-bottom: 8px; line-height: 1.3;
}
.timeline-col .tc-body {
    font-size: 0.95rem; color: #90a4ae; line-height: 1.7;
}
.limit-item {
    display: flex; align-items: flex-start; gap: 8px;
    margin-bottom: 7px; font-size: 0.95rem; color: #ef9a9a; line-height: 1.5;
}
.limit-item .li-num {
    background: #b71c1c; color: white; border-radius: 50%;
    width: 18px; height: 18px; display: flex; align-items: center;
    justify-content: center; font-size: 0.78rem; font-weight: 700;
    flex-shrink: 0; margin-top: 1px;
}
.result-chip {
    display: inline-block; background: #1b3a1e; border: 1px solid #66bb6a;
    color: #66bb6a; border-radius: 6px; padding: 3px 9px;
    font-size: 0.9rem; font-weight: 700; margin: 3px 3px 3px 0;
}
.arrow-col {
    display: flex; align-items: center; justify-content: center;
    font-size: 1.8rem; color: #4fc3f7; padding-top: 2.5rem;
}

/* ══════════════════════════════════════════════
   모바일 반응형 (≤ 768px)
══════════════════════════════════════════════ */
@media (max-width: 768px) {
  /* ── Hero ── */
  .hero { padding: 1.4rem 1rem; }
  .hero h1 { font-size: 1.25rem; }
  .hero p  { font-size: 0.9rem; }

  /* ── Streamlit 컬럼 → 세로 스택 ── */
  [data-testid="column"] {
    width: 100% !important;
    flex: 1 1 100% !important;
    min-width: 100% !important;
  }
  .arrow-col { display: none; }

  /* ── Metric card ── */
  .metric-card { padding: 0.7rem 0.8rem; }
  .metric-card .val { font-size: 1.45rem; }

  /* ── Step / Reason / R-DWA / A-DWA box ── */
  .step-box  { padding: 0.7rem 0.85rem; }
  .step-box .step-body { font-size: 0.88rem; }
  .reason-box { font-size: 0.85rem; padding: 0.6rem 0.8rem; }
  .rdwa-box  { padding: 0.7rem 0.85rem; }
  .rdwa-box .rdwa-body { font-size: 0.82rem; }
  .adwa-box  { padding: 0.7rem 0.85rem; }
  .adwa-box .adwa-body { font-size: 0.88rem; }

  /* ── Weight bar ── */
  .weight-bar-wrap .bar-lbl { width: 60px; font-size: 0.82rem; }

  /* ── Info box ── */
  .info-box { font-size: 0.88rem; padding: 0.8rem 0.9rem; }

  /* ── Explain box (glossary) ── */
  .explain-box { padding: 0.9rem 1rem; }
  .explain-box .eb-simple { font-size: 0.88rem; }
  .explain-box .eb-analogy { font-size: 0.83rem; }
  .explain-box .eb-formula { font-size: 0.8rem; }

  /* ── SVG / chart 컨테이너 가로 스크롤 ── */
  [data-testid="stIframe"] { overflow-x: auto !important; }

  /* ── 비교 테이블 가로 스크롤 ── */
  .cmp-tbl { font-size: 0.76rem; }
  table { display: block; overflow-x: auto; white-space: nowrap; }

  /* ── 탭 레이블 작게 ── */
  .stTabs [data-baseweb="tab"] { font-size: 0.78rem; padding: 6px 8px; }

  /* ── 배경 섹션 ── */
  .bg-section { padding: 1rem 1rem; }
  .timeline-col { padding: 0.7rem 0.8rem; }
  .timeline-col .tc-body { font-size: 0.85rem; }
}

/* ══ 초소형 화면 (≤ 480px) ══ */
@media (max-width: 480px) {
  .hero h1 { font-size: 1.05rem; }
  .hero p  { font-size: 0.82rem; }
  .metric-card .val { font-size: 1.2rem; }
  .step-box .step-title { font-size: 0.88rem; }
  .tag { font-size: 0.75rem; padding: 2px 7px; }
}
</style>
""", unsafe_allow_html=True)

# ── 논문 데이터 상수 ──────────────────────────────────────────────────────────
QUERY_EXAMPLES = {
    "안기찬 교수가 담당하는 과목은?": "simple",
    "문남미 교수가 참여한 연구 프로젝트 목록은?": "multi-hop",
    "벤처대학원 심사위원 중 55세 이하 교수는?": "conditional",
    "박두순 교수의 소속 학과는?": "simple",
    "최유주 교수가 개설한 과목을 모두 찾아줘": "multi-hop",
    "오삼권 교수 지도 학생 중 박사과정 40세 이하는?": "conditional",
}

# 질의 유형 분류 이유 (논문 §III-B 기반)
QUERY_REASONS = {
    "안기찬 교수가 담당하는 과목은?": {
        "keywords": ["안기찬 교수"],
        "rule": "특정 Named Entity 단독 참조",
        "why": "「안기찬 교수」라는 특정 개체(Named Entity) 하나에 대한 담당 과목 속성 조회. 관계 탐색·제약 없이 Vector 유사도 검색(α↑)만으로 답변 가능 → Simple 분류",
    },
    "문남미 교수가 참여한 연구 프로젝트 목록은?": {
        "keywords": ["참여한", "목록"],
        "rule": "다단계 관계 키워드('참여') + 집합 반환 키워드('목록')",
        "why": "「참여한」= 교수→프로젝트 관계 탐색, 「목록」= 다수 개체 열거. 교수→프로젝트 Graph BFS(β↑) 필요 → Multi-hop 분류",
    },
    "벤처대학원 심사위원 중 55세 이하 교수는?": {
        "keywords": ["55세 이하"],
        "rule": "수치 제약 패턴 감지 (나이/이하/이상/이내)",
        "why": "「55세 이하」라는 수치 부등식 제약이 포함됨. 단순 문서 검색으로는 나이 필터링 불가 → OWL2 Ontology 추론(γ↑)이 필수이므로 Conditional 분류",
    },
    "박두순 교수의 소속 학과는?": {
        "keywords": ["박두순 교수"],
        "rule": "특정 Named Entity 단독 참조",
        "why": "「박두순 교수」개체의 소속(학과) 속성 단순 조회. 다단계 탐색·수치 제약 없음 → Simple 분류",
    },
    "최유주 교수가 개설한 과목을 모두 찾아줘": {
        "keywords": ["모두", "개설한"],
        "rule": "전체 열거 키워드('모두') + 관계 키워드('개설한')",
        "why": "「모두」= 전체 집합 반환, 「개설한」= 교수→과목 관계. 교수→과목 Graph BFS(β↑) 필요 → Multi-hop 분류",
    },
    "오삼권 교수 지도 학생 중 박사과정 40세 이하는?": {
        "keywords": ["40세 이하"],
        "rule": "수치 제약 패턴 감지 (나이/이하)",
        "why": "「40세 이하」라는 수치 제약 → Ontology의 age 속성 부등식 추론(γ↑) 필요. 지도→학생 관계도 포함되어 있으나 제약 조건이 지배적 → Conditional 분류",
    },
}

# R-DWA Stage 1 Base Weights (논문 Table IV-1)
RDWA_BASE = {
    "simple":      {"alpha": 0.60, "beta": 0.20, "gamma": 0.20},
    "multi-hop":   {"alpha": 0.20, "beta": 0.60, "gamma": 0.20},
    "conditional": {"alpha": 0.20, "beta": 0.20, "gamma": 0.60},
}
# R-DWA Stage 2 조정 강도 λ (논문 식 IV-1, 단일 λ = 0.3)
RDWA_LAMBDA = 0.3

# A-DWA PPO Actor 출력 (학습된 가중치, 3-seed 평균)
ADWA_WEIGHTS = {
    "simple":      {"alpha": 0.62, "beta": 0.23, "gamma": 0.15},
    "multi-hop":   {"alpha": 0.14, "beta": 0.61, "gamma": 0.25},
    "conditional": {"alpha": 0.12, "beta": 0.23, "gamma": 0.65},
}
# 밀도 신호
DENSITY_SIM = {
    "simple":      {"s_e": 0.82, "s_r": 0.12, "s_c": 0.06},
    "multi-hop":   {"s_e": 0.55, "s_r": 0.42, "s_c": 0.03},
    "conditional": {"s_e": 0.30, "s_r": 0.18, "s_c": 0.52},
}
TYPE_LABELS = {
    "simple":      ("Simple",      "tag-simple"),
    "multi-hop":   ("Multi-hop",   "tag-multihop"),
    "conditional": ("Conditional", "tag-conditional"),
}
VECTOR_RESULTS = {
    "simple":      ["안기찬 교수 프로필: 벤처대학원 교수(심사위원장), 전공 경영학·창업...",
                    "안기찬 교수 강의계획서: 창업경영론, 기술사업화전략 (2026-1학기)..."],
    "multi-hop":   ["문남미 교수 연구실 소개: AI·NLP 기반 지식 검색 연구...",
                    "문남미 교수 참여 프로젝트 목록 문서 (2022~2026)..."],
    "conditional": ["벤처대학원 교수 명단 및 프로필 문서...",
                    "심사위원 나이·직위 정보 포함 인사 데이터..."],
}
GRAPH_RESULTS = {
    "simple":      ["안기찬 --[담당]--> 창업경영론\n안기찬 --[담당]--> 기술사업화전략\n안기찬 --[소속]--> 벤처대학원"],
    "multi-hop":   ["문남미 --[참여]--> Triple-Hybrid RAG 프로젝트 (2024~2026)\n"
                    "문남미 --[참여]--> NLP 기반 지식그래프 구축 (2022~2024)\n"
                    "문남미 --[지도]--> 신동욱 --[수행]--> A-DWA 연구"],
    "conditional": ["벤처대학원 --[소속]--> 안기찬 (나이:61) [심사위원장]\n"
                    "벤처대학원 --[소속]--> 문남미 (나이:53) [지도교수]\n"
                    "벤처대학원 --[소속]--> 오삼권 (나이:50)\n"
                    "벤처대학원 --[소속]--> 박두순 (나이:57)\n"
                    "벤처대학원 --[소속]--> 최유주 (나이:47)"],
}
ONTOLOGY_RESULTS = {
    "simple":      ["[제약 없음 — 단순 Named Entity 속성 조회]"],
    "multi-hop":   ["Professor ⊑ Person, participatesIn(Project) 추론 경로 적용\n"
                    "supervisedBy(신동욱, 문남미) → 역방향 추론 지원"],
    "conditional": ["age ≤ 55 제약 적용 (OWL2 DataProperty: hasAge)\n"
                    "→ 문남미 (53) ✓  오삼권 (50) ✓  최유주 (47) ✓\n"
                    "→ 박두순 (57) ✗  안기찬 (61) ✗\n"
                    "※ 합성 데이터 기반 시뮬레이션"],
}

# 전체 성능 테이블 (논문 Table VI-4, Corrected Baseline, 5,000 QA, 3-seed avg.)
PERF_OVERALL = pd.DataFrame({
    "정책":         ["Vector-only", "R-DWA", "A-DWA", "Discrete Oracle"],
    "F1_strict":    [0.334, 0.529, 0.562, 0.554],
    "F1_substring": [0.334, 0.482, 0.507, 0.504],
    "F1_char":      [0.317, 0.469, 0.494, 0.487],
    "EM":           [0.250, 0.387, 0.388, 0.388],
    "Faithfulness": [0.470, 0.544, 0.580, 0.570],
    "Latency (s)":  [0.680, 0.711, 0.763, 0.749],
})
PERF_BY_TYPE = pd.DataFrame({
    "Query Type": ["Simple","Simple","Multi-hop","Multi-hop","Conditional","Conditional"],
    "Policy":     ["R-DWA","A-DWA","R-DWA","A-DWA","R-DWA","A-DWA"],
    "F1_strict":  [0.874, 0.906, 0.354, 0.365, 0.223, 0.304],
})

def ppo_curve(seed, n=200):
    rng = np.random.RandomState(seed)
    x = np.arange(n)
    base = 0.375 + 0.187 * (1 - np.exp(-x / 45))
    noise = rng.randn(n) * 0.008 * (1 - x / (n * 1.5))
    return np.clip(base + noise, 0.35, 0.60)

PPO_SEEDS = {42: ppo_curve(42), 123: ppo_curve(123), 999: ppo_curve(999)}

DARK_BG  = "#0d1b2a"
PAPER_BG = "#0a1929"

def rdwa_stage2(base, density, lam=RDWA_LAMBDA):
    """논문 식 IV-1: 단일 λ로 관계밀도 s_r→β, 제약밀도 s_c→γ 조정 후 정규화."""
    sr, sc = density["s_r"], density["s_c"]
    a = base["alpha"] * (1 - lam * (sr + sc) / 2)
    b = base["beta"]  + lam * sr * (1 - base["beta"])
    g = base["gamma"] + lam * sc * (1 - base["gamma"])
    a, b, g = max(0.01, a), max(0.01, b), max(0.01, g)
    s = a + b + g
    return {"alpha": round(a/s, 3), "beta": round(b/s, 3), "gamma": round(g/s, 3)}

def _t2c(a, b, c):
    """Ternary (a=top γ, b=left β, c=right α) → unit-triangle Cartesian"""
    return c + 0.5 * a, a * (3**0.5) / 2

def _c2t(x, y):
    """Unit-triangle Cartesian → normalised ternary"""
    a = max(0.01, y * 2 / (3**0.5))
    c = max(0.01, x - y / (3**0.5))
    b = max(0.01, 1.0 - a - c)
    s = a + b + c
    return a/s, b/s, c/s

def _label_pos(pts, scale=0.26):
    """Push each label outward from the cluster centroid."""
    xs, ys = zip(*[_t2c(*p) for p in pts])
    cx, cy = np.mean(xs), np.mean(ys)
    result = []
    for i, (a, b, c) in enumerate(pts):
        x, y = _t2c(a, b, c)
        dx, dy = x - cx, y - cy
        dist = (dx**2 + dy**2)**0.5
        if dist < 0.03:
            ang = [2.6, 0.5, -1.2, 1.8][i % 4]
            dx, dy = np.cos(ang) * scale, np.sin(ang) * scale
        else:
            dx, dy = dx/dist*scale, dy/dist*scale
        lx = float(np.clip(x + dx, 0.06, 0.93))
        ly = float(np.clip(y + dy, 0.05, 0.84))
        result.append(_c2t(lx, ly))
    return result

def _add_tern_labels(fig, ref_pts, labels, colors):
    """Add connector lines + outside labels to a ternary figure."""
    lbl_pts = _label_pos(ref_pts)
    for (ma, mb, mc), (la, lb, lc), lbl, col in zip(ref_pts, lbl_pts, labels, colors):
        fig.add_trace(go.Scatterternary(
            a=[ma, la], b=[mb, lb], c=[mc, lc],
            mode="lines",
            line=dict(color=col, width=1.4, dash="dot"),
            hoverinfo="skip", showlegend=False,
        ))
        fig.add_trace(go.Scatterternary(
            a=[la], b=[lb], c=[lc],
            mode="text", text=[lbl],
            textfont=dict(size=12, color=col),
            textposition="middle center",
            hoverinfo="skip", showlegend=False,
        ))

def hex_to_rgba(hex_color, alpha=0.13):
    h = hex_color.lstrip('#')
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"

def weight_bar_html(weights, label, color):
    bars = ""
    for src, key, col in [("α Vector","alpha","#4fc3f7"),("β Graph","beta","#ce93d8"),("γ Ontology","gamma","#ef9a9a")]:
        pct = int(weights[key] * 100)
        bars += (
            f'<div class="weight-bar-wrap">'
            f'<span class="bar-lbl">{src}</span>'
            f'<div class="weight-bar-bg"><div class="weight-bar-fill" style="width:{pct}%;background:{col};"></div></div>'
            f'<span style="font-size:0.92rem;color:#cfd8dc;width:42px;">{weights[key]:.3f}</span>'
            f'</div>'
        )
    return f'<div style="margin-bottom:6px;font-size:0.95rem;color:{color};font-weight:700;">{label}</div>' + bars

APP_URL = "https://triple-rag-phd2.streamlit.app/"

@st.cache_data
def _make_qr_b64(url: str) -> str:
    qr = qrcode.QRCode(version=2, box_size=6, border=3,
                       error_correction=qrcode.constants.ERROR_CORRECT_M)
    qr.add_data(url)
    qr.make(fit=True)
    img = qr.make_image(fill_color="#4fc3f7", back_color="#0d1117")
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

# ── 헤더 ───────────────────────────────────────────────────────────────────
_qr_b64 = _make_qr_b64(APP_URL)
st.markdown(f"""
<div class="hero" style="display:flex;align-items:center;justify-content:space-between;gap:1.2rem;">
  <div style="flex:1;min-width:0;">
    <h1>🧠 Triple-Hybrid RAG 시뮬레이션 대시보드</h1>
    <p>
      PPO 기반 적응형 동적 가중치 학습을 통한 Triple-Hybrid RAG 프레임워크 성능 최적화 연구<br>
      신동욱 · 호서대학교 벤처대학원 · 지도교수 문남미 · 2026
    </p>
  </div>
  <div style="flex-shrink:0;text-align:center;">
    <a href="{APP_URL}" target="_blank">
      <img src="data:image/png;base64,{_qr_b64}"
           style="width:110px;height:110px;border-radius:8px;display:block;"/>
    </a>
    <div style="font-size:0.68rem;color:#90caf9;margin-top:4px;">📱 모바일로 접속</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── 다이얼로그 정의 ────────────────────────────────────────────────────────
@st.dialog("📄 선행 논문 (JKSCI 2025) — 상세 보기", width="large")
def dialog_prev():
    st.markdown("""
    <div style="background:#1a1200;border-left:4px solid #ffb74d;border-radius:10px;padding:1.4rem 1.6rem;font-size:1.05rem;line-height:1.8;color:#ffe0b2;">
      <div style="font-size:1.3rem;font-weight:900;color:#ffcc80;margin-bottom:1rem;">
        Triple-Hybrid RAG + R-DWA 제안
      </div>
      저자의 선행 연구[1]에서 세 가지 지식 소스(Vector · Graph · Ontology)를 결합한
      <b style="color:#ffcc80;">Triple-Hybrid RAG 프레임워크</b>와
      2단계 규칙 기반 가중치 알고리즘 <b style="color:#ffcc80;">R-DWA</b>를 최초 제안.
      <hr style="border-color:#3a2800;margin:1rem 0;">
      <b style="color:#ffb74d;font-size:1.1rem;">R-DWA 작동 원리</b><br><br>
      <b>Stage 1.</b> 질의 유형(Simple / Multi-hop / Conditional) 룩업테이블로 기본 가중치(α, β, γ) 선택<br>
      <b>Stage 2.</b> 관계·제약 밀도(s_r, s_c)로 단일 λ=0.3 연속 보정 → 정규화 (합 = 1)<br>
      <br>
      <div style="background:#120d00;border-radius:8px;padding:0.9rem 1.1rem;font-family:monospace;font-size:0.97rem;color:#ffe082;">
        α' = α_base × (1 − λ·(s_r + s_c)/2)<br>
        β' = β_base + λ·s_r·(1 − β_base)<br>
        γ' = γ_base + λ·s_c·(1 − γ_base)<br>
        (α, β, γ) = normalize(α', β', γ'),  λ = 0.3
      </div>
      <hr style="border-color:#3a2800;margin:1rem 0;">
      <b style="color:#ffb74d;font-size:1.1rem;">선행 논문 보고 수치</b><br><br>
      F1 <b>0.86</b> &nbsp;·&nbsp; EM <b>0.78</b> &nbsp;·&nbsp; Faithfulness <b>0.89</b><br>
      <span style="font-size:0.9rem;color:#8d6e63;">
        ※ 평가 방법 이슈로 후속 재검증 필요<br>
        → 박사논문에서 Corrected Baseline 수립 (F1_strict 0.529)
      </span>
    </div>
    """, unsafe_allow_html=True)

@st.dialog("⚠️ R-DWA의 구조적 한계 — 상세 보기", width="large")
def dialog_limit():
    st.markdown("""
    <div style="background:#1a0a0a;border-left:4px solid #ef5350;border-radius:10px;padding:1.4rem 1.6rem;font-size:1.05rem;line-height:1.8;color:#ffcdd2;">
      <div style="font-size:1.3rem;font-weight:900;color:#ef9a9a;margin-bottom:0.5rem;">
        박사논문의 출발점
      </div>
      <div style="font-size:0.95rem;color:#90a4ae;margin-bottom:1.2rem;">
        규칙 기반 설계의 본질적 한계로 인해 다음 4가지 문제가 실험 분석에서 드러남 (논문 §IV-3)
      </div>
    """, unsafe_allow_html=True)
    for num, title, body in [
        ("1", "도메인 편차 반영 불가",
         "질의 분포의 도메인별 편차를 고정 규칙 테이블이 따라가지 못함. 새로운 도메인 데이터가 추가될수록 테이블과 실제 분포 간 괴리가 커짐."),
        ("2", "수작업 튜닝 의존",
         "기본 가중치 테이블과 λ(람다) 값을 도메인 전문가가 수동으로 설계·조정해야 함. 자동화가 불가능하며 확장성이 낮음."),
        ("3", "조건부 질의 성능 한계",
         "Conditional 질의에서 R-DWA F1_strict = 0.223으로 매우 낮음. 수치 제약·논리 연산이 복합된 질의를 규칙 테이블이 처리하는 데 한계 존재."),
        ("4", "도메인 이전 시 재튜닝 필수",
         "새 도메인 적용 시 가중치 테이블과 λ 전체를 처음부터 다시 설계해야 함. 이전 학습 결과를 재활용할 수 없음."),
    ]:
        st.markdown(f"""
        <div style="display:flex;gap:12px;margin-bottom:14px;align-items:flex-start;">
          <div style="background:#b71c1c;color:white;border-radius:50%;width:28px;height:28px;
                      display:flex;align-items:center;justify-content:center;
                      font-weight:700;font-size:0.95rem;flex-shrink:0;margin-top:2px;">{num}</div>
          <div>
            <div style="color:#ef9a9a;font-weight:700;font-size:1.05rem;">{title}</div>
            <div style="color:#cfd8dc;font-size:0.97rem;">{body}</div>
          </div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

@st.dialog("🎓 박사학위논문 (2026) — 상세 보기", width="large")
def dialog_phd():
    st.markdown("""
    <div style="background:#001a09;border-left:4px solid #66bb6a;border-radius:10px;padding:1.4rem 1.6rem;font-size:1.05rem;line-height:1.8;color:#c8e6c9;">
      <div style="font-size:1.3rem;font-weight:900;color:#a5d6a7;margin-bottom:1rem;">
        A-DWA — PPO로 가중치를 스스로 학습
      </div>
      R-DWA의 구조적 한계를 해소하기 위해, 가중치 결정 문제를
      <b style="color:#a5d6a7;">MDP (Markov Decision Process)</b>로 공식화하고
      <b style="color:#a5d6a7;">PPO (Proximal Policy Optimization)</b>로 최적 정책을 학습하는
      <b style="color:#a5d6a7;">A-DWA</b>를 제안.
      <hr style="border-color:#1b3a1e;margin:1rem 0;">
      <b style="color:#66bb6a;font-size:1.1rem;">핵심 해결책</b><br><br>
      · <b>18-dim 상태벡터</b>로 질의 분포를 실시간 포착<br>
      · <b>Actor-Critic (5,636 파라미터)</b> — 경량 학습형 설계<br>
      · 오프라인 보상 캐시 <b>330K 엔트리</b>로 학습 비용 89% 절감<br>
      · LLM 호출 <b>0회</b>로 PPO 학습 완료<br>
      <div style="background:#071a0d;border-radius:8px;padding:0.9rem 1.1rem;font-family:monospace;font-size:0.97rem;color:#a5d6a7;margin-top:1rem;">
        State s = [밀도 ρ 3 + 질의유형 ℓ 3 + 소스통계 σ 9 + 질의메타 μ 3]  (18-dim)<br>
        Actor π_θ(s): Linear(18→64) → Tanh×2 → Actor head(64→3, Softplus) → Dirichlet 평균<br>
        Output: (α, β, γ) ∈ Δ³  (합 = 1 보장)
      </div>
      <hr style="border-color:#1b3a1e;margin:1rem 0;">
      <b style="color:#66bb6a;font-size:1.1rem;">실증 성과</b><br><br>
    </div>
    """, unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    for col, (val, lbl) in zip([c1,c2,c3,c4], [
        ("F1_strict\n+6.2%", "vs R-DWA"),
        ("Conditional\n+36.7%", "vs R-DWA"),
        ("Discrete Oracle\n소폭 상회", "이산 격자 참조값"),
        ("비용\n89% 절감", "오프라인 캐시"),
    ]):
        with col:
            st.markdown(f"""
            <div style="background:#0d2a14;border:1px solid #2e7d32;border-radius:10px;
                        padding:0.8rem;text-align:center;">
              <div style="font-size:1.05rem;font-weight:900;color:#66bb6a;white-space:pre-line;">{val}</div>
              <div style="font-size:0.82rem;color:#607d8b;margin-top:4px;">{lbl}</div>
            </div>""", unsafe_allow_html=True)
    st.markdown("""
    <div style="margin-top:1rem;font-size:0.9rem;color:#607d8b;">
      3가지 연구 질문(RQ1~3) 실증 검증 완료 &nbsp;·&nbsp;
      BCa 95% CI Paired Bootstrap 통계 유의성 확인
    </div>
    """, unsafe_allow_html=True)

# ── 연구 배경: 선행 논문 → 한계 → 박사논문 ────────────────────────────────
with st.expander("📖 연구 배경 — 왜 이 박사논문을 쓰게 되었는가?", expanded=True):
    st.markdown('<div class="bg-section"><div class="bg-title">연구 동기 · Research Motivation</div>', unsafe_allow_html=True)

    col_prev, col_arrow1, col_limit, col_arrow2, col_phd = st.columns([5, 1, 5, 1, 5])

    with col_prev:
        btn_prev = st.button("🔍 크게 보기", key="btn_prev", use_container_width=True)
        if btn_prev:
            dialog_prev()
        st.markdown("""
        <div class="timeline-col rdwa">
          <div class="tc-head" style="color:#ffb74d;">📄 선행 논문 (JKSCI 2025)</div>
          <div class="tc-title" style="color:#ffe0b2;">Triple-Hybrid RAG<br>+ R-DWA 제안</div>
          <div class="tc-body">
            저자의 선행 연구[1]에서 세 가지 지식 소스(Vector·Graph·Ontology)를
            결합한 <b style="color:#ffcc80;">Triple-Hybrid RAG 프레임워크</b>와
            2단계 규칙 기반 가중치 알고리즘 <b style="color:#ffcc80;">R-DWA</b>를 최초 제안.<br><br>
            <b>R-DWA 작동 원리</b><br>
            ① 질의 유형(Simple/Multi-hop/Conditional) 룩업테이블로 기본 가중치 선택<br>
            ② 밀도 신호(s_e, s_r, s_c)로 λ 보정 → 정규화<br><br>
            <b>선행 논문 보고 수치</b><br>
            F1 0.86 · EM 0.78 · Faithfulness 0.89<br>
            <span style="font-size:0.85rem;color:#607d8b;">(평가 방법 이슈로 후속 재검증 필요 → 박사논문에서 Corrected Baseline 수립)</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

    with col_arrow1:
        st.markdown('<div class="arrow-col">→</div>', unsafe_allow_html=True)

    with col_limit:
        btn_limit = st.button("🔍 크게 보기", key="btn_limit", use_container_width=True)
        if btn_limit:
            dialog_limit()
        st.markdown("""
        <div class="timeline-col limit">
          <div class="tc-head" style="color:#ef5350;">⚠️ R-DWA의 구조적 한계 발견</div>
          <div class="tc-title" style="color:#ffcdd2;">박사논문의 출발점</div>
          <div class="tc-body" style="color:#cfd8dc;">
            규칙 기반 설계의 본질적 한계로 인해 다음 4가지 문제가
            실험 분석에서 드러남 (논문 §IV-3):
          </div>
          <br>
          <div class="limit-item">
            <div class="li-num">1</div>
            <div><b style="color:#ef9a9a;">도메인 편차 반영 불가</b><br>
            질의 분포의 도메인별 편차를 고정 규칙 테이블이 따라가지 못함</div>
          </div>
          <div class="limit-item">
            <div class="li-num">2</div>
            <div><b style="color:#ef9a9a;">수작업 튜닝 의존</b><br>
            기본 가중치 테이블과 λ(람다) 값을 전문가가 수동으로 설계·조정해야 함</div>
          </div>
          <div class="limit-item">
            <div class="li-num">3</div>
            <div><b style="color:#ef9a9a;">조건부 질의 성능 한계</b><br>
            Conditional 질의에서 개선 여지가 크게 존재
            (R-DWA F1_strict 0.223으로 낮음)</div>
          </div>
          <div class="limit-item">
            <div class="li-num">4</div>
            <div><b style="color:#ef9a9a;">도메인 이전 시 재튜닝 필수</b><br>
            새 도메인 적용 시 가중치 테이블·λ(람다) 전체를 다시 설계해야 함</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    with col_arrow2:
        st.markdown('<div class="arrow-col">→</div>', unsafe_allow_html=True)

    with col_phd:
        btn_phd = st.button("🔍 크게 보기", key="btn_phd", use_container_width=True)
        if btn_phd:
            dialog_phd()
        st.markdown("""
        <div class="timeline-col adwa">
          <div class="tc-head" style="color:#66bb6a;">🎓 박사학위논문 (2026)</div>
          <div class="tc-title" style="color:#c8e6c9;">A-DWA — PPO로<br>가중치를 스스로 학습</div>
          <div class="tc-body">
            R-DWA의 구조적 한계를 해소하기 위해,
            가중치 결정 문제를 <b style="color:#a5d6a7;">MDP (Markov Decision Process)</b>로
            공식화하고 <b style="color:#a5d6a7;">PPO (Proximal Policy Optimization)</b>로
            최적 정책을 학습하는 <b style="color:#a5d6a7;">A-DWA</b>를 제안.<br><br>
            <b>핵심 해결책</b><br>
            · 18-dim 상태벡터로 질의 분포를 실시간 포착<br>
            · Actor-Critic (5,636 파라미터) — 경량 학습형 설계<br>
            · 오프라인 보상 캐시 330K 엔트리로 학습 비용 89% 절감<br>
            · LLM 호출 0회로 PPO 학습 완료<br><br>
            <b>실증 성과</b><br>
          </div>
          <div>
            <span class="result-chip">F1_strict +6.2%</span>
            <span class="result-chip">Conditional +36.7%</span>
            <span class="result-chip">Discrete Oracle 소폭 상회</span>
            <span class="result-chip">비용 89% 절감</span>
          </div>
          <div style="font-size:0.85rem;color:#607d8b;margin-top:8px;">
            3가지 연구 질문(RQ1~3) 실증 검증 완료<br>
            BCa 95% CI Paired Bootstrap 통계 유의성 확인
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

c1, c2, c3, c4, c5 = st.columns(5)
for col, (val, lbl, delta) in zip([c1,c2,c3,c4,c5], [
    ("0.562", "A-DWA F1_strict",       "+6.2% vs R-DWA"),
    ("0.304", "Conditional F1_strict",  "+36.7% vs R-DWA"),
    ("5,636", "Actor-Critic 파라미터", "경량 설계"),
    ("5,000 QA", "합성 대학 벤치마크", "3-seed 평균"),
    ("89%",   "비용 절감",             "오프라인 캐시"),
]):
    with col:
        st.markdown(f"""
        <div class="metric-card">
          <div class="val">{val}</div>
          <div class="lbl">{lbl}</div>
          <div class="delta">{delta}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs([
    "🔬 파이프라인 시뮬레이터",
    "📐 가중치 탐색기",
    "📊 실험 결과",
    "🏗️ 아키텍처",
])

# ══════════════════════════════════════════════════════════════
# TAB 1 – Pipeline Simulator
# ══════════════════════════════════════════════════════════════
with tab1:
    # ── 논문 Fig. III-1 + 단계 설명 (토글 가능) ─────────────────
    with st.expander("📄 논문 Fig. III-1 — Triple-Hybrid RAG 4단계 파이프라인", expanded=True):
        fig_col, exp_col = st.columns([5, 4], gap="large")

        with fig_col:
            _fig_path = _HERE / "fig_iii1_crop.png"
            if _fig_path.exists():
                _fig_b64 = base64.b64encode(_fig_path.read_bytes()).decode()
                st.markdown(
                    f'<img src="data:image/png;base64,{_fig_b64}" '
                    f'style="width:100%;border-radius:8px;"/>',
                    unsafe_allow_html=True,
                )
            else:
                st.info("이미지 파일을 찾을 수 없습니다 (fig_iii1_crop.png)")

        with exp_col:
            st.markdown("""
            <div style="display:flex;flex-direction:column;gap:10px;">

            <div style="background:#0a1929;border-left:4px solid #4fc3f7;border-radius:8px;padding:0.9rem 1.1rem;">
              <div style="color:#4fc3f7;font-weight:700;font-size:1rem;margin-bottom:6px;">
                1️⃣ Query Analysis — 질의 분석
              </div>
              <div style="color:#cfd8dc;font-size:0.95rem;line-height:1.75;">
                <b>User Query</b>(자연어)가 <b>Query Analyzer</b>로 입력됩니다.<br>
                NER(개체명 인식) · 관계 키워드 · 수치 제약을 감지해<br>
                질의를 세 유형 중 하나로 분류합니다.<br><br>
                <span style="background:#1565c0;color:white;border-radius:4px;padding:1px 8px;font-size:0.88rem;">Simple</span>
                &nbsp;특정 개체 속성 단순 조회 → <b style="color:#4fc3f7;">α↑</b><br>
                <span style="background:#4a148c;color:white;border-radius:4px;padding:1px 8px;font-size:0.88rem;">Multi-hop</span>
                &nbsp;다단계 관계 탐색 필요 → <b style="color:#ce93d8;">β↑</b><br>
                <span style="background:#b71c1c;color:white;border-radius:4px;padding:1px 8px;font-size:0.88rem;">Conditional</span>
                &nbsp;수치·논리 제약 포함 → <b style="color:#ef9a9a;">γ↑</b>
              </div>
            </div>

            <div style="background:#0a1929;border-left:4px solid #ce93d8;border-radius:8px;padding:0.9rem 1.1rem;">
              <div style="color:#ce93d8;font-weight:700;font-size:1rem;margin-bottom:6px;">
                2️⃣ Density Signals — 밀도 신호 추출
              </div>
              <div style="color:#cfd8dc;font-size:0.95rem;line-height:1.75;">
                Query Analyzer가 질의 텍스트에서 3가지 밀도 신호를 산출합니다.<br><br>
                <b style="color:#4fc3f7;">s_e</b> (Entity Density) — Named Entity 빈도<br>
                &nbsp;&nbsp;예: 「홍성민 교수」→ s_e 높음<br>
                <b style="color:#ce93d8;">s_r</b> (Relation Density) — 관계 키워드 빈도<br>
                &nbsp;&nbsp;예: 「참여한」「개설한」→ s_r 높음<br>
                <b style="color:#ef9a9a;">s_c</b> (Constraint Density) — 수치/범위 조건 빈도<br>
                &nbsp;&nbsp;예: 「55세 이하」→ s_c 높음<br><br>
                이 세 값이 DWA 알고리즘의 핵심 입력이자<br>
                18-dim 상태 벡터의 첫 3차원입니다.
              </div>
            </div>

            <div style="background:#0a1929;border-left:4px solid #ffb74d;border-radius:8px;padding:0.9rem 1.1rem;">
              <div style="color:#ffb74d;font-weight:700;font-size:1rem;margin-bottom:6px;">
                3️⃣ DWA 가중치 결정 + Score Integration
              </div>
              <div style="color:#cfd8dc;font-size:0.95rem;line-height:1.75;">
                <b>R-DWA (규칙 기반)</b><br>
                Stage 1: 질의 유형 → 기본 가중치 (α_base, β_base, γ_base)<br>
                Stage 2: λ(람다) × 밀도 신호로 연속 보정 (α', β', γ')<br>
                Normalize: (α', β', γ') / (α'+β'+γ') → 합=1<br><br>
                <b>A-DWA (PPO 학습)</b><br>
                Actor π_θ(18-dim 상태) → Softplus → Dirichlet 평균 → (α, β, γ) ∈ Δ³<br><br>
                <b>통합 점수:</b><br>
                S_total = α·S_vector + β·S_graph + γ·S_ontology
              </div>
            </div>

            <div style="background:#0a1929;border-left:4px solid #66bb6a;border-radius:8px;padding:0.9rem 1.1rem;">
              <div style="color:#66bb6a;font-weight:700;font-size:1rem;margin-bottom:6px;">
                4️⃣ Retrieval & Answer — 검색 + 답변 생성
              </div>
              <div style="color:#cfd8dc;font-size:0.95rem;line-height:1.75;">
                <b style="color:#4fc3f7;">Vector Store</b> (FAISS / ChromaDB)<br>
                &nbsp;&nbsp;텍스트 임베딩 유사도로 관련 문서 검색<br>
                <b style="color:#ce93d8;">Knowledge Graph</b> (BFS, scalable to Neo4j)<br>
                &nbsp;&nbsp;관계 엣지를 따라 다단계 탐색<br>
                <b style="color:#ef9a9a;">Ontology Engine</b> (OWL2 / HermiT)<br>
                &nbsp;&nbsp;수치 제약·클래스 계층 추론<br><br>
                가중치 적용 후 <b>merge_contexts</b>로 통합,<br>
                <b>LLM</b> (GPT-4o-mini, T=0.0) → <b>Final Answer</b>
              </div>
            </div>

            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # ── 실시간 파이프라인 시뮬레이터 ───────────────────────────
    st.subheader("실시간 파이프라인 시뮬레이터")
    st.caption("질의 선택 → 위 논문 그림 기준 4단계를 실시간으로 계산합니다")

    for k, v in [("_custom_active", ""), ("_input_ver", 0)]:
        if k not in st.session_state:
            st.session_state[k] = v

    def _on_custom_enter():
        cur_key = f"custom_input_{st.session_state['_input_ver']}"
        v = st.session_state.get(cur_key, "").strip()
        if v:
            st.session_state["_custom_active"] = v
        st.session_state["_input_ver"] += 1   # 키 교체 → 새 빈 위젯 생성

    def _on_query_change():
        st.session_state["_custom_active"] = ""
        st.session_state["_input_ver"] += 1   # 예시 선택 시 입력창도 초기화

    sim_l, sim_r = st.columns([1, 1], gap="large")
    with sim_l:
        query_input = st.selectbox("예시 질의 선택", list(QUERY_EXAMPLES.keys()), index=0,
                                   key="query_select", on_change=_on_query_change)
    with sim_r:
        input_key = f"custom_input_{st.session_state['_input_ver']}"
        st.text_input("직접 입력 (Enter로 적용)", placeholder="예: 융합공학과 조교수 중 30대는?",
                      key=input_key, on_change=_on_custom_enter)
        if st.session_state["_custom_active"]:
            st.caption(f"✅ 적용 중: {st.session_state['_custom_active']}")

    custom = st.session_state["_custom_active"]

    if custom.strip():
        query_text = custom.strip()
        # 예시 질의와 완전 일치하면 예시 데이터 그대로 사용
        if query_text in QUERY_EXAMPLES:
            detected_type = QUERY_EXAMPLES[query_text]
            custom_reason = ""
            is_custom = False   # 예시 데이터 경로로 처리
        else:
            # 키워드 기반 분류 + 구조화된 reason 생성
            cond_kws  = ["이하", "이상", "세", "이내", "초과", "미만"]
            multi_kws = ["모두", "전체", "목록", "참여", "관련", "개설"]
            found_cond  = [k for k in cond_kws  if k in query_text]
            found_multi = [k for k in multi_kws if k in query_text]
            if found_cond:
                detected_type = "conditional"
                custom_reason = {
                    "keywords": found_cond,
                    "rule": "수치 제약 패턴 감지 (나이/이하/이상/이내)",
                    "why": f"「{'·'.join(found_cond)}」라는 수치 부등식 제약이 포함됨. "
                           "단순 문서 검색으로는 나이 필터링 불가 → OWL2 Ontology 추론(γ↑)이 필수이므로 Conditional 분류",
                }
            elif found_multi:
                detected_type = "multi-hop"
                custom_reason = {
                    "keywords": found_multi,
                    "rule": "다단계 관계·집합 반환 키워드 감지",
                    "why": f"「{'·'.join(found_multi)}」키워드가 다수 개체 열거 또는 관계 탐색을 요구함. "
                           "Graph BFS(β↑)로 다단계 연결을 탐색해야 하므로 Multi-hop 분류",
                }
            else:
                detected_type = "simple"
                # 첫 번째 명사구(2~5자 연속 단어) 추출 시도
                import re as _re
                nouns = _re.findall(r'[가-힣]{2,5}', query_text)
                kw_list = nouns[:2] if nouns else ["질의 텍스트"]
                custom_reason = {
                    "keywords": kw_list,
                    "rule": "수치 제약·관계 키워드 없음 → 단순 속성 조회",
                    "why": f"수치 부등식이나 다단계 관계 키워드가 없음. "
                           "특정 개체 속성을 Vector 유사도 검색(α↑)만으로 답변 가능 → Simple 분류",
                }
            is_custom = True
    else:
        query_text = query_input
        detected_type = QUERY_EXAMPLES[query_input]
        custom_reason = ""
        is_custom = False

    # 파이프라인은 A-DWA 기준으로 표시 (Step②에서 R-DWA와 나란히 비교)
    type_lbl, type_css = TYPE_LABELS[detected_type]
    density = DENSITY_SIM[detected_type]
    base_w  = RDWA_BASE[detected_type]
    adj_w   = rdwa_stage2(base_w, density)
    lw      = ADWA_WEIGHTS[detected_type]
    active_w = lw  # A-DWA 고정
    use_adwa = True

    st.markdown("---")

    st.markdown(
        f"<span style='font-size:0.95rem;color:#78909c;font-weight:500;'>질의</span>&nbsp;&nbsp;"
        f"<span style='font-size:1.45rem;font-weight:900;color:#e8f4fd;letter-spacing:-0.3px;'>{query_text}</span>"
        f"&nbsp;&nbsp;<span class='tag {type_css}' style='font-size:0.92rem;vertical-align:middle;'>{type_lbl}</span>",
        unsafe_allow_html=True,
    )
    st.markdown("")

    if "pipeline_open" not in st.session_state:
        st.session_state["pipeline_open"] = True
    btn_label = "🔼 파이프라인 접기" if st.session_state["pipeline_open"] else "🔽 파이프라인 펼치기 (4단계)"
    if st.button(btn_label, key="toggle_pipeline"):
        st.session_state["pipeline_open"] = not st.session_state["pipeline_open"]
        st.rerun()

    if st.session_state["pipeline_open"]:

     # ── STEP 1: Query Analysis ──────────────────────────────────
     with st.expander("① Query Analysis — 질의 유형 감지 + 밀도 신호", expanded=True):
        ca, cb = st.columns([1, 1.2])
        with ca:
            desc_map = {
                "simple":      "개체(Entity) 중심 단순 속성 조회",
                "multi-hop":   "다단계 관계 추론 필요 (Graph BFS)",
                "conditional": "수치·논리 제약 포함 (Ontology 추론)",
            }
            # 분류 이유
            r = custom_reason if is_custom else QUERY_REASONS[query_text]
            kw_tags = " ".join(f'<span class="r-kw">{k}</span>' for k in r["keywords"])
            label = "분류 근거 (자동 감지)" if is_custom else "분류 근거"
            reason_html = f"""
                <div class="reason-box">
                  <div class="r-label">{label}</div>
                  <div class="r-body">
                    감지 키워드: {kw_tags}<br>
                    규칙: {r['rule']}<br>
                    이유: {r['why']}
                  </div>
                </div>"""

            st.markdown(f"""
            <div class="step-box">
              <div class="step-title">🔍 질의 유형: {type_lbl}</div>
              <div class="step-body">
                {desc_map[detected_type]}<br>
                <span style="font-size:0.92rem;color:#78909c;">
                  Simple → α↑(Vector) &nbsp;|&nbsp; Multi-hop → β↑(Graph) &nbsp;|&nbsp; Conditional → γ↑(Ontology)
                </span>
              </div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown(reason_html, unsafe_allow_html=True)

            st.markdown(f"""
            <div class="step-box" style="margin-top:8px;">
              <div class="step-title">📡 밀도 신호 → 18-dim 상태벡터 핵심 3개</div>
              <div class="step-body">
                s_e (Entity Density)     = <b>{density['s_e']:.2f}</b> — 개체 밀도<br>
                s_r (Relation Density)   = <b>{density['s_r']:.2f}</b> — 관계 밀도<br>
                s_c (Constraint Density) = <b>{density['s_c']:.2f}</b> — 수치 제약 밀도
              </div>
            </div>""", unsafe_allow_html=True)

        with cb:
            fig_d = go.Figure(go.Bar(
                x=["s_e (개체)", "s_r (관계)", "s_c (제약)"],
                y=[density["s_e"], density["s_r"], density["s_c"]],
                marker_color=["#4fc3f7", "#ce93d8", "#ef9a9a"],
                text=[f"{v:.2f}" for v in [density["s_e"], density["s_r"], density["s_c"]]],
                textposition="outside",
            ))
            fig_d.update_layout(
                title="밀도 신호 (Density Signals)", height=230,
                margin=dict(t=40,b=20,l=20,r=20), yaxis=dict(range=[0,1.15]),
                paper_bgcolor=DARK_BG, plot_bgcolor=DARK_BG, font_color="#cfd8dc",
            )
            st.plotly_chart(fig_d, key="density_bar", use_container_width=True)

    # ── STEP 2: DWA ──────────────────────────────────────────────
     with st.expander("② DWA 가중치 결정 — R-DWA 2단계 / A-DWA PPO", expanded=True):
        if not use_adwa:
            st.markdown("##### R-DWA : 2-Stage Rule-based Dynamic Weighting Algorithm")
            cs1, cs2, cs3 = st.columns(3)

            _sr, _sc = density["s_r"], density["s_c"]
            a_raw = base_w["alpha"] * (1 - RDWA_LAMBDA * (_sr + _sc) / 2)
            b_raw = base_w["beta"]  + RDWA_LAMBDA * _sr * (1 - base_w["beta"])
            g_raw = base_w["gamma"] + RDWA_LAMBDA * _sc * (1 - base_w["gamma"])

            with cs1:
                st.markdown(f"""
                <div class="rdwa-box">
                  <div class="rdwa-title"><span class="stage-badge">Stage 1</span>기본 가중치 (Base Weight)</div>
                  <div class="rdwa-body">
                    질의 유형 [{type_lbl}] 룩업테이블<br><br>
                    α = {base_w['alpha']:.2f}<br>
                    β = {base_w['beta']:.2f}<br>
                    γ = {base_w['gamma']:.2f}
                  </div>
                </div>""", unsafe_allow_html=True)
            with cs2:
                st.markdown(f"""
                <div class="rdwa-box">
                  <div class="rdwa-title"><span class="stage-badge">Stage 2</span>λ(람다) 연속 보정 · λ={RDWA_LAMBDA}</div>
                  <div class="rdwa-body">
                    α' = {base_w['alpha']:.2f}×(1−{RDWA_LAMBDA}×({density['s_r']:.2f}+{density['s_c']:.2f})/2) = <b>{a_raw:.3f}</b><br>
                    β' = {base_w['beta']:.2f} + {RDWA_LAMBDA}×{density['s_r']:.2f}×(1−{base_w['beta']:.2f}) = <b>{b_raw:.3f}</b><br>
                    γ' = {base_w['gamma']:.2f} + {RDWA_LAMBDA}×{density['s_c']:.2f}×(1−{base_w['gamma']:.2f}) = <b>{g_raw:.3f}</b>
                  </div>
                </div>""", unsafe_allow_html=True)
            with cs3:
                s_raw = a_raw + b_raw + g_raw
                st.markdown(f"""
                <div class="rdwa-box">
                  <div class="rdwa-title"><span class="stage-badge">Normalize</span>정규화 (합=1)</div>
                  <div class="rdwa-body">
                    S = {a_raw:.3f}+{b_raw:.3f}+{g_raw:.3f} = {s_raw:.3f}<br><br>
                    α = {a_raw:.3f}/S = <b>{adj_w['alpha']:.3f}</b><br>
                    β = {b_raw:.3f}/S = <b>{adj_w['beta']:.3f}</b><br>
                    γ = {g_raw:.3f}/S = <b>{adj_w['gamma']:.3f}</b>
                  </div>
                </div>""", unsafe_allow_html=True)
        else:
            st.markdown("##### A-DWA : PPO Actor-Critic 학습 기반 가중치")
            # ── 18-dim 입력 구성 설명 ──
            st.markdown(f"""
            <div style="background:#0d1a10;border:1px solid #388e3c;border-radius:10px;padding:14px 18px;margin-bottom:10px;">
              <div style="color:#a5d6a7;font-size:13px;font-weight:700;margin-bottom:10px;">
                📐 왜 18차원인가? — 입력 상태벡터 s 구성
              </div>
              <div style="display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:10px;font-size:12px;">
                <div style="background:#0a2510;border-left:3px solid #26a69a;border-radius:6px;padding:10px;">
                  <div style="color:#80cbc4;font-weight:700;margin-bottom:6px;">🔵 밀도 신호 ρ · 3차원</div>
                  <div style="color:#cfd8dc;line-height:1.8;">
                    s_e = <b>{density['s_e']:.2f}</b> &nbsp;개체밀도<br>
                    s_r = <b>{density['s_r']:.2f}</b> &nbsp;관계밀도<br>
                    s_c = <b>{density['s_c']:.2f}</b> &nbsp;제약밀도
                  </div>
                  <div style="color:#546e7a;font-size:11px;margin-top:6px;">질의에 얼마나 많은 개체·관계·수치 제약이 담겼는지</div>
                </div>
                <div style="background:#1a1000;border-left:3px solid #f57c00;border-radius:6px;padding:10px;">
                  <div style="color:#ffb74d;font-weight:700;margin-bottom:6px;">🟠 질의 유형 one-hot ℓ · 3차원</div>
                  <div style="color:#cfd8dc;line-height:1.8;">
                    [1, 0, 0] = Simple<br>
                    [0, 1, 0] = Multi-hop<br>
                    [0, 0, 1] = Conditional
                  </div>
                  <div style="color:#546e7a;font-size:11px;margin-top:6px;">어떤 유형의 질의인지 직접 알려줌</div>
                </div>
                <div style="background:#0a1020;border-left:3px solid #5c6bc0;border-radius:6px;padding:10px;">
                  <div style="color:#9fa8da;font-weight:700;margin-bottom:6px;">🟣 소스별 예비검색 통계 σ · 9차원</div>
                  <div style="color:#cfd8dc;line-height:1.8;font-size:11.5px;">
                    Vector · Graph · Ontology<br>
                    각 소스 예비검색 후보의<br>
                    요약 통계 3개씩 (3×3)
                  </div>
                  <div style="color:#546e7a;font-size:11px;margin-top:6px;">추가 검색 없이 공통 파이프라인 후보 품질 신호 반영</div>
                </div>
                <div style="background:#1a0a14;border-left:3px solid #ad4d8c;border-radius:6px;padding:10px;">
                  <div style="color:#f48fb1;font-weight:700;margin-bottom:6px;">🟤 질의 메타 μ · 3차원</div>
                  <div style="color:#cfd8dc;line-height:1.8;">
                    질의 길이<br>
                    엔티티 수<br>
                    제약 표현 수
                  </div>
                  <div style="color:#546e7a;font-size:11px;margin-top:6px;">질의 복잡도·검색 필요성 보조 신호</div>
                </div>
              </div>
              <div style="text-align:center;margin-top:10px;color:#546e7a;font-size:12px;">
                <span style="color:#80cbc4;font-weight:700;">3</span> (밀도 ρ) +
                <span style="color:#ffb74d;font-weight:700;">3</span> (질의유형 ℓ) +
                <span style="color:#9fa8da;font-weight:700;">9</span> (소스통계 σ) +
                <span style="color:#f48fb1;font-weight:700;">3</span> (질의메타 μ) =
                <span style="color:#a5d6a7;font-size:14px;font-weight:700;">18차원</span>
              </div>
            </div>
            """, unsafe_allow_html=True)

            # ── 신경망 구조 다이어그램 (components.html → SVG 확실히 렌더링) ──
            se, sr, sc = density['s_e'], density['s_r'], density['s_c']
            hidden_bars = "".join(
                f'<rect x="310" y="{50+i*14}" width="48" height="10" rx="3" fill="#1565c0" opacity="{0.45+i*0.07:.2f}"/>'
                for i in range(8)
            )
            conn_in_hid = "".join(
                f'<line x1="115" y1="{30+i*20}" x2="296" y2="{55+j*14}" stroke="#1e3a2a" stroke-width="0.6"/>'
                for i in range(8) for j in range(8)
            )
            conn_hid_out = "".join(
                f'<line x1="374" y1="{55+i*14}" x2="547" y2="{74+j*56}" stroke="#1e2a3a" stroke-width="0.6"/>'
                for i in range(8) for j in range(2)
            )
            nn_html = f"""<!DOCTYPE html><html><head><meta charset="utf-8"></head><body style="margin:0;padding:0;background:#0d1117;">
<div style="background:#0d1117;border:1.5px solid #4caf50;border-radius:10px;padding:16px 10px 12px 10px;">
  <div style="color:#a5d6a7;font-size:14px;font-weight:700;text-align:center;margin-bottom:12px;font-family:sans-serif;">
    &#129504; PPO Actor-Critic 신경망 구조 &nbsp;·&nbsp; 5,636 파라미터
  </div>
  <svg viewBox="0 0 860 230" xmlns="http://www.w3.org/2000/svg"
       style="width:100%;display:block;font-family:'Segoe UI',Arial,sans-serif;">
    <defs>
      <marker id="a" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto">
        <polygon points="0 0,8 3,0 6" fill="#546e7a"/>
      </marker>
    </defs>

    <!-- 연결선 (배경) -->
    {conn_in_hid}
    {conn_hid_out}

    <!-- ■ 입력 그룹: 밀도신호 ρ (3) -->
    <rect x="8" y="14" width="100" height="44" rx="7" fill="#0a2510" stroke="#26a69a" stroke-width="1.5"/>
    <text x="58" y="28" fill="#80cbc4" font-size="10.5" font-weight="700" text-anchor="middle">밀도 신호 ρ</text>
    <text x="58" y="41" fill="#cfd8dc" font-size="9" text-anchor="middle">s_e={se:.2f} s_r={sr:.2f}</text>
    <text x="58" y="52" fill="#26a69a" font-size="10" font-weight="700" text-anchor="middle">s_c={sc:.2f} &#215; 3</text>

    <!-- ■ 입력 그룹: 질의유형 ℓ (3) -->
    <rect x="8" y="62" width="100" height="40" rx="7" fill="#1a1000" stroke="#f57c00" stroke-width="1.5"/>
    <text x="58" y="77" fill="#ffb74d" font-size="10.5" font-weight="700" text-anchor="middle">질의 유형 ℓ</text>
    <text x="58" y="90" fill="#cfd8dc" font-size="9" text-anchor="middle">one-hot 벡터</text>
    <text x="58" y="99" fill="#f57c00" font-size="10" font-weight="700" text-anchor="middle">&#215; 3</text>

    <!-- ■ 입력 그룹: 소스통계 σ (9) -->
    <rect x="8" y="106" width="100" height="52" rx="7" fill="#0a1020" stroke="#5c6bc0" stroke-width="1.5"/>
    <text x="58" y="120" fill="#9fa8da" font-size="10.5" font-weight="700" text-anchor="middle">소스 통계 σ</text>
    <text x="58" y="133" fill="#cfd8dc" font-size="9" text-anchor="middle">Vector·Graph·Onto</text>
    <text x="58" y="144" fill="#cfd8dc" font-size="9" text-anchor="middle">각 3개 (3&#215;3)</text>
    <text x="58" y="155" fill="#5c6bc0" font-size="10" font-weight="700" text-anchor="middle">&#215; 9</text>

    <!-- ■ 입력 그룹: 질의메타 μ (3) -->
    <rect x="8" y="162" width="100" height="44" rx="7" fill="#1a0a14" stroke="#ad4d8c" stroke-width="1.5"/>
    <text x="58" y="177" fill="#f48fb1" font-size="10.5" font-weight="700" text-anchor="middle">질의 메타 μ</text>
    <text x="58" y="190" fill="#cfd8dc" font-size="9" text-anchor="middle">길이·엔티티·제약</text>
    <text x="58" y="201" fill="#ad4d8c" font-size="10" font-weight="700" text-anchor="middle">&#215; 3</text>

    <!-- 18 합계 배지 -->
    <rect x="114" y="98" width="46" height="26" rx="5" fill="#1b3a1f" stroke="#66bb6a" stroke-width="1.5"/>
    <text x="137" y="116" fill="#a5d6a7" font-size="14" font-weight="800" text-anchor="middle">18</text>

    <!-- Shared Linear1 화살표 -->
    <line x1="162" y1="111" x2="292" y2="111" stroke="#42a5f5" stroke-width="2.5" marker-end="url(#a)"/>
    <rect x="170" y="97" width="92" height="22" rx="5" fill="#0d2040"/>
    <text x="216" y="112" fill="#42a5f5" font-size="10.5" font-weight="700" text-anchor="middle">Linear 18→64</text>

    <!-- ■ 공유 은닉층 (64, Tanh ×2) -->
    <rect x="294" y="38" width="80" height="148" rx="8" fill="#0d1f40" stroke="#1565c0" stroke-width="2"/>
    {hidden_bars}
    <text x="334" y="30" fill="#90caf9" font-size="9" text-anchor="middle">Shared Linear ×2</text>
    <text x="334" y="200" fill="#90caf9" font-size="9" text-anchor="middle">64→64 · Tanh</text>
    <text x="334" y="214" fill="#42a5f5" font-size="13" font-weight="700" text-anchor="middle">64</text>

    <!-- Tanh 화살표 -->
    <line x1="376" y1="111" x2="543" y2="111" stroke="#ff7043" stroke-width="2.5" marker-end="url(#a)"/>
    <rect x="412" y="97" width="54" height="22" rx="5" fill="#2a1008"/>
    <text x="439" y="112" fill="#ff7043" font-size="12" font-weight="700" text-anchor="middle">Tanh</text>

    <!-- ■ Actor head -->
    <rect x="545" y="40" width="124" height="54" rx="8" fill="#0d2010" stroke="#4caf50" stroke-width="2"/>
    <text x="607" y="60" fill="#a5d6a7" font-size="13" font-weight="700" text-anchor="middle">Actor head</text>
    <text x="607" y="76" fill="#81c784" font-size="9.5" text-anchor="middle">64→3 · Softplus</text>
    <text x="607" y="88" fill="#81c784" font-size="9.5" text-anchor="middle">Dirichlet 농도 c</text>

    <!-- ■ Critic head -->
    <rect x="545" y="108" width="124" height="50" rx="8" fill="#1a0838" stroke="#9575cd" stroke-width="2"/>
    <text x="607" y="130" fill="#ce93d8" font-size="13" font-weight="700" text-anchor="middle">Critic head</text>
    <text x="607" y="147" fill="#b39ddb" font-size="9.5" text-anchor="middle">64→1 · V(s) 추정</text>

    <!-- Dirichlet 평균 화살표 (Actor 출력) -->
    <line x1="671" y1="67" x2="722" y2="67" stroke="#ab47bc" stroke-width="2.5" marker-end="url(#a)"/>
    <text x="700" y="58" fill="#ab47bc" font-size="9" font-weight="700" text-anchor="middle">Dirichlet</text>
    <text x="700" y="68" fill="#ab47bc" font-size="9" font-weight="700" text-anchor="middle">평균</text>

    <!-- ■ 출력 (α,β,γ) -->
    <rect x="724" y="38" width="128" height="62" rx="8" fill="#0a1a08" stroke="#66bb6a" stroke-width="2"/>
    <text x="788" y="58" fill="#a5d6a7" font-size="13" font-weight="700" text-anchor="middle">α · β · γ</text>
    <text x="788" y="74" fill="#cfd8dc" font-size="11" text-anchor="middle">{lw['alpha']:.3f} &#183; {lw['beta']:.3f} &#183; {lw['gamma']:.3f}</text>
    <text x="788" y="89" fill="#66bb6a" font-size="9.5" text-anchor="middle">&#8712; &#916;&#179; (합 = 1) · ā = c/‖c‖&#8321;</text>

    <!-- Critic 보상 출력 -->
    <line x1="671" y1="133" x2="722" y2="160" stroke="#9575cd" stroke-width="1.5" stroke-dasharray="5,3" marker-end="url(#a)"/>
    <rect x="724" y="148" width="128" height="38" rx="7" fill="#1a0838" stroke="#6a1b9a" stroke-width="1.5"/>
    <text x="788" y="165" fill="#ce93d8" font-size="11" font-weight="700" text-anchor="middle">V(s) 보상 예측</text>
    <text x="788" y="180" fill="#9575cd" font-size="10" text-anchor="middle">PPO GAE &#955;=0.95 · 출력</text>

    <!-- 하단 수식 -->
    <text x="430" y="225" fill="#546e7a" font-size="10.5" text-anchor="middle">
      18 &#8594;
      <tspan fill="#42a5f5" font-weight="700">Linear(64)</tspan>
      <tspan fill="#546e7a"> &#8594; </tspan>
      <tspan fill="#ff7043" font-weight="700">Tanh ×2</tspan>
      <tspan fill="#546e7a"> &#8594; [</tspan>
      <tspan fill="#a5d6a7" font-weight="700">Actor 64→3</tspan>
      <tspan fill="#546e7a"> | </tspan>
      <tspan fill="#ce93d8" font-weight="700">Critic 64→1</tspan>
      <tspan fill="#546e7a">] &#8594; </tspan>
      <tspan fill="#ab47bc" font-weight="700">Dirichlet</tspan>
      <tspan fill="#546e7a"> → (α,β,γ)</tspan>
    </text>
  </svg>
</div>
</body></html>"""
            components.html(nn_html, height=310, scrolling=False)

            # ── 현재 질의 출력 요약 ──
            st.markdown(f"""
            <div class="adwa-box" style="margin-top:8px;">
              <div class="adwa-title">현재 질의 추론 결과 — MDP State → Dirichlet 평균 → (α,β,γ) ∈ Δ³</div>
              <div class="adwa-body">
                입력 s = [밀도 ρ(s_e=<b>{density['s_e']:.2f}</b>, s_r=<b>{density['s_r']:.2f}</b>, s_c=<b>{density['s_c']:.2f}</b>) 3,
                질의유형 ℓ 3, 소스통계 σ 9, 질의메타 μ 3] → 합계 <b>18차원</b><br><br>
                출력 (<b>Dirichlet 평균</b>):&nbsp;&nbsp;
                α = <b style="color:#66bb6a;">{lw['alpha']:.3f}</b> &nbsp;·&nbsp;
                β = <b style="color:#42a5f5;">{lw['beta']:.3f}</b> &nbsp;·&nbsp;
                γ = <b style="color:#ab47bc;">{lw['gamma']:.3f}</b>
                &nbsp;&nbsp;→ 합 = <b>{lw['alpha']+lw['beta']+lw['gamma']:.3f}</b>
              </div>
            </div>""", unsafe_allow_html=True)

        # 가중치 바 (A-DWA 왼쪽, R-DWA 오른쪽)
        ca, cb = st.columns(2)
        with ca:
            st.markdown(
                weight_bar_html(lw, "A-DWA (PPO) 학습된 가중치", "#66bb6a"),
                unsafe_allow_html=True,
            )
        with cb:
            st.markdown(
                weight_bar_html(adj_w, "R-DWA 최종 가중치 (Stage2 보정 후)", "#ffb74d"),
                unsafe_allow_html=True,
            )

        # 3-simplex — A-DWA 수치 강조 표시
        fig_tern = go.Figure()
        _ref3 = [
            (lw["gamma"],    lw["beta"],    lw["alpha"]),
            (adj_w["gamma"], adj_w["beta"], adj_w["alpha"]),
            (1/3,            1/3,           1/3),
        ]
        fig_tern.add_trace(go.Scatterternary(
            a=[p[0] for p in _ref3],
            b=[p[1] for p in _ref3],
            c=[p[2] for p in _ref3],
            mode="markers",
            hovertext=[
                f"A-DWA (PPO 학습)<br>α={lw['alpha']:.3f}  β={lw['beta']:.3f}  γ={lw['gamma']:.3f}",
                f"R-DWA (2단계 보정)<br>α={adj_w['alpha']:.3f}  β={adj_w['beta']:.3f}  γ={adj_w['gamma']:.3f}",
                "Uniform (균등 배분)<br>α=0.333  β=0.333  γ=0.333",
            ],
            hoverinfo="text",
            marker=dict(
                size=[26, 18, 12],
                color=["#66bb6a", "#ffb74d", "#78909c"],
                symbol=["star", "diamond", "circle"],
                line=dict(color="white", width=2),
            ),
        ))
        fig_tern.update_layout(
            ternary=dict(
                sum=1,
                aaxis=dict(title="γ Ontology", color="#ef9a9a"),
                baxis=dict(title="β Graph",    color="#ce93d8"),
                caxis=dict(title="α Vector",   color="#4fc3f7"),
                bgcolor=DARK_BG,
            ),
            paper_bgcolor=DARK_BG, font_color="#cfd8dc",
            title=f"3-simplex Δ³ — 가중치 위치 비교 ({type_lbl})",
            height=400, margin=dict(t=55, b=70, l=80, r=80), showlegend=False,
        )
        st.plotly_chart(fig_tern, key="simplex_pipeline", use_container_width=True)
        # 가중치 수치 표
        st.markdown(f"""
        <div style="display:flex;gap:12px;margin-top:4px;flex-wrap:wrap;">
          <div style="background:#001a09;border:1px solid #66bb6a;border-radius:8px;padding:6px 14px;font-size:0.92rem;">
            <span style="color:#66bb6a;font-weight:700;">★ A-DWA</span>&nbsp;
            α=<b>{lw['alpha']:.3f}</b> &nbsp;β=<b>{lw['beta']:.3f}</b> &nbsp;γ=<b>{lw['gamma']:.3f}</b>
          </div>
          <div style="background:#1a1200;border:1px solid #ffb74d;border-radius:8px;padding:6px 14px;font-size:0.92rem;">
            <span style="color:#ffb74d;font-weight:700;">◆ R-DWA</span>&nbsp;
            α=<b>{adj_w['alpha']:.3f}</b> &nbsp;β=<b>{adj_w['beta']:.3f}</b> &nbsp;γ=<b>{adj_w['gamma']:.3f}</b>
          </div>
          <div style="background:#0d1b2a;border:1px solid #78909c;border-radius:8px;padding:6px 14px;font-size:0.92rem;">
            <span style="color:#78909c;font-weight:700;">● Uniform</span>&nbsp;
            α=0.333 &nbsp;β=0.333 &nbsp;γ=0.333
          </div>
        </div>
        """, unsafe_allow_html=True)

    # ── STEP 3: 지식 소스 검색 ──────────────────────────────────
     with st.expander("③ 지식 소스 병렬 검색 (Vector / Graph / Ontology)", expanded=True):
        ca, cb, cc = st.columns(3)
        with ca:
            st.markdown(f"""
            <div class="step-box">
              <div class="step-title" style="color:#4fc3f7;">🔵 Vector Store (FAISS)</div>
              <div class="step-body">
                적용 가중치 α = <b>{active_w['alpha']:.3f}</b><br>
                임베딩: text-embedding-3-small 1536-dim<br>
                코퍼스: 2,542 문서
              </div>
            </div>""", unsafe_allow_html=True)
            for r in VECTOR_RESULTS[detected_type]:
                st.markdown(f"- {r}")
        with cb:
            st.markdown(f"""
            <div class="step-box">
              <div class="step-title" style="color:#ce93d8;">🟣 Graph Store (NetworkX BFS)</div>
              <div class="step-body">
                적용 가중치 β = <b>{active_w['beta']:.3f}</b><br>
                노드: 60 학과 + 577 교수 + 1,505 과목 + 400 프로젝트<br>
                엣지: 1,158 협력관계 · 탐색: BFS max_depth = 3
              </div>
            </div>""", unsafe_allow_html=True)
            for r in GRAPH_RESULTS[detected_type]:
                st.code(r, language=None)
        with cc:
            st.markdown(f"""
            <div class="step-box">
              <div class="step-title" style="color:#ef9a9a;">🔴 Ontology (OWL2 / HermiT)</div>
              <div class="step-body">
                적용 가중치 γ = <b>{active_w['gamma']:.3f}</b><br>
                추론: Owlready2 HermiT 추론기<br>
                처리: 클래스 계층 + 수치 제약
              </div>
            </div>""", unsafe_allow_html=True)
            for r in ONTOLOGY_RESULTS[detected_type]:
                st.markdown(f"- {r}")

    # ── STEP 4: merge → LLM ──────────────────────────────────────
     with st.expander("④ merge_contexts → Score Fusion → LLM → 최종 답변", expanded=True):
        st.markdown(f"""
        <div class="step-box">
          <div class="step-title">Score Fusion 공식</div>
          <div class="step-body" style="font-family:monospace;">
            S_total = {active_w['alpha']:.3f}×S_vector + {active_w['beta']:.3f}×S_graph + {active_w['gamma']:.3f}×S_ontology
          </div>
        </div>""", unsafe_allow_html=True)

        answers = {
            "simple":      "안기찬 교수는 벤처대학원에서 창업경영론, 기술사업화전략을 담당하고 있습니다.",
            "multi-hop":   "문남미 교수가 참여한 연구 프로젝트: Triple-Hybrid RAG 프로젝트(2024~2026), NLP 기반 지식그래프 구축(2022~2024)",
            "conditional": "벤처대학원 심사위원 중 55세 이하: 문남미 교수(53세), 오삼권 교수(50세), 최유주 교수(47세)",
        }
        st.success(f"🤖 **LLM 답변 (GPT-4o-mini, T=0.0):** {answers[detected_type]}")

        f1_r = {"simple":0.874,"multi-hop":0.354,"conditional":0.223}[detected_type]
        f1_l = {"simple":0.906,"multi-hop":0.365,"conditional":0.304}[detected_type]
        imp  = (f1_l - f1_r) / f1_r * 100
        active_f1 = f1_l if use_adwa else f1_r
        st.markdown(
            f"예상 F1_strict — "
            f"<span style='color:#ffb74d'>R-DWA: {f1_r:.3f}</span>"
            f" &nbsp;→&nbsp; "
            f"<span style='color:#66bb6a;font-weight:700'>A-DWA: {f1_l:.3f}</span>"
            f"<span style='color:#66bb6a'> (+{imp:.1f}%)</span>"
            f" &nbsp;|&nbsp; 이 시뮬레이터는 <span style='color:#66bb6a;font-weight:700'>A-DWA 고정</span> 적용",
            unsafe_allow_html=True,
        )

# ══════════════════════════════════════════════════════════════
# TAB 2 – Weight Explorer
# ══════════════════════════════════════════════════════════════
with tab2:
    st.subheader("가중치 탐색기 (3-simplex Δ³)")
    st.caption("α + β + γ = 1 제약 아래 가중치를 직접 조정하고 예상 F1을 확인하세요.")

    col_ctrl, col_tern = st.columns([1, 1.6], gap="large")
    with col_ctrl:
        qt = st.radio("질의 유형", ["Simple", "Multi-hop", "Conditional"], horizontal=True, key="qt_radio")
        qt_key = {"Simple":"simple","Multi-hop":"multi-hop","Conditional":"conditional"}[qt]

        # key 에 qt_key 포함 → 유형 변경 시 슬라이더 초기화
        alpha = st.slider("α (Vector)", 0.0, 1.0,
                          float(RDWA_BASE[qt_key]["alpha"]), 0.01,
                          key=f"sl_alpha_{qt_key}")
        beta_max = max(0.0, round(1.0 - alpha, 2))
        beta_default = min(float(RDWA_BASE[qt_key]["beta"]), beta_max)
        beta = st.slider("β (Graph)", 0.0, beta_max, beta_default, 0.01,
                         key=f"sl_beta_{qt_key}")
        gamma = round(max(0.0, 1.0 - alpha - beta), 2)
        st.markdown(f"γ (Ontology) = **{gamma:.2f}** *(자동 계산)*")

        st.markdown("---")
        adj2 = rdwa_stage2(RDWA_BASE[qt_key], DENSITY_SIM[qt_key])
        st.markdown("**비교 기준점**")
        for name, a, b, g, color in [
            ("Uniform", 1/3, 1/3, 1/3, "#78909c"),
            ("R-DWA Base", RDWA_BASE[qt_key]["alpha"], RDWA_BASE[qt_key]["beta"], RDWA_BASE[qt_key]["gamma"], "#ff8f00"),
            ("R-DWA (보정)", adj2["alpha"], adj2["beta"], adj2["gamma"], "#ffb74d"),
            ("A-DWA", ADWA_WEIGHTS[qt_key]["alpha"], ADWA_WEIGHTS[qt_key]["beta"], ADWA_WEIGHTS[qt_key]["gamma"], "#66bb6a"),
        ]:
            st.markdown(
                f"<span style='color:{color};font-weight:700;'>{name}</span> "
                f"α={a:.2f} β={b:.2f} γ={g:.2f}",
                unsafe_allow_html=True,
            )

    with col_tern:
        def estimate_f1(a, b, g, qtype):
            optima = {
                "simple":      (0.62, 0.23, 0.15, 0.906),
                "multi-hop":   (0.14, 0.61, 0.25, 0.365),
                "conditional": (0.12, 0.23, 0.65, 0.304),
            }
            oa, ob, og, peak = optima[qtype]
            dist = np.sqrt((a-oa)**2 + (b-ob)**2 + (g-og)**2)
            return round(max(0.05, peak * np.exp(-2.5 * dist)), 3)

        user_f1 = estimate_f1(alpha, beta, gamma, qt_key)
        rdwa_f1 = estimate_f1(adj2["alpha"], adj2["beta"], adj2["gamma"], qt_key)
        adwa_f1 = estimate_f1(ADWA_WEIGHTS[qt_key]["alpha"], ADWA_WEIGHTS[qt_key]["beta"], ADWA_WEIGHTS[qt_key]["gamma"], qt_key)

        pts_a, pts_b, pts_c, pts_f = [], [], [], []
        for ia in range(21):
            for ib in range(21-ia):
                ic = 20 - ia - ib
                a_, b_, c_ = ia/20, ib/20, ic/20
                pts_a.append(c_); pts_b.append(b_); pts_c.append(a_)
                pts_f.append(estimate_f1(a_, b_, c_, qt_key))

        fig2 = go.Figure()
        fig2.add_trace(go.Scatterternary(
            a=pts_a, b=pts_b, c=pts_c, mode="markers",
            marker=dict(size=9, color=pts_f, colorscale="Viridis", showscale=True,
                        colorbar=dict(title="F1 추정", x=1.02),
                        cmin=0.05, cmax=max(pts_f)+0.02),
            hovertemplate="α=%{c:.2f} β=%{b:.2f} γ=%{a:.2f}<br>F1≈%{marker.color:.3f}<extra></extra>",
        ))
        lw2 = ADWA_WEIGHTS[qt_key]
        _ref4 = [
            (1/3,           1/3,            1/3),
            (adj2["gamma"], adj2["beta"],   adj2["alpha"]),
            (lw2["gamma"],  lw2["beta"],    lw2["alpha"]),
            (gamma,         beta,           alpha),
        ]
        fig2.add_trace(go.Scatterternary(
            a=[p[0] for p in _ref4],
            b=[p[1] for p in _ref4],
            c=[p[2] for p in _ref4],
            mode="markers",
            hovertext=[
                f"Uniform<br>α=0.333  β=0.333  γ=0.333<br>F1≈{estimate_f1(1/3,1/3,1/3,qt_key):.3f}",
                f"R-DWA (보정)<br>α={adj2['alpha']:.3f}  β={adj2['beta']:.3f}  γ={adj2['gamma']:.3f}<br>F1≈{rdwa_f1:.3f}",
                f"A-DWA (PPO학습)<br>α={lw2['alpha']:.3f}  β={lw2['beta']:.3f}  γ={lw2['gamma']:.3f}<br>F1≈{adwa_f1:.3f}",
                f"현재 슬라이더<br>α={alpha:.3f}  β={beta:.3f}  γ={gamma:.3f}<br>F1≈{user_f1:.3f}",
            ],
            hoverinfo="text",
            marker=dict(
                size=[12, 16, 22, 18],
                color=["#78909c", "#ffb74d", "#66bb6a", "#f44336"],
                symbol=["circle", "diamond", "star", "x"],
                line=dict(color="white", width=1.5),
            ),
        ))
        fig2.update_layout(
            ternary=dict(sum=1,
                aaxis=dict(title="γ Ontology", color="#ef9a9a"),
                baxis=dict(title="β Graph",    color="#ce93d8"),
                caxis=dict(title="α Vector",   color="#4fc3f7"),
                bgcolor=DARK_BG),
            paper_bgcolor=DARK_BG, font_color="#cfd8dc",
            title=f"{qt} 질의 — F1 추정 히트맵 (밝을수록 F1 높음)",
            height=480, margin=dict(t=60, b=70, l=80, r=120), showlegend=False,
        )
        st.plotly_chart(fig2, key="weight_explorer", use_container_width=True)
        # 기준점 수치 표 (Uniform → R-DWA → A-DWA → 현재 순서)
        st.markdown(f"""
        <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin-top:6px;">
          <div style="background:#0d1b2a;border:1px solid #78909c;border-radius:8px;padding:8px 12px;font-size:0.88rem;line-height:1.6;">
            <span style="color:#78909c;font-weight:700;">● Uniform</span><br>
            α=0.333 β=0.333 γ=0.333<br>
            <b style="color:#78909c;">F1≈{estimate_f1(1/3,1/3,1/3,qt_key):.3f}</b>
          </div>
          <div style="background:#1a1200;border:1px solid #ffb74d;border-radius:8px;padding:8px 12px;font-size:0.88rem;line-height:1.6;">
            <span style="color:#ffb74d;font-weight:700;">◆ R-DWA</span><br>
            α={adj2['alpha']:.3f} β={adj2['beta']:.3f} γ={adj2['gamma']:.3f}<br>
            <b style="color:#ffb74d;">F1≈{rdwa_f1:.3f}</b>
          </div>
          <div style="background:#001a09;border:1px solid #66bb6a;border-radius:8px;padding:8px 12px;font-size:0.88rem;line-height:1.6;">
            <span style="color:#66bb6a;font-weight:700;">★ A-DWA</span><br>
            α={lw2['alpha']:.3f} β={lw2['beta']:.3f} γ={lw2['gamma']:.3f}<br>
            <b style="color:#66bb6a;">F1≈{adwa_f1:.3f}</b>
          </div>
          <div style="background:#1a0000;border:1px solid #f44336;border-radius:8px;padding:8px 12px;font-size:0.88rem;line-height:1.6;">
            <span style="color:#f44336;font-weight:700;">✕ 현재</span><br>
            α={alpha:.3f} β={beta:.3f} γ={gamma:.3f}<br>
            <b style="color:#f44336;">F1≈{user_f1:.3f}</b>
          </div>
        </div>
        """, unsafe_allow_html=True)

    # ── 예상 F1 비교 게이지 ──────────────────────────────────────
    unif_f1 = estimate_f1(1/3, 1/3, 1/3, qt_key)
    st.markdown(f"""
    <div style="background:#071520;border:1px solid #1e3a55;border-radius:10px;padding:0.8rem 1.2rem;margin:0.5rem 0 0.8rem 0;">
      <div style="font-size:1rem;font-weight:700;color:#4fc3f7;margin-bottom:4px;">📊 가중치별 예상 F1_strict 점수 비교</div>
      <div style="font-size:0.92rem;color:#90a4ae;line-height:1.6;">
        현재 슬라이더 가중치 조합이 <b style="color:white;">{qt}</b> 질의에서 얼마나 좋은 성능을 낼지
        A-DWA 학습 결과로부터 역산한 <b style="color:#ffd54f;">예상 F1_strict</b>입니다.
        게이지 바늘이 오른쪽(0.6~0.75)에 가까울수록 더 좋은 가중치 조합입니다.
        F1_strict는 0~1 범위로, <b style="color:#66bb6a;">0.5 이상</b>이면 우수한 수준입니다.
      </div>
    </div>
    """, unsafe_allow_html=True)

    g1, g2, g3, g4 = st.columns(4)
    gauge_items = [
        ("Uniform\n(균등 배분)", unif_f1, "#78909c", "gauge_unif",
         "α=0.33 β=0.33 γ=0.33", "아무 판단 없이 1/3씩 배분한 경우"),
        ("R-DWA\n(2단계 보정)", rdwa_f1, "#ffb74d", "gauge_rdwa",
         f"α={adj2['alpha']:.2f} β={adj2['beta']:.2f} γ={adj2['gamma']:.2f}", "규칙 기반 알고리즘 기준선"),
        ("A-DWA\n(PPO 학습)", adwa_f1, "#66bb6a", "gauge_adwa",
         f"α={ADWA_WEIGHTS[qt_key]['alpha']:.2f} β={ADWA_WEIGHTS[qt_key]['beta']:.2f} γ={ADWA_WEIGHTS[qt_key]['gamma']:.2f}", "박사논문 제안 알고리즘 (최적)"),
        ("현재 슬라이더\n가중치", user_f1, "#f44336", "gauge_user",
         f"α={alpha:.2f} β={beta:.2f} γ={gamma:.2f}", "직접 조정한 가중치의 예상 점수"),
    ]
    for col, (lbl, val, color, gkey, weights_str, desc) in zip([g1, g2, g3, g4], gauge_items):
        with col:
            delta_vs_rdwa = val - rdwa_f1
            delta_str = f"+{delta_vs_rdwa:.3f}" if delta_vs_rdwa >= 0 else f"{delta_vs_rdwa:.3f}"
            delta_color = "#66bb6a" if delta_vs_rdwa > 0.001 else ("#ef5350" if delta_vs_rdwa < -0.001 else "#90a4ae")
            fig_g = go.Figure(go.Indicator(
                mode="gauge+number",
                value=val,
                gauge=dict(
                    axis=dict(range=[0, 0.75], tickfont=dict(size=10)),
                    bar=dict(color=color),
                    bgcolor="#1a2634", bordercolor="#2d4a6b",
                    steps=[
                        dict(range=[0, 0.35], color="#0d1b2a"),
                        dict(range=[0.35, 0.5], color="#0f2030"),
                        dict(range=[0.5, 0.75], color="#0d2010"),
                    ],
                    threshold=dict(line=dict(color="#4fc3f7", width=2), thickness=0.75, value=0.529),
                ),
                number=dict(font=dict(color=color, size=32), suffix=""),
                title=dict(text=lbl.replace("\n", "<br>"), font=dict(color="#cfd8dc", size=13)),
            ))
            fig_g.update_layout(
                height=230, margin=dict(t=50, b=10, l=10, r=10),
                paper_bgcolor=PAPER_BG,
            )
            st.plotly_chart(fig_g, key=gkey, use_container_width=True)
            st.markdown(
                f"<div style='text-align:center;'>"
                f"<span style='font-size:0.9rem;color:#cfd8dc;font-weight:700;'>{weights_str}</span><br>"
                f"<span style='font-size:0.88rem;color:{delta_color};font-weight:700;'>R-DWA 대비 {delta_str}</span><br>"
                f"<span style='font-size:0.82rem;color:#78909c;'>{desc}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

# ══════════════════════════════════════════════════════════════
# TAB 3 – Results
# ══════════════════════════════════════════════════════════════
with tab3:
    st.subheader("실험 결과 대시보드")
    st.caption("5,000 QA 합성 대학 벤치마크 · List Prompt · corrected baseline (논문 Table VI-4)")

    sub1, sub2 = st.tabs(["전체 성능 비교", "PPO 학습 동역학"])

    with sub1:
        cl, cr = st.columns(2, gap="large")
        with cl:
            st.markdown("**전체 정책 비교 (F1_strict)**")
            bar_colors = ["#546e7a","#ffb74d","#66bb6a","#ef9a9a"]
            fig_bar = go.Figure(go.Bar(
                x=PERF_OVERALL["정책"], y=PERF_OVERALL["F1_strict"],
                marker_color=bar_colors,
                text=[f"{v:.3f}" for v in PERF_OVERALL["F1_strict"]],
                textposition="outside",
            ))
            fig_bar.add_hline(y=0.554, line_dash="dash", line_color="#ef9a9a",
                              annotation_text="Oracle 0.554", annotation_position="top left",
                              annotation_font_color="#ef9a9a")
            fig_bar.update_layout(
                yaxis=dict(range=[0.30,0.68], title="F1_strict"),
                paper_bgcolor=DARK_BG, plot_bgcolor=DARK_BG,
                font_color="#cfd8dc", height=300, margin=dict(t=20,b=20,l=20,r=20),
            )
            st.plotly_chart(fig_bar, key="overall_bar", use_container_width=True)
            st.caption("A-DWA(0.562)가 R-DWA(0.529) 대비 F1_strict +6.2% 향상. Discrete Oracle(0.554)을 소폭 상회하며, Vector-only(0.334) 대비 +68.3% 개선.")

            st.markdown("**지표별 레이더 (R-DWA vs A-DWA)**")
            metric_cols = ["F1_strict","F1_substring","EM","Faithfulness"]
            fig_radar = go.Figure()
            for _, row in PERF_OVERALL[PERF_OVERALL["정책"].isin(["R-DWA","A-DWA"])].iterrows():
                is_rdwa = (row["정책"] == "R-DWA")
                color = "#ffb74d" if is_rdwa else "#66bb6a"
                fig_radar.add_trace(go.Scatterpolar(
                    r=[row[m] for m in metric_cols] + [row[metric_cols[0]]],
                    theta=metric_cols + [metric_cols[0]],
                    fill="toself", name=row["정책"],
                    line_color=color, fillcolor=hex_to_rgba(color, 0.13),
                ))
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0.3,0.65])),
                paper_bgcolor=DARK_BG, plot_bgcolor=DARK_BG,
                font_color="#cfd8dc", height=280,
                margin=dict(t=20,b=20,l=20,r=20), legend=dict(x=0.7,y=1),
            )
            st.plotly_chart(fig_radar, key="radar", use_container_width=True)
            st.caption("F1_strict·F1_substring·EM·Faithfulness 4개 지표 전반에서 A-DWA(초록)가 R-DWA(주황)를 상회. 가장 큰 향상은 Faithfulness(0.544→0.580). EM은 0.387→0.388로 사실상 동일 — 다중 정답 리스트 특성상 EM은 구조적으로 낮음.")

        with cr:
            st.markdown("**질의 유형별 F1_strict**")
            fig_type = px.bar(
                PERF_BY_TYPE, x="Query Type", y="F1_strict", color="Policy",
                barmode="group",
                color_discrete_map={"R-DWA":"#ffb74d","A-DWA":"#66bb6a"},
                text="F1_strict",
            )
            fig_type.update_traces(
                texttemplate="%{text:.3f}",
                textposition="outside",
            )
            fig_type.update_layout(
                paper_bgcolor=DARK_BG, plot_bgcolor=DARK_BG, font_color="#cfd8dc",
                height=280, margin=dict(t=20,b=20,l=20,r=20),
                yaxis=dict(title="F1_strict", range=[0,1.0]),
                legend=dict(x=0.7,y=1),
            )
            st.plotly_chart(fig_type, key="type_bar", use_container_width=True)
            st.caption("Conditional 질의에서 R-DWA(0.223) vs A-DWA(0.304)로 +36.7% 격차(절대 +0.081) — 수치·논리 제약이 복합된 질의는 규칙 테이블이 대응 불가. Simple 질의는 두 알고리즘 모두 0.87~0.91 수준으로 유사.")

            st.markdown("**핵심 결과 테이블 (Table VI-4)**")
            disp = PERF_OVERALL.copy()
            def vs_rdwa(x):
                if abs(x - 0.529) < 0.001: return "기준"
                return f"+{(x-0.529)/0.529*100:.1f}%" if x > 0.529 else f"{(x-0.529)/0.529*100:.1f}%"
            disp["vs R-DWA"] = disp["F1_strict"].apply(vs_rdwa)
            disp_show = disp[["정책","F1_strict","F1_substring","EM","Faithfulness","vs R-DWA"]]
            def _row_style(row):
                p = row["정책"]
                if p == "A-DWA":
                    return ["background-color:#1b3a1f;color:#a5d6a7;font-weight:700"] * len(row)
                elif p == "R-DWA":
                    return ["background-color:#2a1f00;color:#ffe082"] * len(row)
                elif p == "Discrete Oracle":
                    return ["background-color:#1a1a2e;color:#ef9a9a"] * len(row)
                elif p == "Vector-only":
                    return ["color:#78909c"] * len(row)
                return [""] * len(row)
            st.dataframe(
                disp_show.style.apply(_row_style, axis=1),
                hide_index=True, height=210,
            )
            st.caption("vs R-DWA 기준: A-DWA +6.2%, Discrete Oracle +4.7%. A-DWA가 Discrete Oracle(66개 이산 격자 참조값)을 소폭 초과한 것은 연속 가중치 공간에서 더 유리한 조합을 찾았기 때문 — 절대적 상한 초과를 뜻하지는 않음.")

            st.markdown("**Conditional 질의 — A-DWA가 Discrete Oracle 초과**")
            cond_df = pd.DataFrame({
                "정책": ["R-DWA","A-DWA","Discrete Oracle"],
                "F1_strict": [0.223, 0.304, 0.290],
            })
            fig_cond = go.Figure(go.Bar(
                x=cond_df["정책"], y=cond_df["F1_strict"],
                marker_color=["#ffb74d","#66bb6a","#ef9a9a"],
                text=["0.223","0.304 (+36.7%)","0.290"], textposition="outside",
            ))
            fig_cond.update_layout(
                paper_bgcolor=DARK_BG, plot_bgcolor=DARK_BG, font_color="#cfd8dc",
                height=220, margin=dict(t=10,b=20,l=20,r=20),
                yaxis=dict(range=[0,0.45], title="F1_strict"),
            )
            st.plotly_chart(fig_cond, key="cond_bar", use_container_width=True)
            st.caption("Conditional 집계(n=1,250)에서 A-DWA(0.304)가 Discrete Oracle(0.290)을 초과. 연속 가중치가 66개 이산 격자에 없는 중간 조합을 활용한 결과 — 본 연구에서 가장 뚜렷한 유형별 개선 구간(논문 Table A-4).")

    with sub2:
        st.markdown("**PPO 학습 곡선 (3-seed: 42 / 123 / 999)**")
        st.caption("3개 고정 시드 독립 학습 · F1_strict std < 0.007 · 약 150스텝에서 안정적 수렴")
        episodes = np.arange(200)
        fig_ppo = go.Figure()
        seed_colors = {42:"#66bb6a", 123:"#4fc3f7", 999:"#ce93d8"}
        all_vals = []
        for seed, vals in PPO_SEEDS.items():
            all_vals.append(vals)
            fig_ppo.add_trace(go.Scatter(
                x=episodes, y=vals, mode="lines",
                name=f"seed {seed}", line=dict(color=seed_colors[seed], width=1.5),
            ))
        mean_vals = np.mean(all_vals, axis=0)
        fig_ppo.add_trace(go.Scatter(
            x=episodes, y=mean_vals, mode="lines",
            name="3-seed 평균", line=dict(color="white", width=2.5, dash="dot"),
        ))
        for yv, lbl, col in [(0.529,"R-DWA 0.529","#ffb74d"),(0.554,"Oracle 0.554","#ef9a9a")]:
            fig_ppo.add_hline(y=yv, line_dash="dash", line_color=col,
                              annotation_text=lbl, annotation_position="right",
                              annotation_font_color=col)
        fig_ppo.update_layout(
            paper_bgcolor=DARK_BG, plot_bgcolor=DARK_BG, font_color="#cfd8dc",
            xaxis=dict(title="PPO 업데이트 스텝"),
            yaxis=dict(title="F1_strict", range=[0.34,0.60]),
            height=380, margin=dict(t=20,b=40,l=40,r=80),
            legend=dict(x=0.01,y=0.99),
        )
        st.plotly_chart(fig_ppo, key="ppo_curves", use_container_width=True)
        st.caption("초기 정책(~0.375)에서 출발해 약 150스텝에서 R-DWA 기준선(0.529)을 돌파, 이후 0.562 수준에서 안정 수렴. 3개 독립 시드(42·123·999) 모두 후기 보상이 초기보다 증가(+0.0157~+0.0183)하며 유사 수렴 — 재현성 확인.")
        st.markdown(f"""
        <div style="background:#1a1a2e;border-left:3px solid #ef9a9a;border-radius:6px;
                    padding:12px 16px;margin-top:4px;font-size:0.88rem;color:#cfd8dc;line-height:1.7;">
          <b style="color:#ffb74d;">📋 R-DWA란?</b> — R-DWA는 학습 없이 <b>룩업테이블</b>로
          질의 유형(Simple/Multi-hop/Conditional)별 기본 가중치를 선택하고,
          단일 λ=0.3 보정으로 관계·제약 밀도(s_r, s_c)를 반영해 정규화하는 2단계 규칙 기반 방식입니다(F1_strict 0.529).
          PPO 학습 곡선의 주황 점선(R-DWA 0.529)이 이 기준선을 나타냅니다.<br><br>
          <b style="color:#ef9a9a;">📌 Oracle이란?</b> — 실험에서 각 질의마다 <b>정답을 미리 알고</b>
          최적 가중치(α,β,γ)를 직접 선택했을 때의 성능 상한선입니다.
          현실에서는 정답을 알 수 없으므로 달성 불가능한 이론적 천장값(F1_strict = 0.554)입니다.<br><br>
          <b style="color:#66bb6a;">✅ 주목할 점</b> — 3-seed 평균(흰 점선)이 수렴 후 <b>Discrete Oracle 0.554를 소폭 상회</b>합니다.
          이는 A-DWA가 66개 이산 격자에 없는 연속 가중치 조합을 학습함으로써,
          이산 격자 참조값보다 더 유리한 조합을 찾았음을 의미합니다 (연속 공간 전체의 이론적 상한 초과는 아님).
        </div>""", unsafe_allow_html=True)

        pc1, pc2, pc3 = st.columns(3)
        for col, (lbl, val, color) in zip([pc1,pc2,pc3], [
            ("초기 정책 성능",          "~0.375",        "#78909c"),
            ("수렴 평균 (3-seed avg)", "0.562 ± 0.007", "#66bb6a"),
            ("수렴 스텝",             "~150 steps",    "#4fc3f7"),
        ]):
            with col:
                st.markdown(f"""
                <div class="metric-card">
                  <div class="val" style="font-size:1.4rem;color:{color};">{val}</div>
                  <div class="lbl">{lbl}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**PPO 학습 환경 (MDP 5-tuple)**")
        st.dataframe(pd.DataFrame({
            "구성 요소": ["State (s)", "Action (a)", "Reward R(s,a)", "Policy π_θ", "알고리즘"],
            "정의": [
                "18-dim 벡터 [밀도 ρ 3 + 질의유형 ℓ 3 + 소스통계 σ 9 + 질의메타 μ 3]",
                "3-simplex Δ³ 위의 연속 가중치 (α,β,γ), α+β+γ=1 (Dirichlet 정책)",
                "0.5·F1_strict + 0.3·EM + 0.2·Faithfulness − 0.1·max(0,ℓ−5)  (오프라인 캐시, LLM 호출 0회)",
                "Actor-Critic (5,636파라미터: 18→64→64 Tanh →[Actor 64→3 Softplus|Critic 64→1])",
                "PPO Clip ε=0.2, GAE λ=0.95, lr=3e-4, 시드 {42,123,999}",
            ],
        }), hide_index=True)

# ══════════════════════════════════════════════════════════════
# TAB 4 – Architecture
# ══════════════════════════════════════════════════════════════
with tab4:
    st.subheader("Triple-Hybrid RAG 아키텍처")

    components.html("""<!DOCTYPE html><html><head><meta charset="utf-8">
<style>body{margin:0;background:#0d1117;}</style></head><body>
<div style="background:#0d1117;border-radius:12px;padding:20px 12px 12px 12px;overflow-x:auto;">
<svg viewBox="0 0 860 690" xmlns="http://www.w3.org/2000/svg"
     style="width:100%;max-width:860px;display:block;margin:0 auto;font-family:'Segoe UI',Arial,sans-serif;">
  <defs>
    <marker id="arr" markerWidth="9" markerHeight="7" refX="8" refY="3.5" orient="auto">
      <polygon points="0 0, 9 3.5, 0 7" fill="#4fc3f7"/>
    </marker>
    <marker id="arr_g" markerWidth="9" markerHeight="7" refX="8" refY="3.5" orient="auto">
      <polygon points="0 0, 9 3.5, 0 7" fill="#66bb6a"/>
    </marker>
    <marker id="arr_p" markerWidth="9" markerHeight="7" refX="8" refY="3.5" orient="auto">
      <polygon points="0 0, 9 3.5, 0 7" fill="#9575cd"/>
    </marker>
    <marker id="arr_r" markerWidth="9" markerHeight="7" refX="8" refY="3.5" orient="auto">
      <polygon points="0 0, 9 3.5, 0 7" fill="#ef5350"/>
    </marker>
  </defs>

  <!-- ── 화살표 ── -->
  <line x1="420" y1="76" x2="430" y2="119" stroke="#4fc3f7" stroke-width="2" marker-end="url(#arr)"/>
  <line x1="430" y1="171" x2="430" y2="206" stroke="#4fc3f7" stroke-width="2" marker-end="url(#arr)"/>
  <line x1="303" y1="258" x2="152" y2="379" stroke="#66bb6a" stroke-width="2" stroke-dasharray="6,3" marker-end="url(#arr_g)"/>
  <line x1="430" y1="308" x2="430" y2="379" stroke="#9575cd" stroke-width="2" stroke-dasharray="6,3" marker-end="url(#arr_p)"/>
  <line x1="557" y1="258" x2="708" y2="379" stroke="#ef5350" stroke-width="2" stroke-dasharray="6,3" marker-end="url(#arr_r)"/>
  <line x1="140" y1="463" x2="318" y2="503" stroke="#4fc3f7" stroke-width="2" marker-end="url(#arr)"/>
  <line x1="430" y1="463" x2="430" y2="503" stroke="#4fc3f7" stroke-width="2" marker-end="url(#arr)"/>
  <line x1="720" y1="463" x2="542" y2="503" stroke="#4fc3f7" stroke-width="2" marker-end="url(#arr)"/>
  <line x1="430" y1="557" x2="430" y2="607" stroke="#4fc3f7" stroke-width="2.5" marker-end="url(#arr)"/>

  <!-- 가중치 레이블 -->
  <rect x="196" y="308" width="28" height="18" rx="4" fill="#1b3a1f"/>
  <text x="210" y="321" fill="#a5d6a7" font-size="13" font-weight="800" text-anchor="middle">α</text>
  <rect x="436" y="346" width="28" height="18" rx="4" fill="#1a1a3a"/>
  <text x="450" y="359" fill="#d1c4e9" font-size="13" font-weight="800" text-anchor="middle">β</text>
  <rect x="624" y="308" width="28" height="18" rx="4" fill="#3a0e0e"/>
  <text x="638" y="321" fill="#ffcdd2" font-size="13" font-weight="800" text-anchor="middle">γ</text>

  <!-- NODE 1: User Query -->
  <polygon points="308,30 552,30 532,76 288,76" fill="#0d47a1" stroke="#42a5f5" stroke-width="2"/>
  <text x="420" y="50" fill="white" font-size="15" font-weight="700" text-anchor="middle">User Query</text>
  <text x="412" y="67" fill="#90caf9" font-size="11.5" text-anchor="middle">자연어 입력 (Natural Language)</text>

  <!-- NODE 2: QueryAnalyzer -->
  <rect x="290" y="120" width="280" height="52" rx="8" fill="#004d40" stroke="#26a69a" stroke-width="2"/>
  <text x="430" y="142" fill="white" font-size="14" font-weight="700" text-anchor="middle">QueryAnalyzer</text>
  <text x="430" y="160" fill="#80cbc4" font-size="11" text-anchor="middle">의도분석 · NER · 18-dim 상태벡터 산출</text>

  <!-- NODE 3: DWA -->
  <polygon points="430,206 562,258 430,310 298,258" fill="#1a0038" stroke="#ce93d8" stroke-width="2"/>
  <text x="430" y="250" fill="white" font-size="14" font-weight="700" text-anchor="middle">DWA</text>
  <text x="430" y="265" fill="#e1bee7" font-size="11" text-anchor="middle">R-DWA / A-DWA</text>
  <text x="430" y="279" fill="#9e9e9e" font-size="10" text-anchor="middle">가중치 결정 알고리즘</text>

  <!-- NODE 4: Vector Store (실린더) -->
  <rect x="68" y="395" width="144" height="52" fill="#1b5e20" stroke="#66bb6a" stroke-width="2"/>
  <ellipse cx="140" cy="395" rx="72" ry="16" fill="#2e7d32" stroke="#66bb6a" stroke-width="2"/>
  <ellipse cx="140" cy="447" rx="72" ry="16" fill="#1b5e20" stroke="#66bb6a" stroke-width="2"/>
  <text x="140" y="419" fill="white" font-size="12.5" font-weight="700" text-anchor="middle">Vector Store</text>
  <text x="140" y="434" fill="#a5d6a7" font-size="10.5" text-anchor="middle">FAISS · 1536-dim</text>

  <!-- NODE 5: Graph Store (실린더) -->
  <rect x="358" y="395" width="144" height="52" fill="#1a237e" stroke="#7986cb" stroke-width="2"/>
  <ellipse cx="430" cy="395" rx="72" ry="16" fill="#283593" stroke="#7986cb" stroke-width="2"/>
  <ellipse cx="430" cy="447" rx="72" ry="16" fill="#1a237e" stroke="#7986cb" stroke-width="2"/>
  <text x="430" y="419" fill="white" font-size="12.5" font-weight="700" text-anchor="middle">Graph Store</text>
  <text x="430" y="434" fill="#c5cae9" font-size="10.5" text-anchor="middle">NetworkX · BFS</text>

  <!-- NODE 6: Ontology (실린더) -->
  <rect x="648" y="395" width="144" height="52" fill="#7f0000" stroke="#ef5350" stroke-width="2"/>
  <ellipse cx="720" cy="395" rx="72" ry="16" fill="#b71c1c" stroke="#ef5350" stroke-width="2"/>
  <ellipse cx="720" cy="447" rx="72" ry="16" fill="#7f0000" stroke="#ef5350" stroke-width="2"/>
  <text x="720" y="419" fill="white" font-size="12.5" font-weight="700" text-anchor="middle">Ontology</text>
  <text x="720" y="434" fill="#ffcdd2" font-size="10.5" text-anchor="middle">OWL2 · HermiT</text>

  <!-- NODE 7: merge_contexts (육각형) -->
  <polygon points="296,530 350,505 510,505 564,530 510,557 350,557" fill="#bf360c" stroke="#ff8a65" stroke-width="2"/>
  <text x="430" y="527" fill="white" font-size="13.5" font-weight="700" text-anchor="middle">merge_contexts</text>
  <text x="430" y="545" fill="#ffe0b2" font-size="10.5" text-anchor="middle">Score Fusion: α·Sv + β·Sg + γ·So</text>

  <!-- NODE 8: Final Answer -->
  <polygon points="286,608 574,608 594,656 266,656" fill="#1c313a" stroke="#80deea" stroke-width="2"/>
  <text x="432" y="629" fill="white" font-size="14" font-weight="700" text-anchor="middle">LLM → Final Answer</text>
  <text x="422" y="646" fill="#80deea" font-size="11" text-anchor="middle">GPT-4o-mini · temperature=0 · List Prompt</text>

  <!-- 범례 -->
  <g transform="translate(12,672)">
    <polygon points="0,3 22,3 20,14 -2,14" fill="#0d47a1" stroke="#42a5f5" stroke-width="1"/>
    <text x="27" y="12" fill="#78909c" font-size="10">입력 / 출력</text>
    <rect x="115" y="3" width="22" height="11" rx="2" fill="#004d40" stroke="#26a69a" stroke-width="1"/>
    <text x="142" y="12" fill="#78909c" font-size="10">처리 / 분석</text>
    <polygon points="242,8 256,3 270,8 256,14" fill="#1a0038" stroke="#ce93d8" stroke-width="1"/>
    <text x="277" y="12" fill="#78909c" font-size="10">알고리즘 / 결정</text>
    <rect x="394" y="5" width="16" height="9" fill="#1b5e20" stroke="#66bb6a" stroke-width="1"/>
    <ellipse cx="402" cy="5" rx="8" ry="3" fill="#2e7d32" stroke="#66bb6a" stroke-width="1"/>
    <text x="416" y="12" fill="#78909c" font-size="10">지식 저장소</text>
    <polygon points="550,8 560,3 580,3 590,8 580,14 560,14" fill="#bf360c" stroke="#ff8a65" stroke-width="1"/>
    <text x="598" y="12" fill="#78909c" font-size="10">융합 처리</text>
  </g>
</svg>
</div>
</body></html>""", height=730, scrolling=False)

    st.markdown("---")
    arch_l, arch_r = st.columns(2)
    with arch_l:
        st.markdown("**핵심 컴포넌트 비교: R-DWA vs A-DWA**")
        st.caption("같은 밀도 신호를 받지만, 가중치를 결정하는 방식이 핵심 차이입니다.")
        st.markdown("""
<style>
.cmp-tbl{width:100%;border-collapse:collapse;font-size:0.84rem;}
.cmp-tbl th{padding:8px 10px;text-align:center;font-size:0.82rem;font-weight:700;}
.cmp-tbl td{padding:7px 10px;border-bottom:1px solid #1e1e2e;vertical-align:top;line-height:1.5;}
.cmp-tbl tr:last-child td{border-bottom:none;}
.cmp-item{color:#90a4ae;font-size:0.8rem;font-weight:600;}
.cmp-desc{color:#546e7a;font-size:0.74rem;margin-top:2px;}
.hl-g{color:#a5d6a7;font-weight:700;}
.hl-r{color:#ef9a9a;font-weight:700;}
.hl-o{color:#ffb74d;font-weight:700;}
.sub{color:#546e7a;font-size:0.75rem;}
</style>
<table class="cmp-tbl">
  <thead>
    <tr style="background:#1a1a2e;">
      <th style="text-align:left;color:#78909c;width:30%;">항목</th>
      <th style="color:#ffe082;background:#261c00;width:35%;">R-DWA<br><span style="font-weight:400;font-size:0.74rem;">규칙 기반 · 학습 없음</span></th>
      <th style="color:#a5d6a7;background:#0d2010;width:35%;">A-DWA (PPO)<br><span style="font-weight:400;font-size:0.74rem;">강화학습 · 자동 최적화</span></th>
    </tr>
  </thead>
  <tbody>
    <tr style="background:#111120;">
      <td><div class="cmp-item">밀도 신호 입력</div><div class="cmp-desc">가중치 결정에 쓰는 입력값</div></td>
      <td style="color:#cfd8dc;">(s_e, s_r, s_c)<br><span class="sub">3차원만 사용</span></td>
      <td><span class="hl-g">18-dim 상태 벡터</span><br><span class="sub">밀도3 + 유형3 + 소스통계9 + 메타3</span></td>
    </tr>
    <tr>
      <td><div class="cmp-item">가중치 결정</div><div class="cmp-desc">α·β·γ를 어떻게 정하는가</div></td>
      <td style="color:#cfd8dc;">룩업테이블 + λ 보정<br><span class="sub">전문가가 설계한 고정 규칙</span></td>
      <td><span class="hl-g">π_θ(s) Dirichlet 정책</span><br><span class="sub">AI가 보상을 받으며 스스로 학습</span></td>
    </tr>
    <tr style="background:#111120;">
      <td><div class="cmp-item">(α,β,γ) 예시</div><div class="cmp-desc">Conditional 질의 기준</div></td>
      <td style="color:#cfd8dc;">(0.18, 0.23, <span class="hl-o">0.59</span>)<br><span class="sub">base (0.2,0.2,0.6) → λ 보정·정규화</span></td>
      <td style="color:#cfd8dc;">(0.12, 0.23, <span class="hl-g">0.65</span>)<br><span class="sub">γ를 더 높게 — 온톨로지 강화</span></td>
    </tr>
    <tr style="background:#0d2010;outline:1px solid #2e7d32;">
      <td><div class="cmp-item">F1_strict 성능</div><div class="cmp-desc">↑ 높을수록 정답에 가까움</div></td>
      <td><span class="hl-o">0.529</span><span class="sub"> (기준)</span></td>
      <td><span class="hl-g" style="font-size:1rem;">0.562 ± 0.007</span><br><span style="color:#a5d6a7;font-size:0.78rem;">▲ +6.2% · Discrete Oracle(0.554) 소폭 상회</span></td>
    </tr>
    <tr>
      <td><div class="cmp-item">파라미터 수</div><div class="cmp-desc">학습 가능한 변수 수</div></td>
      <td><span class="hl-o">0</span><span class="sub"> (규칙 기반)</span></td>
      <td><span class="hl-g">5,636</span><br><span class="sub">18→64→64 Tanh→[Actor 3 Softplus|Critic 1]</span></td>
    </tr>
    <tr style="background:#111120;">
      <td><div class="cmp-item">도메인 이전성</div><div class="cmp-desc">다른 분야 적용 시</div></td>
      <td><span class="hl-r">λ 재튜닝 필요</span><br><span class="sub">규칙을 전문가가 다시 설계</span></td>
      <td style="color:#cfd8dc;">도메인 특화 재학습<br><span class="sub">데이터만 있으면 자동 적응</span></td>
    </tr>
  </tbody>
</table>
        """, unsafe_allow_html=True)

    with arch_r:
        st.markdown("**오프라인 보상 캐시 — 비용 절감 핵심**")
        st.caption("PPO 학습 중 매번 GPT-4o-mini를 호출하면 비용 폭발 → 미리 계산해 저장한 캐시로 89% 절감.")
        st.markdown("""
<table class="cmp-tbl">
  <thead>
    <tr style="background:#1a1a2e;">
      <th style="text-align:left;color:#78909c;width:40%;">항목</th>
      <th style="text-align:left;color:#ffe0b2;background:#1a1000;width:60%;">값 / 수치</th>
    </tr>
  </thead>
  <tbody>
    <tr style="background:#111120;">
      <td><div class="cmp-item">캐시 크기</div><div class="cmp-desc">미리 계산한 보상값 총 수</div></td>
      <td style="color:#cfd8dc;">5,000 QA × 66 이산 가중치<br><span class="hl-g">= 330,000 엔트리</span></td>
    </tr>
    <tr>
      <td><div class="cmp-item">저장소</div><div class="cmp-desc">DB 종류와 용량</div></td>
      <td style="color:#cfd8dc;">SQLite<br><span class="sub">16.8 MB — 별도 서버 불필요한 파일 DB</span></td>
    </tr>
    <tr style="background:#111120;">
      <td><div class="cmp-item">초기 구축 비용</div><div class="cmp-desc">캐시를 처음 만드는 비용 (1회)</div></td>
      <td style="color:#cfd8dc;">330,000건 1회 사전구축, <span class="hl-o">실측 $37.82</span><br><span class="sub">이후 3-seed 학습 전 과정에서 재사용</span></td>
    </tr>
    <tr style="background:#0d2010;">
      <td><div class="cmp-item">on-policy 대비 절감</div><div class="cmp-desc">캐시 미사용(약 $344) 대비</div></td>
      <td>
        <span class="hl-g" style="font-size:1.05rem;">약 89% 절감</span>
        <span class="sub"> $344 → $37.82</span><br>
        <div style="background:#0a2010;border-radius:4px;margin-top:5px;height:10px;width:100%;border:1px solid #2e7d32;">
          <div style="background:linear-gradient(90deg,#388e3c,#66bb6a);height:10px;border-radius:3px;width:89%;"></div>
        </div>
        <div style="display:flex;justify-content:space-between;margin-top:2px;">
          <span class="sub">$0</span><span style="color:#a5d6a7;font-size:0.74rem;font-weight:700;">89%</span><span class="sub">$344</span>
        </div>
      </td>
    </tr>
    <tr>
      <td><div class="cmp-item">PPO 학습 중 LLM 호출</div><div class="cmp-desc">학습 시 외부 API 호출 횟수</div></td>
      <td><span class="hl-g" style="font-size:1.05rem;">0회</span><br>
      <span class="sub">캐시 DB 조회만 — 학습 속도 획기적 단축</span></td>
    </tr>
  </tbody>
</table>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════
# SIDEBAR – 용어사전 (언제든 열 수 있는 슬라이딩 패널)
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 📚 핵심 용어사전")
    st.caption("비전공자도 이해할 수 있는 설명 + 비유 + 수식")

    TERMS = [
        # ── Tab 1: 파이프라인 시뮬레이터 ─────────────────────────────────
        {
            "tab":"Tab 1", "icon":"🤖", "name":"RAG", "eng":"Retrieval-Augmented Generation",
            "cat":"핵심 개념", "cat_css":"cat-core",
            "simple":"AI가 질문에 답할 때, 기억 속 지식만 쓰는 게 아니라 먼저 관련 문서를 검색해 가져온 뒤 그 내용을 근거로 답변을 생성하는 방식입니다. 없는 내용을 지어내는 '환각' 문제를 크게 줄입니다.",
            "analogy":"💡 비유: 시험 볼 때 암기한 내용만 쓰는 게 아니라, 교과서에서 관련 내용을 먼저 찾아 읽고 답을 쓰는 오픈북 방식과 같습니다.",
            "formula":"최종 답변 = LLM(Query + Retrieved Documents)",
        },
        {
            "tab":"Tab 1", "icon":"🔺", "name":"Triple-Hybrid RAG", "eng":"세 가지 지식 소스를 혼합한 RAG",
            "cat":"핵심 개념", "cat_css":"cat-core",
            "simple":"Vector(문서 유사도) + Graph(관계망 탐색) + Ontology(규칙·제약 추론) 세 가지 지식 소스를 동시에 활용하고, 가중치(α, β, γ)로 각 기여도를 동적으로 조절합니다. 본 논문의 핵심 아키텍처입니다.",
            "analogy":"💡 비유: 형사가 목격자 증언(Vector), 인맥 관계도(Graph), 법률 조문(Ontology)을 동시에 참고해 사건을 해결하는 것과 같습니다.",
            "formula":"S_total = α·S_vector + β·S_graph + γ·S_ontology, α+β+γ=1",
        },
        {
            "tab":"Tab 1", "icon":"❓", "name":"질의 유형", "eng":"Query Types (Simple / Multi-hop / Conditional)",
            "cat":"핵심 개념", "cat_css":"cat-core",
            "simple":"질문의 복잡도를 세 유형으로 분류합니다: Simple(단순 개체 속성 조회 → α↑), Multi-hop(다단계 관계 탐색 필요 → β↑), Conditional(수치·논리 제약 포함 → γ↑). 질의 유형이 가중치 결정에 가장 큰 영향을 줍니다.",
            "analogy":"💡 비유 — Simple: '서울의 인구는?', Multi-hop: '서울 본사 회사 중 창업 10년 이내는?', Conditional: '30대 이하 공학 전공 임원은?'",
            "formula":"Simple→(α=0.60,β=0.25,γ=0.15) | Multi-hop→(0.15,0.60,0.25) | Conditional→(0.10,0.20,0.70)",
        },
        {
            "tab":"Tab 1", "icon":"📡", "name":"밀도 신호", "eng":"Density Signals (s_e, s_r, s_c)",
            "cat":"핵심 개념", "cat_css":"cat-core",
            "simple":"질의에 '개체(Named Entity)', '관계(relation)', '수치 제약(constraint)'이 얼마나 풍부한지 0~1 수치로 표현한 것입니다. 이 세 값이 DWA 알고리즘의 핵심 입력이자 18-dim 상태벡터의 첫 3차원입니다.",
            "analogy":"💡 비유: 요리에 소금/설탕/식초가 얼마나 필요한지 농도를 미리 파악하고 조미하는 것처럼, 질의의 성분 농도로 가중치를 조절합니다.",
            "formula":"s_e:개체밀도, s_r:관계밀도, s_c:제약밀도 ∈ [0,1] | Simple→s_e↑, Multi-hop→s_r↑, Conditional→s_c↑",
        },
        {
            "tab":"Tab 1", "icon":"🔵", "name":"Vector 검색", "eng":"Semantic Search (FAISS)",
            "cat":"핵심 개념", "cat_css":"cat-core",
            "simple":"문장을 수천 차원 숫자(벡터)로 변환하고, 질문 벡터와 가장 유사한 문서 벡터를 찾아오는 방식입니다. 단어가 달라도 '의미'가 비슷하면 잘 찾아냅니다. Simple 질의에 효과적입니다.",
            "analogy":"💡 비유: '강아지'를 검색해도 '반려견'이 있는 문서를 찾아주는 의미 기반 검색입니다.",
            "formula":"유사도 = cos(q_vec, d_vec) | 모델: text-embedding-3-small 1536-dim",
        },
        {
            "tab":"Tab 1", "icon":"🟣", "name":"Graph 검색", "eng":"Knowledge Graph BFS (NetworkX)",
            "cat":"핵심 개념", "cat_css":"cat-core",
            "simple":"교수→과목, 교수→학과 등의 '관계'를 그물망처럼 연결해 저장하고, 여러 단계를 거쳐 연결된 정보를 찾아오는 방식입니다. Multi-hop 질의에 필수적입니다.",
            "analogy":"💡 비유: SNS에서 '내 친구의 친구 중 서울 거주자'를 찾듯, 관계를 타고 여러 홉을 거쳐 답을 찾습니다.",
            "formula":"BFS(시작 노드, max_depth=3) | 그래프: 577교수+60학과+1,505과목+400프로젝트",
        },
        {
            "tab":"Tab 1", "icon":"🔴", "name":"Ontology 검색", "eng":"OWL2 Reasoning (HermiT)",
            "cat":"핵심 개념", "cat_css":"cat-core",
            "simple":"'나이 ≤ 55', '부교수 이상' 같은 논리 규칙과 수치 제약을 자동 추론하는 지식 체계입니다. Conditional 질의에서 단순 키워드 매칭이 할 수 없는 필터링을 담당합니다.",
            "analogy":"💡 비유: 법률 조문처럼 '요건 A이면 결론 B'라는 규칙을 컴퓨터가 자동 적용하는 것입니다.",
            "formula":"Professor ⊑ Person, age(x) ≤ 55 → eligible(x) | 추론기: Owlready2 HermiT",
        },
        {
            "tab":"Tab 1", "icon":"📏", "name":"R-DWA", "eng":"Rule-based Dynamic Weighting Algorithm",
            "cat":"핵심 개념", "cat_css":"cat-core",
            "simple":"질의 유형에 따라 사람이 설계한 규칙으로 α,β,γ를 정하는 2단계 알고리즘입니다. Stage1: 질의 유형 룩업 → Stage2: 밀도 신호로 λ(람다) 보정 → 정규화. 본 논문의 비교 기준선(baseline)입니다.",
            "analogy":"💡 비유: 요리 레시피처럼 '조건부 질의엔 Ontology 70%'라는 규칙을 사람이 미리 작성한 것입니다.",
            "formula":"α'=α_base·(1−λ·(s_r+s_c)/2), β'=β_base+λ·s_r·(1−β_base), γ'=γ_base+λ·s_c·(1−γ_base), λ=0.3 → Normalize",
        },
        {
            "tab":"Tab 1", "icon":"🤖", "name":"A-DWA", "eng":"Adaptive Dynamic Weight Learning (PPO 기반 적응형 동적 가중치 학습)",
            "cat":"강화학습", "cat_css":"cat-rl",
            "simple":"규칙을 사람이 직접 짜는 R-DWA와 달리, PPO 강화학습으로 AI가 스스로 최적 가중치를 학습합니다. 본 논문의 핵심 기여로, R-DWA 대비 F1_strict +6.2%, Conditional 질의 +36.7% 향상을 달성합니다.",
            "analogy":"💡 비유: 레시피 없이 수천 번 맛을 보면서 스스로 최적의 간을 터득한 요리사와 같습니다.",
            "formula":"π_θ*(s) = argmax_θ E[R(s,a)] | 3-seed 평균 F1_strict: 0.562 ± 0.007",
        },
        {
            "tab":"Tab 1", "icon":"🔺", "name":"3-simplex Δ³", "eng":"Probability Simplex (가중치 공간)",
            "cat":"핵심 개념", "cat_css":"cat-core",
            "simple":"α+β+γ=1 제약을 만족하는 모든 가중치 조합을 나타내는 삼각형 수학 공간입니다. A-DWA의 출력은 이 삼각형 위의 한 점이며, 삼각형의 각 꼭짓점은 한 소스에 100% 집중하는 극단적 가중치를 나타냅니다.",
            "analogy":"💡 비유: 예산 100만원을 세 항목에 배분할 때 가능한 모든 비율 조합을 나타낸 삼각형 지도와 같습니다.",
            "formula":"Δ³ = {(α,β,γ)∈ℝ³ | α≥0, β≥0, γ≥0, α+β+γ=1}",
        },
        # ── Tab 3: 성능 분석 ──────────────────────────────────────────────
        {
            "tab":"Tab 3", "icon":"📊", "name":"F1_strict", "eng":"Token-set F1 (엄격한 정확도 지표)",
            "cat":"평가 지표", "cat_css":"cat-eval",
            "simple":"AI 답변이 정답과 얼마나 겹치는지 토큰(단어) 단위로 정밀 측정합니다. Precision(답변의 정확성)과 Recall(정답 포괄성)의 조화 평균입니다. 본 논문의 주 평가 지표입니다.",
            "analogy":"💡 비유: '홍성민 교수, 융합공학과'가 정답인데 AI가 '홍성민 교수, 전자공학과'라 답하면 절반만 점수를 받습니다.",
            "formula":"F1 = 2·P·R/(P+R) | P=공통토큰/예측길이, R=공통토큰/정답길이",
        },
        {
            "tab":"Tab 3", "icon":"🎯", "name":"EM", "eng":"Exact Match (완전 일치 정확도)",
            "cat":"평가 지표", "cat_css":"cat-eval",
            "simple":"AI 답변이 정답과 토큰 하나까지 완전히 일치할 때만 1점을 줍니다. F1_strict보다 훨씬 엄격하며, 단순 사실 조회(Simple 질의)에서 의미 있는 지표입니다. 부분 일치는 0점 처리됩니다.",
            "analogy":"💡 비유: 주관식 채점에서 '서울특별시'가 정답인데 '서울'이라고 쓰면 F1은 부분 점수지만 EM은 0점입니다. 정확한 표현까지 맞춰야 합니다.",
            "formula":"EM = 1 if normalize(pred)==normalize(gold) else 0 | A-DWA EM: 0.388 (R-DWA 0.387, 다중정답 리스트 특성상 EM은 구조적으로 낮음)",
        },
        {
            "tab":"Tab 3", "icon":"📏", "name":"F1_substring", "eng":"Substring F1 (부분 문자열 F1)",
            "cat":"평가 지표", "cat_css":"cat-eval",
            "simple":"F1_strict보다 완화된 지표로, 정답이 답변의 부분 문자열로 포함되어 있으면 점수를 부여합니다. 목록형 답변이나 긴 설명문처럼 정답 표현이 다양할 때 유용합니다. Multi-hop 질의 평가에 주로 활용됩니다.",
            "analogy":"💡 비유: 정답이 '홍성민'인데 AI가 '홍성민 교수님은 융합공학과 소속입니다'라고 답하면 F1_strict는 낮지만 F1_substring은 1.0입니다.",
            "formula":"Substring F1: 정답 토큰이 예측에 포함되면 Recall=1 | A-DWA F1_substring: 0.507 (+5.2% vs R-DWA 0.482)",
        },
        {
            "tab":"Tab 3", "icon":"✅", "name":"Faithfulness", "eng":"충실도 (환각 방지 지표)",
            "cat":"평가 지표", "cat_css":"cat-eval",
            "simple":"AI 답변이 검색된 문서 내용에 충실한지 측정합니다. 없는 내용을 지어내는 '환각(Hallucination)' 현상을 수치로 측정해 RAG 품질을 평가합니다.",
            "analogy":"💡 비유: 기자가 취재한 내용만 기사에 쓰는지(충실도 높음), 아니면 추측으로 지어내는지(충실도 낮음) 측정하는 것과 같습니다.",
            "formula":"2-branch 검증: 엔티티 추출 → 문서 커버리지 체크 → 점수 ∈ [0,1]",
        },
        {
            "tab":"Tab 3", "icon":"🧭", "name":"Discrete Oracle", "eng":"Discrete Oracle (이산 격자 참조 상한)",
            "cat":"핵심 개념", "cat_css":"cat-core",
            "simple":"각 질의마다 66개 이산 가중치 격자 중 최고 보상 조합을 사후에 선택한 참조 모델입니다. 연속 공간 전체의 이론적 상한이 아니라 이산 격자 내 참조값이며, F1_strict 0.554를 기록합니다. A-DWA(0.562)가 이를 소폭 상회한 것은 격자에 없는 중간 가중치 조합을 활용했기 때문입니다.",
            "analogy":"💡 비유: 객관식 66개 보기 중 정답을 미리 보고 최선을 고른 모범답안. A-DWA는 보기에 없는 연속값까지 탐색해 더 나은 조합을 찾습니다.",
            "formula":"Discrete Oracle F1_strict: 0.554 | A-DWA: 0.562 (+1.4%) | 66개 이산 격자 × 5,000 QA 사후 최적 선택",
        },
        # ── Tab 4: 아키텍처 (강화학습 내부 구조) ─────────────────────────
        {
            "tab":"Tab 4", "icon":"🗺️", "name":"MDP", "eng":"Markov Decision Process",
            "cat":"강화학습", "cat_css":"cat-rl",
            "simple":"AI 학습의 게임 규칙을 정의한 수학적 틀입니다. 현재 상태(State)를 보고 행동(Action)을 취하면 보상(Reward)을 받고 다음 상태로 이동하는 순환 구조입니다.",
            "analogy":"💡 비유: 체스처럼 '현재 판세(State)'를 보고 '수(Action)'를 두면 '승패점(Reward)'이 바뀌고 다음 판세가 펼쳐지는 것입니다.",
            "formula":"MDP=(S,A,R,π) | S:18-dim, A:Δ³, R=0.5·F1_strict+0.3·EM+0.2·Faith−0.1·max(0,ℓ−5)",
        },
        {
            "tab":"Tab 4", "icon":"🎮", "name":"PPO", "eng":"Proximal Policy Optimization",
            "cat":"강화학습", "cat_css":"cat-rl",
            "simple":"강화학습 알고리즘 중 하나로, '너무 급격하게 정책을 바꾸지 말라'는 클리핑(ε=0.2) 제약을 통해 안정적으로 학습합니다. OpenAI의 ChatGPT 학습에도 사용된 검증된 알고리즘입니다.",
            "analogy":"💡 비유: 주식을 조금씩 조정하며 리스크를 관리하는 투자자처럼, 급격한 변화 없이 점진적으로 최적해에 수렴합니다.",
            "formula":"L_CLIP(θ) = E[min(r_t·A_t, clip(r_t, 1-ε, 1+ε)·A_t)] | ε=0.2, GAE λ=0.95",
        },
        {
            "tab":"Tab 4", "icon":"🧠", "name":"Actor-Critic", "eng":"Actor-Critic Network (5,636 파라미터)",
            "cat":"강화학습", "cat_css":"cat-rl",
            "simple":"PPO 학습에 쓰이는 신경망으로 두 역할을 합니다. Actor(배우)는 현재 상태를 보고 가중치(행동)를 결정하고, Critic(평론가)은 그 행동이 얼마나 좋은지 가치를 평가합니다.",
            "analogy":"💡 비유: 배우가 대사를 치면 감독이 피드백을 주고, 배우는 그 피드백으로 다음 연기를 개선하는 협력 구조입니다.",
            "formula":"18 → Linear(64) → Tanh → Linear(64) → Tanh → [Actor 64→3 Softplus→Dirichlet | Critic 64→1]",
        },
        {
            "tab":"Tab 4", "icon":"➡️", "name":"Linear", "eng":"Linear Layer (선형 변환층, 완전 연결층)",
            "cat":"강화학습", "cat_css":"cat-rl",
            "simple":"신경망의 가장 기본 연산으로, 입력 벡터에 가중치 행렬 W와 편향 b를 곱해 새로운 공간으로 변환합니다. A-DWA에서는 공유층 18→64, 64→64와 Actor head 64→3, Critic head 64→1의 Linear 변환이 사용됩니다.",
            "analogy":"💡 비유: 악보(입력)를 다른 조성(출력)으로 전조(transpose)하는 것처럼, 입력 공간을 더 유용한 표현 공간으로 변환합니다.",
            "formula":"y = Wx + b | 공유: 18→64(1,216), 64→64(4,160) | Actor: 64→3(195) | Critic: 64→1(65) | 합 5,636",
        },
        {
            "tab":"Tab 4", "icon":"⚡", "name":"Tanh", "eng":"Tanh (하이퍼볼릭 탄젠트) — 활성화 함수",
            "cat":"강화학습", "cat_css":"cat-rl",
            "simple":"신경망에 비선형성을 부여하는 활성화 함수입니다. 입력을 −1~1 범위로 매끄럽게 압축합니다. A-DWA는 두 개의 공유 은닉층(18→64, 64→64) 출력에 Tanh를 적용해 안정적인 특징 표현을 학습합니다.",
            "analogy":"💡 비유: 볼륨 노브처럼, 입력이 커져도 −1~1 사이로 부드럽게 수렴시켜 값이 폭주하지 않게 합니다.",
            "formula":"tanh(x) = (eˣ−e⁻ˣ)/(eˣ+e⁻ˣ) | 출력 ∈ (−1,1) | 0 중심 대칭, 안정적 학습",
        },
        {
            "tab":"Tab 4", "icon":"🎯", "name":"Softplus", "eng":"Softplus — Dirichlet 농도 파라미터 생성 함수",
            "cat":"강화학습", "cat_css":"cat-rl",
            "simple":"A-DWA Actor head의 마지막 활성화 함수입니다. 출력을 항상 양수로 만들어 Dirichlet 분포의 농도(concentration) 파라미터 c=(c_α,c_β,c_γ)>0를 생성합니다. 이를 통해 α+β+γ=1, 모든 값 ≥0 조건이 자연스럽게 보장됩니다.",
            "analogy":"💡 비유: ReLU의 부드러운 버전으로, 음수를 0으로 끊지 않고 매끄럽게 양수로 변환해 항상 유효한 분포 파라미터를 만듭니다.",
            "formula":"softplus(x) = ln(1+eˣ) > 0 | c = softplus(Actor(s)) | a ~ Dirichlet(c), 추론 ā = c/‖c‖₁",
        },
        {
            "tab":"Tab 4", "icon":"📊", "name":"Dirichlet 평균", "eng":"Dirichlet Mean (가중치 분포의 기댓값)",
            "cat":"강화학습", "cat_css":"cat-rl",
            "simple":"Dirichlet 분포는 확률 벡터(합=1)를 생성하는 분포입니다. A-DWA의 Actor는 Softplus 출력을 Dirichlet 분포의 농도 파라미터(concentration) c로 해석합니다. 학습 시에는 분포에서 샘플링(a~Dir(c))하고, 추론 시에는 분포의 평균(기댓값)을 최종 가중치로 사용합니다.",
            "analogy":"💡 비유: '세 소스에서 얼마나 뽑을까'의 불확실성 전체를 분포로 표현한 뒤, 그 평균을 최종 가중치로 씁니다. 점 추정보다 더 안정적인 의사결정이 가능합니다.",
            "formula":"c = Softplus(Actor(s)) > 0 | a ~ Dir(c) (학습) | ā = c/‖c‖₁ (추론) = 최종 가중치",
        },
        {
            "tab":"Tab 4", "icon":"💾", "name":"오프라인 보상 캐시", "eng":"Offline Reward Cache (SQLite, 330K 엔트리)",
            "cat":"인프라", "cat_css":"cat-infra",
            "simple":"PPO 학습 중 매번 GPT-4o-mini를 호출하면 비용이 폭발합니다. 대신 5,000 QA × 66 가중치 조합의 보상값을 미리 계산해 SQLite DB에 저장하고, 학습 중엔 DB만 조회합니다. 비용 89% 절감 효과입니다.",
            "analogy":"💡 비유: 매번 식당에서 시식하며 메뉴를 고르는 대신, 모든 메뉴를 미리 맛보고 별점 표를 만들어 참고하는 것과 같습니다.",
            "formula":"5,000 QA × 66 이산점 = 330,000 엔트리 | SQLite | 실측 $37.82 (캐시 미사용 약 $344 대비 89% 절감)",
        },
    ]

    # 탭 구분선 레이블 정의
    _TAB_LABELS = {
        "Tab 1": "① 파이프라인 시뮬레이터",
        "Tab 3": "③ 성능 분석",
        "Tab 4": "④ 아키텍처",
    }

    cat_filter = st.radio(
        "카테고리",
        ["전체", "핵심 개념", "강화학습", "평가 지표", "인프라"],
        horizontal=False,
        key="cat_filter_radio",
    )
    filtered = TERMS if cat_filter == "전체" else [t for t in TERMS if t["cat"] == cat_filter]

    st.markdown("---")

    _prev_tab = None
    for term in filtered:
        _cur_tab = term.get("tab", "")
        if cat_filter == "전체" and _cur_tab != _prev_tab and _cur_tab in _TAB_LABELS:
            st.markdown(
                f"<div style='font-size:0.78rem;font-weight:700;letter-spacing:1.5px;"
                f"color:#607d8b;text-transform:uppercase;margin:14px 0 4px 2px;"
                f"border-left:3px solid #4fc3f7;padding-left:8px;'>"
                f"📌 {_TAB_LABELS[_cur_tab]}</div>",
                unsafe_allow_html=True,
            )
            _prev_tab = _cur_tab
        with st.expander(f"{term['icon']} {term['name']}", expanded=False, key=f"exp_{term['name']}"):
            formula_html = (
                f'<div class="eb-formula">📐 {term["formula"]}</div>'
                if term.get("formula") else ""
            )
            st.markdown(f"""
            <div class="explain-box">
              <div class="eb-title">{term['icon']} {term['name']}</div>
              <div style="font-size:0.82rem;color:#607d8b;margin-bottom:6px;">{term['eng']}</div>
              <span class="tc-cat {term['cat_css']}">{term['cat']}</span>
              <div class="eb-simple">{term['simple']}</div>
              <div class="eb-analogy">{term['analogy']}</div>
              {formula_html}
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**💡 용어 간 연관 관계**")
    st.dataframe(pd.DataFrame({
        "용어": ["Triple-Hybrid RAG", "R-DWA", "A-DWA", "오프라인 보상 캐시"],
        "관계": ["구성 ←", "입력 ←", "학습 알고리즘", "보상 공급 →"],
        "연결 대상": ["Vector + Graph + Ontology", "밀도 신호", "PPO / MDP", "A-DWA PPO"],
    }), hide_index=True, use_container_width=True)

# ── 푸터 ──────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#546e7a;font-size:0.85rem;'>"
    "신동욱 박사학위논문 · 호서대학교 벤처대학원 · 지도교수 문남미 · 2026 &nbsp;|&nbsp; "
    "<a href='https://github.com/sdw1621/triple-rag-phd' style='color:#4fc3f7;'>GitHub</a>"
    "</div>",
    unsafe_allow_html=True,
)
