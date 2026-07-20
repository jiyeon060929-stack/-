import sys
import os
from pathlib import Path
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.font_manager as fm
import streamlit as st

# =========================================================
# 카카오톡 / SNS 링크 공유 미리보기(Open Graph) 설정
# =========================================================
st.set_page_config(
    page_title="AI 기반 아동 미디어 코칭 대시보드",
    page_icon="📊",
    layout="wide"
)

# 링크 카카오톡 공유 시 제목 및 설명 세팅
st.markdown(
    """
    <head>
        <meta property="og:title" content="AI 기반 아동 미디어 코칭 대시보드" />
        <meta property="og:description" content="데이터 기반 학부모 미디어 통제 시뮬레이션 및 맞춤형 AI 코칭 서비스" />
        <meta property="og:type" content="website" />
    </head>
    """,
    unsafe_allow_html=True
)

# =========================================================
# 0. 한글 폰트 감지 및 Matplotlib 깨짐 방지
# =========================================================
def init_korean_font():
    font_candidates = ["NanumGothic", "Malgun Gothic", "AppleGothic", "Noto Sans CJK KR"]
    available_fonts = {f.name for f in fm.fontManager.ttflist}
    
    chosen_font = next((f for f in font_candidates if f in available_fonts), None)
    
    if not chosen_font:
        linux_paths = [
            "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
            "/usr/share/fonts/nanum/NanumGothic.ttf"
        ]
        for path in linux_paths:
            if os.path.exists(path):
                fm.fontManager.addfont(path)
                chosen_font = fm.FontProperties(fname=path).get_name()
                break

    if chosen_font:
        matplotlib.rcParams["font.family"] = chosen_font
    matplotlib.rcParams["axes.unicode_minus"] = False

init_korean_font()

sys.path.insert(0, str(Path(__file__).resolve().parent / "model"))
import predictor

# =========================================================
# 1. 화면 설정 및 CSS
# =========================================================
st.set_page_config(
    page_title="데이터 기반 학부모 미디어 통제 시뮬레이션 및 AI 코칭",
    page_icon="📊",
    layout="wide"
)

st.markdown(
    """
    <style>
    /* 전체 배경색 깔끔한 연회색으로 변경 */
    .stApp {
        background-color: #F8FAFC;
    }
    
    /* 상단 블루 그라데이션 메인 배너 */
    .hero-banner {
        background: linear-gradient(135deg, #1E3A8A 0%, #3B82F6 100%);
        padding: 28px 36px;
        border-radius: 16px;
        color: white;
        margin-bottom: 25px;
        box-shadow: 0 10px 20px rgba(30, 58, 138, 0.12);
    }
    .hero-banner h1 {
        color: #FFFFFF !important;
        font-size: 2.1rem !important;
        font-weight: 800 !important;
        margin-bottom: 6px !important;
    }
    .hero-banner p {
        font-size: 1.05rem;
        color: #E0E7FF;
        margin-bottom: 0;
    }

    /* 카드 입체감 (Box Shadow & 모서리 라운딩) */
    .stContainer, div[data-testid="stForm"] {
        background-color: #FFFFFF;
        border-radius: 14px !important;
        padding: 20px !important;
        border: 1px solid #E2E8F0 !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.04) !important;
    }

    /* 주요 섹션 제목 스타일 */
    h2 {
        font-size: 1.5rem !important;
        color: #0F172A !important;
        font-weight: 700 !important;
        padding-bottom: 8px;
    }

    /* 숫자 강조 Metric 스타일 */
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
        font-weight: 800 !important;
        color: #1E293B;
    }
    
    /* 제출 및 실행 버튼 커스텀 */
    .stButton > button {
        background: linear-gradient(90deg, #2563EB 0%, #1D4ED8 100%);
        color: white !important;
        font-size: 1.1rem !important;
        font-weight: 700 !important;
        border-radius: 10px !important;
        padding: 10px 20px !important;
        border: none !important;
        box-shadow: 0 4px 10px rgba(37, 99, 235, 0.2);
    }
    </style>
    
    <div class="hero-banner">
        <h1>📊 AI 기반 아동 미디어 코칭 대시보드</h1>
        <p>『2024 어린이 미디어 이용조사』 원자료 및 BART/DART 예측 모델 기반 동적 시뮬레이션</p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.title("📊 데이터 기반 학부모 미디어 통제 시뮬레이션 및 AI 코칭")
st.write(
    "자녀의 미디어 이용 정보를 입력하면 『2024 어린이 미디어 이용조사』 원자료와 대조하여 "
    "**BART/DART 예측 모형과 SHAP 분석**을 통해 아이 특성별 동적 맞춤 시뮬레이션을 제공합니다."
)

RISK_EMOJI = {"상": "🚨", "중": "⚠️", "하": "✅"}

# =========================================================
# 2. 사용자 입력 폼
# =========================================================
st.header("1️⃣ 자녀 및 보호자 양육 환경 입력")

with st.form("profile_form"):
    col1, col2 = st.columns(2)
    with col1:
        child_age = st.selectbox("자녀의 만 나이", options=list(range(3, 10)), format_func=lambda v: f"만 {v}세")
        child_gender = st.selectbox("자녀의 성별", options=["남아", "여아"])
        daily_time = st.number_input("⏱️ 하루 평균 미디어 이용시간 (시간)", min_value=0.0, max_value=15.0, value=3.0, step=0.5)
        alt_act = st.selectbox("📚 미디어 대체 활동(독서/야외활동 등) 여부", options=["없음", "있음"])

    with col2:
        self_control = st.slider("🎯 자기 조절 수준 (1~5점 척도)", min_value=1.0, max_value=5.0, value=3.0, step=0.5, help="1점: 전혀 못함 ~ 5점: 완벽히 조절함")
        conflict = st.slider("😣 부모와의 갈등 정도 (1~5점 척도)", min_value=1.0, max_value=5.0, value=3.0, step=0.5)
        parent_talk = st.slider("💬 이용 경험 대화 정도 (1~5점 척도)", min_value=1.0, max_value=5.0, value=3.0, step=0.5)
        parent_phone = st.selectbox("📱 자녀 앞 보호자의 스마트폰 사용 여부", options=["적음", "많음"])
        restriction = st.slider("🔒 제한 중심 지도 정도 (적용 규칙 수: 0~12개)", min_value=0, max_value=12, value=6)

    submitted = st.form_submit_button("🔍 데이터 기반 통합 진단 결과 확인", type="primary", use_container_width=True)

if submitted:
    st.session_state["profile"] = {
        "age": child_age,
        "gender_code": 1 if child_gender == "남아" else 2,
        "self_control": self_control,
        "conflict_proxy": conflict,
        "daily_media_hours": daily_time,
        "alt_activity": 1 if alt_act == "있음" else 0,
        "parent_talk": parent_talk,
        "parent_phone_use": 1 if parent_phone == "많음" else 0,
        "restriction_count": restriction,
    }
    st.session_state.pop("selected_solution", None)

# =========================================================
# 3. 결과 대시보드
# =========================================================
if "profile" in st.session_state:
    profile = st.session_state["profile"]
    peer = predictor.peer_comparison(profile)
    pred = predictor.predict_with_ci(profile)

    # 3-1. 또래 비교
    st.divider()
    st.header("2️⃣ 동일 연령·성별 또래 비교")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("자녀 자기조절 점수", f"{profile['self_control']:.1f}점")
    m2.metric("또래 평균 점수", f"{peer['peer_self_control_mean']:.2f}점", delta=f"{profile['self_control'] - peer['peer_self_control_mean']:+.2f}점")
    m3.metric("또래 내 위치", f"{peer['self_control_percentile']:.0f}백분위")
    m4.metric(f"{RISK_EMOJI[peer['risk_level']]} 상대적 위험도", peer["risk_level"])

    # 3-2. 위험도 진단
    st.divider()
    st.header("3️⃣ 아동 미디어 이용행동 위험도 진단")
    risk_col, chart_col = st.columns([1, 1.3])
    
    with risk_col:
        st.markdown(f"## {RISK_EMOJI[peer['risk_level']]} 위험 등급: **{peer['risk_level']}**")
        st.markdown(f"### **{pred['risk_score']:.1f}점** / 100점")
        st.caption(f"95% 베이지안 신뢰구간: {pred['ci_low']:.1f}점 ~ {pred['ci_high']:.1f}점")
        st.progress(min(int(pred["risk_score"]), 100))
        st.write(peer["risk_text"])

    with chart_col:
        fig, ax = plt.subplots(figsize=(5.5, 2.3))
        ax.errorbar(
            [pred["risk_score"]], [0],
            xerr=[[pred["risk_score"] - pred["ci_low"]], [pred["ci_high"] - pred["risk_score"]]],
            fmt="o", color="crimson", markersize=12, capsize=8, elinewidth=2, label="예측 위험도 (95% CI)"
        )
        ax.axvline(peer["peer_risk_mean"], color="gray", linestyle="--", label="또래평균 위험도")
        ax.set_xlim(0, 100)
        ax.set_yticks([])
        ax.set_xlabel("위험도 점수 (0~100)")
        ax.legend(loc="upper right", fontsize=8)
        ax.set_title("BART/DART 예측 위험도 및 95% 베이지안 신뢰구간", fontsize=11)
        st.pyplot(fig)

    # 3-3. SHAP 원인 분석
    st.divider()
    st.header("4️⃣ 맞춤형 코칭 — SHAP 기반 원인 분석")
    contributions = predictor.explain_individual(profile)
    top_factor = contributions[0]
    
    st.markdown(f"🔍 귀하의 자녀는 **'{top_factor['label']}'** 요인이 **{top_factor['direction']}**로 가장 크게 작용했습니다.")
    
    fig2, ax2 = plt.subplots(figsize=(7, 3.2))
    labels = [c["label"] for c in contributions]
    values = [c["shap_value"] for c in contributions]
    colors = ["#e74c3c" if v > 0 else "#2ecc71" for v in values]
    
    ax2.barh(labels[::-1], values[::-1], color=colors[::-1])
    ax2.axvline(0, color="black", linewidth=0.8)
    ax2.set_xlabel("위험도 기여도 (SHAP value)")
    ax2.set_title("변수별 위험도 기여도 (빨강: 위험 상승 / 초록: 위험 완화)", fontsize=11)
    st.pyplot(fig2)

    # 3-4. 개별 맞춤 솔루션 추천 (실시간 조합 생성)
    st.divider()
    st.header("5️⃣ 입력 정보 기반 맞춤 솔루션 추천 ")
    solutions = predictor.recommend_solutions(profile, top_n=3)
    sol_cols = st.columns(3)
    
    for i, (col, sol) in enumerate(zip(sol_cols, solutions)):
        with col:
            box = st.container(border=True)
            with box:
                if i == 0:
                    st.markdown("### ⭐ 최우선 맞춤 추천")
                st.markdown(f"#### {sol['label']}")
                st.markdown(f"<div class='reason-box'>💡 <b>실시간 분석 사유</b><br>{sol['reason']}</div>", unsafe_allow_html=True)
                st.write(sol["description"])
                st.metric("신뢰도 (사후확률)", f"{sol['confidence']*100:.0f}%")
                st.metric("예상 위험도 변화", f"{sol['after_risk']:.1f}점", delta=f"{-sol['improvement']:+.1f}점 ({sol['improvement_pct']:+.1f}%)", delta_color="inverse")

    sol_dict = {s["key"]: s["label"] for s in solutions}
    chosen_key = st.selectbox("🧪 시뮬레이션할 솔루션을 선택하세요", options=list(sol_dict.keys()), format_func=lambda k: sol_dict[k])
    st.session_state["selected_solution"] = chosen_key

    # 3-5. 시뮬레이션
    if "selected_solution" in st.session_state:
        st.divider()
        st.header("6️⃣ 선택 솔루션 적용 시뮬레이션")
        chosen = predictor.simulate_solution(profile, st.session_state["selected_solution"])
        sc1, sc2 = st.columns([1, 1.3])
        
        with sc1:
            st.markdown(f"#### [{chosen['label']}] 적용 시")
            st.write(f"📌 **진단 근거**: {chosen['reason']}")
            st.metric("위험도 변화", f"{chosen['before_risk']:.1f} → {chosen['after_risk']:.1f}점", delta=f"{-chosen['improvement']:+.1f}점 ({chosen['improvement_pct']:+.1f}%)", delta_color="inverse")
            st.caption(f"예상 개선 효과: 평균 {chosen['improvement_pct']:+.1f}%, 95% 신뢰구간 {chosen['ci_low']:.1f}~{chosen['ci_high']:.1f}점 (신뢰도 {chosen['confidence']*100:.0f}%)")

        with sc2:
            fig3, ax3 = plt.subplots(figsize=(5, 2.2))
            ax3.errorbar([chosen["before_risk"]], [1], fmt="o", color="gray", markersize=10, label="적용 전")
            ax3.errorbar(
                [chosen["after_risk"]], [0],
                xerr=[[chosen["after_risk"] - chosen["ci_low"]], [chosen["ci_high"] - chosen["after_risk"]]],
                fmt="o", color="seagreen", markersize=12, capsize=8, elinewidth=2, label="적용 후 (95% CI)"
            )
            ax3.set_xlim(0, 100)
            ax3.set_yticks([0, 1])
            ax3.set_yticklabels(["적용 후", "적용 전"])
            ax3.set_xlabel("위험도 점수")
            ax3.set_title("솔루션 적용 전후 위험도 및 신뢰구간 비교", fontsize=11)
            ax3.legend(loc="upper right", fontsize=8)
            st.pyplot(fig3)

        # 3-6. 이번 주 실천 목표
        st.divider()
        st.header("7️⃣ 이번 주 실천 목표")
        st.success(
            f"⭐ **실천 과제**: {chosen['label']}\n\n"
            f"👉 **행동 지침**: {chosen['description']}\n\n"
            f"🎯 **목표**: 이번 주 동안 꾸준히 실천하여 위험도를 **{chosen['before_risk']:.0f}점 → {chosen['after_risk']:.0f}점** 수준으로 낮춰보세요!"
        )
