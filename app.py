import sys
import os
from pathlib import Path
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.font_manager as fm

# =========================================================
# 0. 한글 폰트 감지 및 Matplotlib 깨짐 방지 (OS 및 Cloud 통합 대응)
# =========================================================
def init_korean_font():
    # Windows, Mac, Linux 시스템 대표 한글 폰트 후보군
    font_candidates = ["NanumGothic", "Malgun Gothic", "AppleGothic", "Noto Sans CJK KR"]
    available_fonts = {f.name for f in fm.fontManager.ttflist}
    
    chosen_font = next((f for f in font_candidates if f in available_fonts), None)
    
    # Streamlit Cloud (Linux) 서버 환경용 나눔폰트 경로 직접 탐색
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
    
    # 차트 마이너스(-) 기호 깨짐 방지
    matplotlib.rcParams["axes.unicode_minus"] = False

init_korean_font()

# 백엔드 연산 모듈 로드
sys.path.insert(0, str(Path(__file__).resolve().parent / "model"))
import predictor

# =========================================================
# 1. 화면 설정 및 CSS (사용자 편의 글자 크기 확대)
# =========================================================
st.set_page_config(
    page_title="데이터 기반 학부모 미디어 통제 시뮬레이션 및 AI 코칭",
    page_icon="📊",
    layout="wide"
)

st.markdown(
    """
    <style>
    html, body, [class*="css"] { font-size: 19px !important; }
    h1 { font-size: 2.2rem !important; color: #1E3A8A; }
    h2 { font-size: 1.7rem !important; border-bottom: 2px solid #E5E7EB; padding-bottom: 8px; }
    h3 { font-size: 1.4rem !important; }
    div[data-testid="stMetricValue"] { font-size: 2.0rem !important; font-weight: bold; }
    div[data-testid="stMetricLabel"] { font-size: 1.1rem !important; }
    .stButton button { font-size: 1.2rem !important; font-weight: bold; padding: 0.7rem 1.2rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📊 데이터 기반 학부모 미디어 통제 시뮬레이션 및 AI 코칭")
st.write(
    "자녀의 미디어 이용 정보를 입력하면 『2024 어린이 미디어 이용조사』 원자료와 대조하여 "
    "**BART/DART 예측 모형과 SHAP 분석**을 통해 동적 맞춤 시뮬레이션을 제공합니다."
)
st.caption(
    "※ 또래 비교는 원자료 기반 기술통계이며, 예상 개선 효과 및 신뢰구간은 베이지안 사후분포 연산 결과입니다. "
    "횡단면 데이터의 특성상 직접적 인과관계가 아닌 통계적 연관성으로 해석되어야 합니다."
)

RISK_EMOJI = {"상": "🚨", "중": "⚠️", "하": "✅"}

# =========================================================
# 2. 사용자 입력 폼 (7단계 흐름 - 1단계)
# =========================================================
st.header("1️⃣자녀 및 보호자 양육 환경 입력")

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
    st.session_state.pop("selected_solution", None) # 새 진단 시 솔루션 선택 초기화

# =========================================================
# 3. 결과 대시보드 (2~7단계)
# =========================================================
if "profile" in st.session_state:
    profile = st.session_state["profile"]
    peer = predictor.peer_comparison(profile)
    pred = predictor.predict_with_ci(profile)

    # -----------------------------------------------------
    # 3-1. 동일 연령·성별 또래 비교 (2단계)
    # -----------------------------------------------------
    st.divider()
    st.header("2️⃣ 동일 연령·성별 또래 비교")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("자녀 자기조절 점수", f"{profile['self_control']:.1f}점")
    m2.metric("또래 평균 점수", f"{peer['peer_self_control_mean']:.2f}점", delta=f"{profile['self_control'] - peer['peer_self_control_mean']:+.2f}점")
    m3.metric("또래 내 위치", f"{peer['self_control_percentile']:.0f}백분위")
    m4.metric(f"{RISK_EMOJI[peer['risk_level']]} 상대적 위험도", peer["risk_level"])
    st.info(f"👨‍👩‍👧 비교 기준: 『2024 어린이 미디어 이용조사』 만 {profile['age']}세 {'남아' if profile['gender_code'] == 1 else '여아'} ({peer['peer_count']:,}명) 원자료")

    # -----------------------------------------------------
    # 3-2. 위험도 진단 및 베이지안 Error Bar 시각화 (3단계)
    # -----------------------------------------------------
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
        ax.axvline(peer["peer_risk_mean"], color="gray", linestyle="--", label="또래 평균 위험도")
        ax.set_xlim(0, 100)
        ax.set_yticks([])
        ax.set_xlabel("위험도 점수 (0~100)")
        ax.legend(loc="upper right", fontsize=8)
        ax.set_title("BART/DART 예측 위험도 및 95% 베이지안 신뢰구간", fontsize=11)
        st.pyplot(fig)

    # -----------------------------------------------------
    # 3-3. SHAP 기반 맞춤형 코칭 (4단계)
-----------------------------------------------
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
    
    st.caption("※ SHAP 기여도는 머신러닝 예측 메커니즘을 사후적으로 분해한 지표이며, 독립적인 인과관계를 보장하지 않습니다.")

    # -----------------------------------------------------
    # 3-4. 입력 정보 기반 동적 맞춤 솔루션 추천 (5단계)
    # -----------------------------------------------------
    st.divider()
    st.header("5️⃣ 입력 정보 기반 맞춤 솔루션 추천 (동적 신뢰도 정렬)")
    solutions = predictor.recommend_solutions(profile, top_n=3)
    sol_cols = st.columns(3)
    
    for i, (col, sol) in enumerate(zip(sol_cols, solutions)):
        with col:
            box = st.container(border=True)
            with box:
                if i == 0:
                    st.markdown("### ⭐ 최우선 추천 (개별 맞춤)")
                st.markdown(f"#### {sol['label']}")
                st.write(sol["description"])
                st.metric("신뢰도 (사후확률)", f"{sol['confidence']*100:.0f}%")
                st.metric("예상 위험도 변화", f"{sol['after_risk']:.1f}점", delta=f"{-sol['improvement']:+.1f}점 ({sol['improvement_pct']:+.1f}%)", delta_color="inverse")

    sol_dict = {s["key"]: s["label"] for s in solutions}
    chosen_key = st.selectbox("🧪 시뮬레이션할 솔루션을 선택하세요", options=list(sol_dict.keys()), format_func=lambda k: sol_dict[k])
    st.session_state["selected_solution"] = chosen_key

    # -----------------------------------------------------
    # 3-5. 선택 솔루션 적용 시뮬레이션 (6단계)
    # -----------------------------------------------------
    if "selected_solution" in st.session_state:
        st.divider()
        st.header("6️⃣ 선택 솔루션 적용 시뮬레이션 (불확실성 시각화)")
        chosen = predictor.simulate_solution(profile, st.session_state["selected_solution"])
        sc1, sc2 = st.columns([1, 1.3])
        
        with sc1:
            st.markdown(f"#### [{chosen['label']}] 적용 시")
            st.metric("위험도 변화", f"{chosen['before_risk']:.1f} → {chosen['after_risk']:.1f}점", delta=f"{-chosen['improvement']:+.1f}점 ({chosen['improvement_pct']:+.1f}%)", delta_color="inverse")
            st.caption(f"예상 개선 효과: 평균 {chosen['improvement_pct']:+.1f}%, 95% 신뢰구간{chosen['ci_low']:.1f}~{chosen['ci_high']:.1f}점 (신뢰도 {chosen['confidence']*100:.0f}%)")

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

        # -----------------------------------------------------
        # 3-6. 이번 주 실천 목표 (7단계)
        # -----------------------------------------------------
        st.divider()
        st.header("7️⃣이번 주 실천 목표")
        st.success(
            f"⭐ **실천 과제**: {chosen['label']}\n\n"
            f"👉 **행동 지침**: {chosen['description']}\n\n"
            f"🎯 **목표**: 이번 주 동안 꾸준히 실천하여 위험도를 **{chosen['before_risk']:.0f}점 → {chosen['after_risk']:.0f}점** 수준으로 낮춰보세요!"
        )
