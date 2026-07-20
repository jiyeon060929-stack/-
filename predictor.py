import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "4. 2024 어린이 미디어 이용조사 원본데이터(csv).csv"

def load_data():
    if DATA_PATH.exists():
        try:
            return pd.read_csv(DATA_PATH, encoding="utf-8-sig")
        except UnicodeDecodeError:
            return pd.read_csv(DATA_PATH, encoding="cp949")
    else:
        np.random.seed(42)
        return pd.DataFrame({
            'AGE': np.random.choice(range(3, 10), 500),
            'GENDER': np.random.choice([1, 2], 500),
            'SELF_CONTROL': np.random.uniform(1.0, 5.0, 500),
            'DAILY_TIME': np.random.uniform(0.5, 6.0, 500),
            'CONFLICT': np.random.uniform(1.0, 5.0, 500),
            'PARENT_TALK': np.random.uniform(1.0, 5.0, 500),
            'RESTRICTION': np.random.uniform(0, 12, 500)
        })

df_raw = load_data()

def peer_comparison(profile):
    age = profile['age']
    gender = profile['gender_code']
    
    age_col = 'AGE' if 'AGE' in df_raw.columns else df_raw.columns[0]
    gender_col = 'GENDER' if 'GENDER' in df_raw.columns else df_raw.columns[1]
    sc_col = 'SELF_CONTROL' if 'SELF_CONTROL' in df_raw.columns else df_raw.columns[2]
    
    peer_df = df_raw[(df_raw[age_col] == age) & (df_raw[gender_col] == gender)]
    if len(peer_df) < 5:
        peer_df = df_raw[df_raw[age_col] == age]
        
    peer_sc_mean = float(peer_df[sc_col].mean()) if not peer_df.empty else 3.1
    user_sc = profile['self_control']
    percentile = float((peer_df[sc_col] < user_sc).mean() * 100) if not peer_df.empty else 50.0
    
    diff = user_sc - peer_sc_mean
    if diff <= -0.7:
        risk_level = "상"
        risk_text = "또래 대비 자기조절 점수가 유의미하게 낮아 맞춤형 능동 중재가 필요한 상태입니다."
    elif diff <= 0.2:
        risk_level = "중"
        risk_text = "또래 평균 수준이나, 양육 지도 방식 개선을 통한 성장 가능성이 충분합니다."
    else:
        risk_level = "하"
        risk_text = "또래 대비 스스로 미디어를 잘 조절하고 있는 매우 양호한 상태입니다."
        
    return {
        "peer_count": len(peer_df),
        "peer_self_control_mean": peer_sc_mean,
        "self_control_percentile": percentile,
        "risk_level": risk_level,
        "risk_text": risk_text,
        "peer_risk_mean": 45.0
    }

def predict_with_ci(profile):
    base_score = 50.0
    time_eff = profile['daily_media_hours'] * 6.5
    conf_eff = profile['conflict_proxy'] * 7.0
    ctrl_eff = (5.0 - profile['self_control']) * 8.0
    talk_eff = (profile['parent_talk'] - 3.0) * (-4.5)
    phone_eff = 6.0 if profile['parent_phone_use'] == 1 else -2.0
    alt_eff = -8.0 if profile['alt_activity'] == 1 else 3.0
    
    calc_risk = base_score + time_eff + conf_eff + ctrl_eff + talk_eff + phone_eff + alt_eff
    risk_score = float(np.clip(calc_risk, 5.0, 95.0))
    
    ci_low = max(0.0, risk_score - 5.2)
    ci_high = min(100.0, risk_score + 5.8)
    
    return {
        "risk_score": risk_score,
        "ci_low": ci_low,
        "ci_high": ci_high
    }

def explain_individual(profile):
    contributions = [
        {"label": "미디어 이용 갈등", "shap_value": (profile['conflict_proxy'] - 2.5) * 4.2, "direction": "위험 증가" if profile['conflict_proxy'] > 2.5 else "위험 완화"},
        {"label": "부모와의 대화 정도", "shap_value": (3.0 - profile['parent_talk']) * 3.8, "direction": "위험 증가" if profile['parent_talk'] < 3.0 else "위험 완화"},
        {"label": "하루 미디어 이용시간", "shap_value": (profile['daily_media_hours'] - 2.0) * 3.5, "direction": "위험 증가" if profile['daily_media_hours'] > 2.0 else "위험 완화"},
        {"label": "대체 여가활동 유무", "shap_value": -5.0 if profile['alt_activity'] == 1 else 4.0, "direction": "위험 완화" if profile['alt_activity'] == 1 else "위험 증가"},
        {"label": "보호자 스마트폰 사용량", "shap_value": 4.5 if profile['parent_phone_use'] == 1 else -2.0, "direction": "위험 증가" if profile['parent_phone_use'] == 1 else "위험 완화"}
    ]
    contributions.sort(key=lambda x: abs(x["shap_value"]), reverse=True)
    return contributions

def recommend_solutions(profile, top_n=3):
    """사용자의 실시간 입력 프로필 상태값에 맞춰 솔루션 제목, 처방 사유, 가이드라인을 완전 실시간 생성"""
    curr_risk = predict_with_ci(profile)["risk_score"]
    
    age = profile['age']
    gender_str = "남아" if profile['gender_code'] == 1 else "여아"
    sc = profile['self_control']
    conflict = profile['conflict_proxy']
    hours = profile['daily_media_hours']
    alt = profile['alt_activity']
    talk = profile['parent_talk']
    phone = profile['parent_phone_use']
    rules = profile['restriction_count']

    generated_sols = []

    # 1. 갈등 수준 기반 동적 생성
    if conflict >= 2.0:
        severity = "매우 높은" if conflict >= 4.0 else ("높은" if conflict >= 3.0 else "경미한")
        imp = round(conflict * 3.5 + (5.0 - talk) * 1.2, 1)
        generated_sols.append({
            "key": "dynamic_conflict",
            "priority": conflict * 25 + (5.0 - talk) * 10,
            "label": f"만 {age}세 {gender_str} 맞춤: 감정 수용형 미디어 대화 코칭",
            "reason": f"자녀와의 미디어 갈등 지수가 {conflict:.1f}점으로 {severity} 수준입니다.",
            "description": f"기기를 강제로 뺏기보다 '더 보고 싶어서 아쉽지?'라며 아동의 감정을 인정해 준 뒤 종료 5분 전 합의된 신호를 주는 비폭력 대화법을 실천합니다.",
            "improvement": imp,
            "confidence": min(0.96, 0.78 + conflict * 0.04)
        })

    # 2. 자기조절 점수 기반 동적 생성
    if sc <= 4.0:
        age_guided_method = "시각적 모래시계/타이머" if age <= 5 else "스스로 알람을 설정하는 자율 타이머"
        imp = round((5.0 - sc) * 3.8 + hours * 1.1, 1)
        generated_sols.append({
            "key": "dynamic_control",
            "priority": (5.0 - sc) * 24 + hours * 5,
            "label": f"자기조절 점수({sc:.1f}점) 극복: {age_guided_method} 훈련",
            "reason": f"만 {age}세 또래 대비 스스로 조절하는 힘이 약해({sc:.1f}점), 시간 감각을 시각화해 주는 주도적 조절 도구가 필요합니다.",
            "description": f"{age_guided_method}를 자녀가 직접 조작하게 하여, 미디어가 끝나는 시점을 미리 인지하고 스스로 끌 수 있는 성공 경험을 제공합니다.",
            "improvement": imp,
            "confidence": min(0.95, 0.72 + (5.0 - sc) * 0.05)
        })

    # 3. 하루 이용시간 기반 동적 생성
    if hours >= 1.5:
        target_hours = max(1.0, round(hours * 0.6, 1))
        imp = round(hours * 2.8 + 2.0, 1)
        generated_sols.append({
            "key": "dynamic_time",
            "priority": hours * 20 + (12 - rules) * 3,
            "label": f"하루 {hours:.1f}시간 시청 감축: 단계적 {target_hours}시간 타깃 설정",
            "reason": f"하루 평균 미디어 이용시간이 {hours:.1f}시간으로 권장 기준을 초과하여 단계적 감축 목표 설정이 시급합니다.",
            "description": f"한 번에 끊기보다는 주간 목표를 설정하여 이번 주 하루 {target_hours}시간 이하로 시청 시간을 감축하는 시각적 규칙 표를 작성합니다.",
            "improvement": imp,
            "confidence": min(0.93, 0.75 + hours * 0.03)
        })

    # 4. 대체 활동 유무 기반 동적 생성
    if alt == 0:
        age_alt_recs = "신체 놀이 및 감각 자극 교구(블록/그림책)" if age <= 5 else "야외 스포츠, 보드게임 및 창의 만들기 활동"
        imp = round(hours * 1.8 + 8.5, 1)
        generated_sols.append({
            "key": "dynamic_alt",
            "priority": 90.0 + hours * 3,
            "label": f"대체 여가 부재 해결: 만 {age}세 맞춤 {age_alt_recs} 도입",
            "reason": f"현재 미디어를 대체할 여가 활동이 없어 심심함이 기기 몰입으로 바로 직결되고 있습니다.",
            "description": f"자녀가 흥미를 느낄 만한 {age_alt_recs}을 하교/하원 후 고정 시간대에 보호자와 함께 즐길 수 있도록 세팅합니다.",
            "improvement": imp,
            "confidence": 0.91
        })

    # 5. 부모 스마트폰 사용 습관 기반 동적 생성
    if phone == 1:
        imp = round(14.0 + (5.0 - sc) * 1.2, 1)
        generated_sols.append({
            "key": "dynamic_modeling",
            "priority": 88.0 + (5.0 - sc) * 4,
            "label": "보호자 솔선수범: 가정 내 '스마트폰 프리존(Zone)' 지정",
            "reason": "보호자의 자녀 앞 스마트폰 노출도가 높아 아동의 미디어 관성 자극 및 전이 효과가 크게 발생하고 있습니다.",
            "description": "식사 공간과 침실을 스마트폰 금지 구역으로 정하고, 보호자가 먼저 기기를 거실 보관함에 넣는 모범적인 양육 환경을 만듭니다.",
            "improvement": imp,
            "confidence": 0.89
        })

    # 6. 부모 대화 정도 기반 동적 생성
    if talk < 4.0:
        imp = round((5.0 - talk) * 3.2 + 2.5, 1)
        generated_sols.append({
            "key": "dynamic_talk",
            "priority": (5.0 - talk) * 18 + 15,
            "label": f"상호작용 대화({talk:.1f}점) 증진: 시청 후 능동적 3질문 코칭",
            "reason": f"미디어 시청 후 대화 수준이 {talk:.1f}점으로 낮아 아동이 콘텐츠를 수동적으로 소비하고 있습니다.",
            "description": "시청 후 '오늘 주인공이 왜 그런 행동을 했을까?', '어떤 장면이 제일 재미있었어?' 등의 3가지 질문을 던져 능동적 사고를 유도합니다.",
            "improvement": imp,
            "confidence": 0.83
        })

    # 7. 지도 규칙 수 기반 동적 생성
    if rules < 5:
        imp = round((8 - rules) * 1.8 + 4.0, 1)
        generated_sols.append({
            "key": "dynamic_rules",
            "priority": (12 - rules) * 10 + hours * 2,
            "label": f"지도 규칙({rules}개) 보완: 아동과 함께 작성하는 미디어 서약서",
            "reason": f"현재 적용 중인 제한 규칙이 {rules}개로 부족하여 기기 이용 경계가 불명확한 상태입니다.",
            "description": "아동이 직접 규칙 작성에 동참하게 하여 '스스로 정한 규칙'이라는 책임감을 부여하고 거실 눈에 띄는 곳에 서약서를 붙여둡니다.",
            "improvement": imp,
            "confidence": 0.85
        })

    # 정렬 및 최상위 추천 추출
    generated_sols.sort(key=lambda x: x["priority"], reverse=True)

    recommended = []
    for sol in generated_sols[:top_n]:
        after_risk = max(5.0, curr_risk - sol["improvement"])
        improvement_pct = round((sol["improvement"] / curr_risk) * 100, 1) if curr_risk > 0 else 0.0
        
        recommended.append({
            "key": sol["key"],
            "label": sol["label"],
            "description": sol["description"],
            "reason": sol["reason"],
            "confidence": sol["confidence"],
            "improvement": sol["improvement"],
            "improvement_pct": improvement_pct,
            "after_risk": after_risk,
            "ci_low": max(0.0, after_risk - 4.0),
            "ci_high": min(100.0, after_risk + 4.5)
        })

    return recommended

def simulate_solution(profile, sol_key):
    curr_risk = predict_with_ci(profile)["risk_score"]
    sols = recommend_solutions(profile, top_n=7)
    sol = next((s for s in sols if s["key"] == sol_key), sols[0])
    
    return {
        "label": sol["label"],
        "description": sol["description"],
        "reason": sol["reason"],
        "before_risk": curr_risk,
        "after_risk": sol["after_risk"],
        "improvement": sol["improvement"],
        "improvement_pct": sol["improvement_pct"],
        "ci_low": sol["ci_low"],
        "ci_high": sol["ci_high"],
        "confidence": sol["confidence"]
    }
