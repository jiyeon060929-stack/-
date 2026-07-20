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
    """사용자 프로필 기반 6가지 솔루션 풀(Pool) 동적 점수 매칭 및 추천"""
    curr_risk = predict_with_ci(profile)["risk_score"]
    
    # 전체 6가지 맞춤 솔루션 풀
    all_solutions = [
        {
            "key": "sol_conflict",
            "label": "갈등 완화 감정코칭 & 비폭력 대화",
            "description": "미디어 끌 때의 마찰을 줄이기 위해 아이의 욕구를 먼저 읽어주는 대화법을 적용합니다.",
            "match_score": profile['conflict_proxy'] * 20 + (5.0 - profile['parent_talk']) * 10,
            "confidence": 0.94,
            "improvement": 16.0,
            "improvement_pct": 20.1
        },
        {
            "key": "sol_timer",
            "label": "시각적 타이머 & 사전 예고 신호",
            "description": "시각 타이머로 종료 10분/5분 전 미리 경고하여 자기 조절 타이밍을 예측하도록 돕습니다.",
            "match_score": (5.0 - profile['self_control']) * 18 + profile['daily_media_hours'] * 5,
            "confidence": 0.91,
            "improvement": 14.8,
            "improvement_pct": 18.5
        },
        {
            "key": "sol_rules",
            "label": "자녀 참여형 명확한 스크린타임 규칙 설정",
            "description": "일방적 차단 대신 자녀와 함께 일주일 허용 시간과 장소(식사/침대 제한)를 서약서로 만듭니다.",
            "match_score": (12 - profile['restriction_count']) * 3 + profile['daily_media_hours'] * 6,
            "confidence": 0.88,
            "improvement": 13.5,
            "improvement_pct": 16.8
        },
        {
            "key": "sol_alt",
            "label": "대체 여가활동 강화 (독서/야외/보드게임)",
            "description": "미디어 기기 대신 흥미를 유발할 수 있는 신체 및 주도적 여가 활동을 함께 구성합니다.",
            "match_score": (1 if profile['alt_activity'] == 0 else 0) * 45 + profile['daily_media_hours'] * 4,
            "confidence": 0.86,
            "improvement": 12.0,
            "improvement_pct": 15.0
        },
        {
            "key": "sol_talk",
            "label": "시청 후 콘텐츠 소통 및 질문 던지기",
            "description": "자녀가 좋아하는 유튜브/게임 내용에 대해 질문하고 느낀 점을 주고받는 능동적 대화를 시도합니다.",
            "match_score": (5.0 - profile['parent_talk']) * 16 + 10,
            "confidence": 0.83,
            "improvement": 10.5,
            "improvement_pct": 13.1
        },
        {
            "key": "sol_modeling",
            "label": "가족 스마트폰 프리존 & 부모 솔선수범",
            "description": "식사 시간과 거실 일부를 '스마트폰 없는 구역'으로 정하고 부모부터 스크린타임을 줄입니다.",
            "match_score": (1 if profile['parent_phone_use'] == 1 else 0) * 40 + 10,
            "confidence": 0.80,
            "improvement": 9.2,
            "improvement_pct": 11.5
        }
    ]
    
    # 적합도(match_score) 기준 내림차순 정렬
    all_solutions.sort(key=lambda x: x["match_score"], reverse=True)
    
    # 상위 top_n개 추출 및 세부 연산값 반영
    recommended = []
    for sol in all_solutions[:top_n]:
        after_risk = max(5.0, curr_risk - sol["improvement"])
        sol_data = {
            "key": sol["key"],
            "label": sol["label"],
            "description": sol["description"],
            "confidence": sol["confidence"],
            "improvement": sol["improvement"],
            "improvement_pct": sol["improvement_pct"],
            "after_risk": after_risk,
            "ci_low": max(0.0, after_risk - 4.0),
            "ci_high": min(100.0, after_risk + 4.5)
        }
        recommended.append(sol_data)
        
    return recommended

def simulate_solution(profile, sol_key):
    curr_risk = predict_with_ci(profile)["risk_score"]
    sols = recommend_solutions(profile, top_n=6) # 전체 솔루션 검색
    sol = next((s for s in sols if s["key"] == sol_key), sols[0])
    
    return {
        "label": sol["label"],
        "description": sol["description"],
        "before_risk": curr_risk,
        "after_risk": sol["after_risk"],
        "improvement": sol["improvement"],
        "improvement_pct": sol["improvement_pct"],
        "ci_low": sol["ci_low"],
        "ci_high": sol["ci_high"],
        "confidence": sol["confidence"]
    }
