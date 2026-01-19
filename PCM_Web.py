import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.neighbors import KNeighborsRegressor

# =========================================================
# 1. 설정 및 데이터 엔진 (기존과 동일)
# =========================================================
DB_FILENAME = 'PCM_DB.xlsx'
INPUT_FEATURES = ['L*', 'a*', 'b*'] 
ALL_PIGMENTS = ['170_백색', '적황', '적갈', '흑색', '특남', '특녹', '바이']

# 실측 원색 데이터
PIGMENT_LAB_INFO = {
    '170_백색': {'L': 91.49, 'a': -1.67, 'b': -1.63},
    '적황':     {'L': 65.71, 'a': 14.16, 'b': 54.22},
    '적갈':     {'L': 36.72, 'a': 29.49, 'b': 24.36},
    '흑색':     {'L': 20.34, 'a': -0.24, 'b': 0.49},
    '특남':     {'L': 25.19, 'a': 3.58,  'b': -26.10},
    '특녹':     {'L': 32.49, 'a': -32.31, 'b': -1.33},
    '바이':     {'L': 21.22, 'a': 2.28,  'b': 0.21},
}

INITIAL_DATA = [
    {'샘플명': 'Base-White', 'L*': 91.5, 'a*': -1.6, 'b*': -1.6, '170_백색': 100, '적황': 0, '적갈': 0, '흑색': 0, '특남': 0, '특녹': 0, '바이': 0},
    {'샘플명': 'Base-Black', 'L*': 20.3, 'a*': -0.2, 'b*': 0.5, '170_백색': 0, '적황': 0, '적갈': 0, '흑색': 100, '특남': 0, '특녹': 0, '바이': 0},
]

class PaintEngine:
    def __init__(self, db_file):
        self.db_file = db_file
        self.load_or_create_data()

    def load_or_create_data(self):
        if not os.path.exists(self.db_file):
            df = pd.DataFrame(INITIAL_DATA)
            for col in INPUT_FEATURES + ALL_PIGMENTS:
                if col not in df.columns: df[col] = 0
            df.to_excel(self.db_file, index=False)
            self.df = df
        else:
            self.df = pd.read_excel(self.db_file)

    def predict(self, l, a, b, active_pigments):
        if len(self.df) < 1: return {}
        X = self.df[INPUT_FEATURES].fillna(0)
        Y = self.df[active_pigments].fillna(0)

        k = min(3, len(self.df))
        model = KNeighborsRegressor(n_neighbors=k, weights='distance')
        model.fit(X, Y)
        pred = model.predict([[l, a, b]])[0]
        
        raw_result = {pig: max(0, val) for pig, val in zip(active_pigments, pred)}
        total = sum(raw_result.values())
        
        if total > 0:
            return {k: (v/total)*100 for k, v in raw_result.items()}
        return {k: 0 for k in raw_result.keys()}

def lab_to_rgb(L, a, b):
    # 정밀 변환 공식
    y = (L + 16) / 116
    x = a / 500 + y
    z = y - b / 200
    yn, xn, zn = 100.0, 95.047, 108.883
    
    def f_inv(t): return t**3 if t > 6/29 else 3 * (6/29)**2 * (t - 4/29)
    
    X, Y, Z = xn * f_inv(x) / 100, yn * f_inv(y) / 100, zn * f_inv(z) / 100
    r_l =  3.2406 * X - 1.5372 * Y - 0.4986 * Z
    g_l = -0.9689 * X + 1.8758 * Y + 0.0415 * Z
    b_l =  0.0557 * X - 0.2040 * Y + 1.0570 * Z
    
    def gamma(c): return 12.92 * c if c <= 0.0031308 else 1.055 * (max(0, c) ** (1/2.4)) - 0.055
    return int(max(0, min(1, gamma(r_l)))*255), int(max(0, min(1, gamma(g_l)))*255), int(max(0, min(1, gamma(b_l)))*255)

# =========================================================
# 2. 웹 화면 구성 (Streamlit)
# =========================================================
st.set_page_config(page_title="PCM Mobile", page_icon="🎨")

# 스타일 적용
st.markdown("""
    <style>
    .stButton>button { width: 100%; background-color: #4CAF50; color: white; height: 3em; font-size: 20px; }
    </style>
    """, unsafe_allow_html=True)

st.title("🎨 PCM Master Mobile")
st.caption("AI 기반 페인트 조색 시스템")

# 엔진 로딩
engine = PaintEngine(DB_FILENAME)
st.success(f"데이터베이스 연결됨: {len(engine.df)}개 데이터")

# 1. 입력창
with st.container():
    st.subheader("1. 목표 색상 (Target)")
    col1, col2, col3 = st.columns(3)
    t_l = col1.number_input("L*", value=90.0, step=1.0)
    t_a = col2.number_input("a*", value=0.0, step=0.1)
    t_b = col3.number_input("b*", value=0.0, step=0.1)

# 2. 안료 선택
st.subheader("2. 안료 선택")
selected_pigments = st.multiselect("사용할 안료를 선택하세요", ALL_PIGMENTS, default=ALL_PIGMENTS)

# 3. 실행 버튼
if st.button("배합비 계산하기 (Click)"):
    if not selected_pigments:
        st.error("안료를 적어도 하나는 선택해야 합니다.")
    else:
        # 예측
        recipe = engine.predict(t_l, t_a, t_b, selected_pigments)
        
        # 4. 결과 보여주기
        st.divider()
        st.subheader("📊 추천 배합비 (Total 100%)")
        
        # 컬러 프리뷰
        r, g, b = lab_to_rgb(t_l, t_a, t_b)
        color_css = f"background-color: rgb({r}, {g}, {b}); width: 100%; height: 100px; border-radius: 10px; border: 2px solid #ddd; margin-bottom: 20px;"
        st.markdown(f'<div style="{color_css}"></div>', unsafe_allow_html=True)
        st.caption(f"예상 색상 (R:{r}, G:{g}, B:{b})")
        
        # 배합표
        sorted_recipe = sorted(recipe.items(), key=lambda x: x[1], reverse=True)
        df_res = pd.DataFrame(sorted_recipe, columns=["안료명", "비율(%)"])
        df_res = df_res[df_res["비율(%)"] > 0.001] # 0인거 숨김
        df_res["비율(%)"] = df_res["비율(%)"].map('{:.2f}'.format)
        
        st.table(df_res)