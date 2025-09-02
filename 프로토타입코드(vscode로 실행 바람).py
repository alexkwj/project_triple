import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import streamlit.components.v1 as components

# -------------------- 전역: 페이지/번역 차단 --------------------
st.set_page_config(page_title="여행지 추천 시스템", layout="wide")

# 자동 번역/교정 차단
components.html("""
<script>
try {
  document.documentElement.setAttribute('translate','no');
  document.documentElement.classList.add('notranslate');
  document.body && document.body.setAttribute('translate','no');
  const meta = document.createElement('meta');
  meta.setAttribute('name','google');
  meta.setAttribute('value','notranslate');
  document.head.appendChild(meta);
} catch (e) {}
</script>
""", height=0)

def T(md: str):
    st.markdown(f'<div translate="no" class="notranslate" lang="ko">{md}</div>', unsafe_allow_html=True)

# -------------------- 스키마/상수 --------------------
NUMERIC_FEATURES = [
    "AGE_GRP",
    "TRAVEL_STYL_1_GROUP","TRAVEL_STYL_2_GROUP","TRAVEL_STYL_3_GROUP",
    "TRAVEL_STYL_4_GROUP","TRAVEL_STYL_5_GROUP","TRAVEL_STYL_6_GROUP",
    "TRAVEL_STYL_7_GROUP","TRAVEL_STYL_8_GROUP",
]
ACCOMPANY_CATS = [
    "나홀로 여행","자녀 동반 여행","2인 가족 여행","2인 여행(가족 외)",
    "부모 동반 여행","3인 이상 여행(가족 외)","3대 동반 여행(친척 포함)","3인 이상 가족 여행(친척 포함)",
]
RESIDENCE_CATS = [11,28,41,48,45,44,50,27,26,30,36,29,31,47,46,43,42]
CATEGORICAL_FEATURES = ["TRAVEL_STATUS_ACCOMPANY", "RESIDENCE_SGG_CD"]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# 가운데(2) 라벨은 화면에 숨김(공백) — 기능적 선택은 가능
STYLE_LABELS = {
    "TRAVEL_STYL_1_GROUP": ("자연 (1)", " ", "도시 (3)"),
    "TRAVEL_STYL_2_GROUP": ("숙박 (1)", " ", "당일 (3)"),
    "TRAVEL_STYL_3_GROUP": ("새로운 지역 (1)", " ", "익숙한 지역 (3)"),
    "TRAVEL_STYL_4_GROUP": ("편하고 비싼 숙소 (1)", " ", "저렴한 숙소 (3)"),
    "TRAVEL_STYL_5_GROUP": ("휴양/휴식 (1)", " ", "액티비티 (3)"),
    "TRAVEL_STYL_6_GROUP": ("숨은 명소 (1)", " ", "유명 명소 (3)"),
    "TRAVEL_STYL_7_GROUP": ("계획 여행 (1)", " ", "즉흥 여행 (3)"),
    "TRAVEL_STYL_8_GROUP": ("사진 촬영 선호 (1)", " ", "사진 촬영 비선호 (3)"),
}

CLUSTER_DESC = {
    0: "당신은 자연 속에서 여유롭게 쉬는 여행을 선호하는 여행자군요!!",
    1: "당신은 가족을 좋아하며 동반 여행을 선호하는 여행자군요!!",
    2: "당신은 체험을 중시하며 거의 모든 활동을 즐기는 친구·연인 동반을 선호하는 만능 액티비티 여행자군요🧗‍♀️",
    3: "당신은 음식과 자연에 집중하며 친구·연인 또는 혼자 여행을 선호하는 가성비 미식·탐험 여행자군요 🍜",
}

# -------------------- 전처리/모델 --------------------
def build_preprocessor():
    num_tf = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    cat_tf = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(
            categories=[ACCOMPANY_CATS, RESIDENCE_CATS],
            handle_unknown="ignore",
            sparse_output=False # sparse_output=False로 변경
        )),
    ])
    return ColumnTransformer([
        ("num", num_tf, NUMERIC_FEATURES),
        ("cat", cat_tf, CATEGORICAL_FEATURES),
    ])

def build_kmeans():
    return KMeans(n_clusters=4, init="k-means++", random_state=42, n_init=10)

@st.cache_resource
def load_resources():
    pre_pkl = Path("preprocessor_final_with_residence.pkl")
    km_pkl  = Path("kmeans_model_final_with_residence.pkl")
    csv_fp  = Path("final.csv")

    if not csv_fp.exists():
        T("**final.csv** 파일을 찾을 수 없습니다. 경로를 확인하세요.")
        st.stop()

    df = pd.read_csv(csv_fp, low_memory=False)
    for c in NUMERIC_FEATURES:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["TRAVEL_STATUS_ACCOMPANY"] = df["TRAVEL_STATUS_ACCOMPANY"].astype("string")
    df["RESIDENCE_SGG_CD"] = pd.to_numeric(df["RESIDENCE_SGG_CD"], errors="coerce")

    pre = None
    km = None

    try:
        with open(pre_pkl, "rb") as f: pre = pickle.load(f)
        with open(km_pkl, "rb") as f: km = pickle.load(f)
        T("✅ 저장된 전처리기/모델 **로드 완료**")
    except (FileNotFoundError, AttributeError): # AttributeError 추가
        T("⚠️ 모델 파일 없음 또는 버전 불일치 → **즉석 학습** 진행")
        pre = build_preprocessor()
        km  = build_kmeans()
        X = pre.fit_transform(df[ALL_FEATURES])
        km.fit(X)
        # 필요 시 저장
        # with open(pre_pkl, "wb") as f: pickle.dump(pre, f)
        # with open(km_pkl, "wb") as f: pickle.dump(km, f)

    # 군집 라벨
    X_all = pre.transform(df[ALL_FEATURES])
    df = df.copy()
    df["cluster"] = km.predict(X_all)

    # 추천 프로필(TF-IDF)
    stopwords = [
        "제주 국제공항","김포국제공항 국내선","사무실","제주동문시장",
        "서귀포 매일 올레시장","서울역","본점","스타벅스","CU","GS25",
        "함덕해수욕장","협재해수욕장","청주국제공항","김포 국제공항 국내선",
        "성산일출봉","오설록 티 뮤지엄","성심당 본점","부산역","대전역","광주공항",
        "서울고속버스터미널","동대구역","인천국제공항","광명역","울산역","경부",
    ]
    if "VISIT_AREA_NM" in df.columns:
        filt = ~df["VISIT_AREA_NM"].isin(stopwords)
        tmp = df.loc[filt, ["cluster","VISIT_AREA_NM"]].dropna()
        tmp["VISIT_AREA_NM_CLEAN"] = tmp["VISIT_AREA_NM"].astype(str).str.replace(" ", "_", regex=False)
        k = km.n_clusters
        grouped = tmp.groupby("cluster")["VISIT_AREA_NM_CLEAN"].apply(lambda s: " ".join(s)).reindex(range(k), fill_value="")
        corpus = grouped.tolist()

        vec = TfidfVectorizer(min_df=2)  # 필요시 1로 낮추기
        tf = vec.fit_transform(corpus)
        vocab = np.array(vec.get_feature_names_out())

        rec_profile = {}
        for i in range(tf.shape[0]):
            row = tf[i].toarray()[0]
            if row.sum() == 0:
                rec_profile[i] = ["(데이터 부족)"]
            else:
                idx = row.argsort()[-5:][::-1]
                rec_profile[i] = [vocab[j].replace("_"," ") for j in idx]
    else:
        rec_profile = {i:["(VISIT_AREA_NM 없음)"] for i in range(km.n_clusters)}

    return pre, km, rec_profile

preprocessor, kmeans, recommendation_profile = load_resources()

# -------------------- 스타일 슬라이더(2 숨김 표시) --------------------
def tri_slider_hidden_mid(key: str, left_label: str, right_label: str, default=2) -> int:
    # 가운데 라벨은 화면에 출력하지 않음(공백)
    T(f"{left_label}  ←→  {right_label}")
    val = st.slider(" ", 1, 3, default, step=1, key=key, label_visibility="collapsed")
    # 선택 표시: 1/3은 구체 라벨, 2는 '중립 (2)'만 간단히 표기
    if val == 1:
        T(f"선택: {left_label}")
    elif val == 3:
        T(f"선택: {right_label}")
    else:
        T("선택: 중립 (2)")
    return val

# -------------------- 입력 폼 --------------------
with st.form("user_form"):
    T("## 1. 기본 정보")

    T("연령대 (10~80)")
    age_grp = st.slider(" ", 10, 80, 30, step=10, label_visibility="collapsed")

    T("관심 지역")
    residence_map = {
        '서울특별시': 11, '인천광역시': 28, '경기도': 41, '강원도': 42, '충청북도': 43,
        '충청남도': 44, '전라북도': 45, '전라남도': 46, '경상북도': 47, '경상남도': 48,
        '대전광역시': 30, '광주광역시': 29, '대구광역시': 27, '부산광역시': 26, '울산광역시': 31,
        '세종특별자치시': 36, '제주특별자치도': 50
    }
    res_name = st.selectbox(" ", list(residence_map.keys()), label_visibility="collapsed")
    res_code = residence_map[res_name]

    T("동반 형태")
    accompany = st.selectbox("  ", ACCOMPANY_CATS, label_visibility="collapsed")

    T("## 2. 여행 스타일 (가운데 값은 기능만, 표시는 최소화)")
    styles = {}
    for key, (left, _, right) in STYLE_LABELS.items():
        styles[key] = tri_slider_hidden_mid(key, left, right, default=2)

    submitted = st.form_submit_button("추천받기")

# -------------------- 예측/출력 --------------------
def predict_cluster(pre, km, inp: dict):
    df_new = pd.DataFrame([inp], columns=ALL_FEATURES)
    for c in NUMERIC_FEATURES:
        df_new[c] = pd.to_numeric(df_new[c], errors="coerce")
    df_new["TRAVEL_STATUS_ACCOMPANY"] = df_new["TRAVEL_STATUS_ACCOMPANY"].astype("string")
    df_new["RESIDENCE_SGG_CD"] = pd.to_numeric(df_new["RESIDENCE_SGG_CD"], errors="coerce")
    X = pre.transform(df_new)
    cluster_id = int(km.predict(X)[0])
    dists = km.transform(X)[0].round(3).tolist()
    return cluster_id, dists

if submitted:
    new_user = {
        "AGE_GRP": age_grp,
        **styles,
        "TRAVEL_STATUS_ACCOMPANY": accompany,
        "RESIDENCE_SGG_CD": res_code,
    }
    cid, dists = predict_cluster(preprocessor, kmeans, new_user)
    recs = recommendation_profile.get(cid, ["추천 데이터 부족"])

    st.success("추천 완료!")
    T("### ⭐ 당신을 위한 맞춤 여행지 ⭐")
    T(f"**{CLUSTER_DESC.get(cid, '군집 설명 없음')}**")
    st.divider()
    T("#### 추천 여행지")
    for p in recs:
        T(f"- {p}")

    with st.expander("디버그 보기 (입력/클러스터 거리)"):
        st.write("입력 특징:", new_user)
        st.write("클러스터 거리(작을수록 가까움):", {f"cluster_{i}":v for i,v in enumerate(dists)})