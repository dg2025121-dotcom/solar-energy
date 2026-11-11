
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# =========================================================
# 0. 페이지 설정 및 디자인
# =========================================================
st.set_page_config(
    page_title="Solar Insight | 태양광 데이터 분석",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Noto Sans KR', sans-serif;
}
.stApp {
    background: linear-gradient(180deg, #f5f9ff 0%, #f8fbff 45%, #ffffff 100%);
}
.block-container {
    max-width: 1400px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}
.hero {
    background: linear-gradient(120deg, #102a43 0%, #176b87 55%, #20a4a8 100%);
    padding: 30px 34px;
    border-radius: 22px;
    color: white;
    margin-bottom: 22px;
    box-shadow: 0 12px 30px rgba(16, 42, 67, 0.14);
}
.hero-kicker {
    color: #b9f3ed;
    font-size: 0.88rem;
    font-weight: 700;
    letter-spacing: 1.5px;
    margin-bottom: 8px;
}
.hero h1 {
    color: white;
    font-size: 2.05rem;
    font-weight: 800;
    margin: 0 0 10px 0;
}
.hero p {
    color: #e7f6fa;
    font-size: 1rem;
    margin: 0;
}
.section-title {
    color: #17324d;
    font-size: 1.35rem;
    font-weight: 800;
    margin: 8px 0 4px 0;
}
.section-subtitle {
    color: #64748b;
    font-size: 0.92rem;
    margin-bottom: 18px;
}
.metric-card {
    background: white;
    border: 1px solid #e5edf5;
    border-radius: 16px;
    padding: 18px 20px;
    box-shadow: 0 5px 18px rgba(25, 55, 90, 0.05);
    min-height: 112px;
}
.metric-label {
    color: #64748b;
    font-size: 0.84rem;
    font-weight: 600;
    margin-bottom: 8px;
}
.metric-value {
    color: #15324b;
    font-size: 1.55rem;
    font-weight: 800;
}
.metric-help {
    color: #8293a6;
    font-size: 0.78rem;
    margin-top: 4px;
}
.info-card {
    background: #ffffff;
    border: 1px solid #e5edf5;
    border-radius: 16px;
    padding: 18px 20px;
    box-shadow: 0 5px 18px rgba(25, 55, 90, 0.04);
}
div[data-testid="stTabs"] button {
    font-weight: 700;
    font-size: 0.98rem;
}
div[data-testid="stTabs"] button[aria-selected="true"] {
    color: #087f8c;
}
div[data-testid="stDataFrame"] {
    border-radius: 12px;
    overflow: hidden;
}
.small-note {
    color: #718096;
    font-size: 0.82rem;
}
</style>
""", unsafe_allow_html=True)


# =========================================================
# 1. 데이터 불러오기
# =========================================================
DATA_URL = (
    "https://raw.githubusercontent.com/dg2025121-dotcom/solar-energy/"
    "refs/heads/main/data/%ED%95%9C%EB%B9%9B_%EC%8B%9C%EA%B0%84%EB%B3%84_"
    "%ED%83%9C%EC%96%91%EA%B4%91%EB%B0%9C%EC%A0%84%EB%9F%89_%EA%B8%B0%EC%98%A8.csv"
)
REQUIRED_COLUMNS = ["거래일", "시간", "총계량전력량", "기온"]


@st.cache_data(show_spinner="태양광 데이터를 불러오는 중입니다...")
def load_data():
    return pd.read_csv(DATA_URL, encoding="utf-8-sig")


try:
    raw_df = load_data()
except Exception as error:
    st.error("GitHub에서 CSV 파일을 불러오지 못했습니다.")
    st.info("CSV 파일이 공개 저장소에 있는지, 주소가 정확한지 확인해 주세요.")
    st.code(str(error))
    st.stop()

missing_columns = [c for c in REQUIRED_COLUMNS if c not in raw_df.columns]
if missing_columns:
    st.error("CSV에 필요한 열이 없습니다: " + ", ".join(missing_columns))
    st.write("현재 파일의 열 이름:", list(raw_df.columns))
    st.stop()


# =========================================================
# 2. 데이터 전처리
# =========================================================
df = raw_df.copy()
df["거래일"] = pd.to_datetime(df["거래일"], errors="coerce")

for col in ["시간", "총계량전력량", "기온"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

original_count = len(df)
missing_count = int(df[REQUIRED_COLUMNS].isna().any(axis=1).sum())

df = df.dropna(subset=REQUIRED_COLUMNS).copy()
df = df[
    (df["시간"].between(1, 24))
    & (df["총계량전력량"] >= 0)
].copy()

# 같은 날짜·시간에 여러 설비 자료가 있으면 발전량은 합계, 기온은 평균으로 집계
df = (
    df.groupby(["거래일", "시간"], as_index=False)
      .agg({
          "총계량전력량": "sum",
          "기온": "mean"
      })
      .sort_values(["거래일", "시간"])
      .reset_index(drop=True)
)

if df.empty:
    st.error("분석할 데이터가 없습니다. CSV 파일의 내용을 확인해 주세요.")
    st.stop()


# =========================================================
# 3. 상단 소개 및 핵심 지표
# =========================================================
st.markdown("""
<div class="hero">
    <div class="hero-kicker">SOLAR INSIGHT · DATA ANALYTICS</div>
    <h1>태양광 발전량 데이터 분석</h1>
    <p>시간대와 기온에 따른 발전량의 변화를 살펴보고, 머신러닝으로 발전량을 예측합니다.</p>
</div>
""", unsafe_allow_html=True)

date_start = df["거래일"].min()
date_end = df["거래일"].max()
total_generation = df["총계량전력량"].sum()
avg_temp = df["기온"].mean()

m1, m2, m3, m4 = st.columns(4)

with m1:
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">분석 데이터</div>'
        f'<div class="metric-value">{len(df):,}<span style="font-size:0.9rem"> 행</span></div>'
        f'<div class="metric-help">전처리 후 데이터</div></div>',
        unsafe_allow_html=True
    )
with m2:
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">분석 시작일</div>'
        f'<div class="metric-value" style="font-size:1.22rem">{date_start:%Y.%m.%d}</div>'
        f'<div class="metric-help">데이터 수집 시작</div></div>',
        unsafe_allow_html=True
    )
with m3:
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">분석 종료일</div>'
        f'<div class="metric-value" style="font-size:1.22rem">{date_end:%Y.%m.%d}</div>'
        f'<div class="metric-help">데이터 수집 마지막 날짜</div></div>',
        unsafe_allow_html=True
    )
with m4:
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">평균 기온</div>'
        f'<div class="metric-value">{avg_temp:.1f}<span style="font-size:0.9rem"> ℃</span></div>'
        f'<div class="metric-help">전체 시간대 평균</div></div>',
        unsafe_allow_html=True
    )

st.write("")


# =========================================================
# 4. 탭 구성
# =========================================================
tab1, tab2, tab3 = st.tabs([
    "📈 시간대별 변화",
    "🌡️ 기온과 발전량 관계",
    "🤖 머신러닝 예측"
])


# =========================================================
# TAB 1. 시간대별 발전량 및 기온
# =========================================================
with tab1:
    st.markdown('<div class="section-title">시간대별 태양광 발전량과 기온</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">하루 중 어느 시간에 발전량과 기온이 가장 높게 나타나는지 비교합니다.</div>',
        unsafe_allow_html=True
    )

    hourly = (
        df.groupby("시간", as_index=False)
          .agg({
              "총계량전력량": "mean",
              "기온": "mean"
          })
          .sort_values("시간")
    )

    peak_gen = hourly.loc[hourly["총계량전력량"].idxmax()]
    peak_temp = hourly.loc[hourly["기온"].idxmax()]

    a, b = st.columns(2)
    with a:
        st.markdown(
            f'<div class="info-card"><div class="metric-label">발전량이 가장 높은 시간</div>'
            f'<div class="metric-value">{int(peak_gen["시간"])}시</div>'
            f'<div class="metric-help">평균 {peak_gen["총계량전력량"]:,.1f} kWh</div></div>',
            unsafe_allow_html=True
        )
    with b:
        st.markdown(
            f'<div class="info-card"><div class="metric-label">기온이 가장 높은 시간</div>'
            f'<div class="metric-value">{int(peak_temp["시간"])}시</div>'
            f'<div class="metric-help">평균 {peak_temp["기온"]:.1f} ℃</div></div>',
            unsafe_allow_html=True
        )

    st.write("")
    chart_left, chart_right = st.columns(2)

    with chart_left:
        fig_gen = go.Figure()
        fig_gen.add_trace(go.Scatter(
            x=hourly["시간"],
            y=hourly["총계량전력량"],
            mode="lines+markers",
            name="평균 발전량",
            line=dict(color="#F4A62A", width=4, shape="spline"),
            marker=dict(size=8, color="#F4A62A"),
            fill="tozeroy",
            fillcolor="rgba(244,166,42,0.12)",
            hovertemplate="%{x}시<br>평균 발전량: %{y:,.1f} kWh<extra></extra>"
        ))
        fig_gen.update_layout(
            title=dict(text="시간대별 평균 태양광 발전량", font=dict(size=17)),
            xaxis_title="시간",
            yaxis_title="평균 발전량 (kWh)",
            template="plotly_white",
            height=390,
            margin=dict(l=15, r=15, t=60, b=20),
            hovermode="x unified"
        )
        fig_gen.update_xaxes(dtick=1, gridcolor="#edf2f7")
        fig_gen.update_yaxes(gridcolor="#edf2f7")
        st.plotly_chart(fig_gen, use_container_width=True)

    with chart_right:
        fig_temp = go.Figure()
        fig_temp.add_trace(go.Scatter(
            x=hourly["시간"],
            y=hourly["기온"],
            mode="lines+markers",
            name="평균 기온",
            line=dict(color="#2786D8", width=4, shape="spline"),
            marker=dict(size=8, color="#2786D8"),
            fill="tozeroy",
            fillcolor="rgba(39,134,216,0.10)",
            hovertemplate="%{x}시<br>평균 기온: %{y:.1f} ℃<extra></extra>"
        ))
        fig_temp.update_layout(
            title=dict(text="시간대별 평균 기온", font=dict(size=17)),
            xaxis_title="시간",
            yaxis_title="평균 기온 (℃)",
            template="plotly_white",
            height=390,
            margin=dict(l=15, r=15, t=60, b=20),
            hovermode="x unified"
        )
        fig_temp.update_xaxes(dtick=1, gridcolor="#edf2f7")
        fig_temp.update_yaxes(gridcolor="#edf2f7")
        st.plotly_chart(fig_temp, use_container_width=True)

    if int(peak_gen["시간"]) == int(peak_temp["시간"]):
        st.info("평균 발전량과 평균 기온이 가장 높은 시간대가 같습니다.")
    else:
        gap = abs(int(peak_gen["시간"]) - int(peak_temp["시간"]))
        st.info(
            f"발전량 최고 시간은 {int(peak_gen['시간'])}시, "
            f"기온 최고 시간은 {int(peak_temp['시간'])}시로 "
            f"약 {gap}시간 차이가 납니다."
        )


# =========================================================
# TAB 2. 기온과 발전량 관계
# =========================================================
with tab2:
    st.markdown('<div class="section-title">기온과 태양광 발전량의 관계</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">야간의 발전량 0에 가까운 값을 제외하고, 낮 시간대의 기온과 발전량을 비교합니다.</div>',
        unsafe_allow_html=True
    )

    day_df = df[df["시간"].between(7, 19)].copy()

    if len(day_df) >= 2:
        corr = day_df["기온"].corr(day_df["총계량전력량"])
    else:
        corr = np.nan

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">분석 시간대</div>'
            f'<div class="metric-value">07~19시</div>'
            f'<div class="metric-help">낮 시간대 데이터</div></div>',
            unsafe_allow_html=True
        )
    with c2:
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">분석 데이터 수</div>'
            f'<div class="metric-value">{len(day_df):,}<span style="font-size:0.9rem"> 행</span></div>'
            f'<div class="metric-help">산점도에 표시되는 데이터</div></div>',
            unsafe_allow_html=True
        )
    with c3:
        corr_text = f"{corr:.3f}" if pd.notna(corr) else "계산 불가"
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">피어슨 상관계수</div>'
            f'<div class="metric-value">{corr_text}</div>'
            f'<div class="metric-help">기온과 발전량의 선형 상관 정도</div></div>',
            unsafe_allow_html=True
        )

    st.write("")

    if len(day_df) >= 2:
        # 추세선 계산
        x = day_df["기온"].to_numpy()
        y = day_df["총계량전력량"].to_numpy()
        if np.unique(x).size >= 2:
            slope, intercept = np.polyfit(x, y, 1)
            line_x = np.linspace(x.min(), x.max(), 100)
            line_y = slope * line_x + intercept
        else:
            line_x, line_y = [], []

        fig_scatter = px.scatter(
            day_df,
            x="기온",
            y="총계량전력량",
            color="시간",
            color_continuous_scale="Turbo",
            opacity=0.55,
            labels={
                "기온": "기온 (℃)",
                "총계량전력량": "태양광 발전량 (kWh)",
                "시간": "시간"
            },
            title="낮 시간대 기온과 태양광 발전량 산점도"
        )
        if len(line_x) > 0:
            fig_scatter.add_trace(go.Scatter(
                x=line_x,
                y=line_y,
                mode="lines",
                name="선형 추세선",
                line=dict(color="#1D3557", width=3, dash="dash")
            ))
        fig_scatter.update_layout(
            template="plotly_white",
            height=520,
            margin=dict(l=15, r=15, t=65, b=20),
            coloraxis_colorbar=dict(title="시간"),
            legend_title_text="구분"
        )
        fig_scatter.update_xaxes(gridcolor="#edf2f7")
        fig_scatter.update_yaxes(gridcolor="#edf2f7")
        st.plotly_chart(fig_scatter, use_container_width=True)

        if corr > 0.3:
            corr_desc = "양의 상관관계가 비교적 뚜렷하게 나타납니다."
        elif corr > 0:
            corr_desc = "약한 양의 상관관계가 나타납니다."
        elif corr < -0.3:
            corr_desc = "음의 상관관계가 비교적 뚜렷하게 나타납니다."
        elif corr < 0:
            corr_desc = "약한 음의 상관관계가 나타납니다."
        else:
            corr_desc = "뚜렷한 선형 상관관계가 나타나지 않습니다."

        st.markdown(
            f'<div class="info-card"><b>그래프 해석</b><br>{corr_desc}'
            f'<br><span class="small-note">상관계수는 두 변수의 함께 변하는 정도를 나타냅니다. '
            f'기온이 발전량의 원인이라는 뜻은 아니며, 일사량·구름량 등 다른 요인도 영향을 줄 수 있습니다.</span></div>',
            unsafe_allow_html=True
        )
    else:
        st.warning("상관관계를 계산할 데이터가 충분하지 않습니다.")


# =========================================================
# TAB 3. 머신러닝 모델 3개
# =========================================================
with tab3:
    st.markdown('<div class="section-title">머신러닝 모델 3개로 태양광 발전량 예측</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">같은 학습·테스트 데이터로 세 모델을 학습하고 예측 성능을 비교합니다.</div>',
        unsafe_allow_html=True
    )

    model_df = df.copy()
    hour = model_df["시간"].astype(float)
    day_of_year = model_df["거래일"].dt.dayofyear.astype(float)

    model_df["시간_sin"] = np.sin(2 * np.pi * hour / 24)
    model_df["시간_cos"] = np.cos(2 * np.pi * hour / 24)
    model_df["연중일_sin"] = np.sin(2 * np.pi * day_of_year / 365.25)
    model_df["연중일_cos"] = np.cos(2 * np.pi * day_of_year / 365.25)

    features = [
        "기온",
        "시간_sin",
        "시간_cos",
        "연중일_sin",
        "연중일_cos"
    ]
    target = "총계량전력량"

    unique_dates = np.array(
        sorted(model_df["거래일"].dt.normalize().unique())
    )

    if len(unique_dates) < 2:
        st.error("학습용과 테스트용 데이터를 나누려면 날짜가 2일 이상 필요합니다.")
        st.stop()

    split_index = int(len(unique_dates) * 0.8)
    split_index = min(max(split_index, 1), len(unique_dates) - 1)
    split_date = pd.Timestamp(unique_dates[split_index])

    train_df = model_df[
        model_df["거래일"].dt.normalize() < split_date
    ].copy()
    test_df = model_df[
        model_df["거래일"].dt.normalize() >= split_date
    ].copy()

    X_train, y_train = train_df[features], train_df[target]
    X_test, y_test = test_df[features], test_df[target]

    info1, info2, info3 = st.columns(3)
    with info1:
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">학습 데이터</div>'
            f'<div class="metric-value">{len(train_df):,}<span style="font-size:0.9rem"> 행</span></div>'
            f'<div class="metric-help">전체 날짜 중 앞의 약 80%</div></div>',
            unsafe_allow_html=True
        )
    with info2:
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">테스트 데이터</div>'
            f'<div class="metric-value">{len(test_df):,}<span style="font-size:0.9rem"> 행</span></div>'
            f'<div class="metric-help">전체 날짜 중 뒤의 약 20%</div></div>',
            unsafe_allow_html=True
        )
    with info3:
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">테스트 시작일</div>'
            f'<div class="metric-value" style="font-size:1.25rem">{split_date:%Y.%m.%d}</div>'
            f'<div class="metric-help">시간 순서 기준 분리</div></div>',
            unsafe_allow_html=True
        )

    models = {
        "선형 회귀": LinearRegression(),
        "랜덤 포레스트": RandomForestRegressor(
            n_estimators=150,
            max_depth=18,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        ),
        "히스토그램 기반 그래디언트 부스팅": HistGradientBoostingRegressor(
            max_iter=150,
            learning_rate=0.08,
            max_leaf_nodes=20,
            l2_regularization=1.0,
            random_state=42
        )
    }

    results = []
    predictions = {}

    with st.spinner("세 가지 머신러닝 모델을 학습하고 있습니다..."):
        for model_name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = np.maximum(model.predict(X_test), 0)
            predictions[model_name] = y_pred

            results.append({
                "모델": model_name,
                "MAE": mean_absolute_error(y_test, y_pred),
                "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
                "R²": r2_score(y_test, y_pred)
            })

    results_df = (
        pd.DataFrame(results)
        .sort_values("RMSE")
        .reset_index(drop=True)
    )
    best_model_name = results_df.iloc[0]["모델"]

    st.markdown("#### 모델별 성능 요약")

    metric_cols = st.columns(3)
    for i, row in results_df.iterrows():
        with metric_cols[i]:
            st.markdown(
                f'<div class="info-card"><div class="metric-label">{row["모델"]}</div>'
                f'<div class="metric-value" style="font-size:1.35rem">{row["RMSE"]:,.1f}</div>'
                f'<div class="metric-help">RMSE · 낮을수록 좋은 성능</div>'
                f'<div class="metric-help">MAE {row["MAE"]:,.1f}　|　R² {row["R²"]:.3f}</div></div>',
                unsafe_allow_html=True
            )

    st.write("")
    st.markdown(
        f'<div class="info-card"><b>🏆 RMSE 기준 가장 좋은 모델</b>'
        f'<div class="metric-value" style="font-size:1.5rem">{best_model_name}</div>'
        f'<span class="small-note">RMSE는 실제 발전량과 예측 발전량의 오차 크기를 나타내며, 값이 낮을수록 좋습니다.</span></div>',
        unsafe_allow_html=True
    )

    st.write("")
    st.markdown("#### 세 모델의 성능 비교")

    score_left, score_right = st.columns(2)

    with score_left:
        fig_rmse = px.bar(
            results_df,
            x="모델",
            y="RMSE",
            color="모델",
            color_discrete_sequence=["#238A9B", "#F4A62A", "#5476C8"],
            text_auto=".1f",
            title="RMSE 비교 (낮을수록 좋음)"
        )
        fig_rmse.update_layout(
            template="plotly_white",
            showlegend=False,
            height=380,
            margin=dict(l=10, r=10, t=60, b=70),
            xaxis_title="",
            yaxis_title="RMSE (kWh)"
        )
        fig_rmse.update_xaxes(tickangle=-15)
        st.plotly_chart(fig_rmse, use_container_width=True)

    with score_right:
        fig_r2 = px.bar(
            results_df,
            x="모델",
            y="R²",
            color="모델",
            color_discrete_sequence=["#238A9B", "#F4A62A", "#5476C8"],
            text_auto=".3f",
            title="R² 비교 (높을수록 좋음)"
        )
        fig_r2.update_layout(
            template="plotly_white",
            showlegend=False,
            height=380,
            margin=dict(l=10, r=10, t=60, b=70),
            xaxis_title="",
            yaxis_title="R²"
        )
        fig_r2.update_xaxes(tickangle=-15)
        st.plotly_chart(fig_r2, use_container_width=True)

    st.markdown("#### 실제 발전량과 예측 발전량 비교")

    selected_model = st.selectbox(
        "비교할 모델을 선택하세요",
        list(models.keys()),
        index=list(models.keys()).index(best_model_name)
    )

    plot_count = min(300, len(test_df))
    chart_df = pd.DataFrame({
        "날짜": test_df["거래일"].iloc[:plot_count].to_numpy(),
        "실제 발전량": y_test.iloc[:plot_count].to_numpy(),
        "예측 발전량": predictions[selected_model][:plot_count]
    })

    fig_compare = go.Figure()
    fig_compare.add_trace(go.Scatter(
        x=chart_df["날짜"],
        y=chart_df["실제 발전량"],
        mode="lines",
        name="실제 발전량",
        line=dict(color="#173F5F", width=2.5)
    ))
    fig_compare.add_trace(go.Scatter(
        x=chart_df["날짜"],
        y=chart_df["예측 발전량"],
        mode="lines",
        name="예측 발전량",
        line=dict(color="#F4A62A", width=2.5)
    ))
    fig_compare.update_layout(
        title=f"{selected_model} · 실제값과 예측값",
        template="plotly_white",
        height=430,
        hovermode="x unified",
        margin=dict(l=15, r=15, t=60, b=20),
        xaxis_title="날짜",
        yaxis_title="발전량 (kWh)",
        legend=dict(orientation="h", y=1.12, x=0)
    )
    fig_compare.update_xaxes(gridcolor="#edf2f7")
    fig_compare.update_yaxes(gridcolor="#edf2f7")
    st.plotly_chart(fig_compare, use_container_width=True)

    st.markdown("#### 랜덤 포레스트 변수 중요도")

    rf_model = models["랜덤 포레스트"]
    importance_df = pd.DataFrame({
        "변수": features,
        "중요도": rf_model.feature_importances_
    }).sort_values("중요도", ascending=True)

    fig_importance = px.bar(
        importance_df,
        x="중요도",
        y="변수",
        orientation="h",
        text_auto=".3f",
        title="발전량 예측에 활용된 변수의 상대적 중요도",
        color="중요도",
        color_continuous_scale=["#B9E4E8", "#167D91"]
    )
    fig_importance.update_layout(
        template="plotly_white",
        height=350,
        margin=dict(l=10, r=10, t=60, b=20),
        coloraxis_showscale=False,
        xaxis_title="중요도",
        yaxis_title=""
    )
    st.plotly_chart(fig_importance, use_container_width=True)

    st.markdown(
        '<div class="info-card"><b>분석할 때 주의할 점</b><br>'
        '이번 모델은 기온, 시간, 계절 정보로 발전량을 예측합니다. '
        '실제 태양광 발전량은 일사량, 구름량, 강수량, 패널 방향과 상태 등에도 영향을 받습니다. '
        '따라서 모델 결과는 예측 성능과 변수 사이의 관계를 살펴보는 자료이며, '
        '기온이 발전량의 원인이라는 것을 증명하지는 않습니다.</div>',
        unsafe_allow_html=True
    )


# =========================================================
# 하단 출처
# =========================================================
st.markdown("---")
st.markdown(
    '<div class="small-note">데이터 출처: 한국수력원자력 한빛 시간별 태양광 발전량·기온 데이터 '
    '| 시각화: Plotly | 머신러닝: Scikit-learn</div>',
    unsafe_allow_html=True
)
