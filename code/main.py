import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---------------------------------------------------------
# 0. Streamlit 기본 설정
# ---------------------------------------------------------
st.set_page_config(
    page_title="태양광 발전량과 기온 분석",
    page_icon="☀️",
    layout="wide"
)

st.title("☀️ 태양광 발전량과 기온의 관계 분석")
st.write(
    "시간대별 태양광 발전량과 기온의 관계를 그래프로 살펴보고, "
    "머신러닝 모델 3개로 태양광 발전량을 예측합니다."
)

# GitHub 저장소의 CSV 주소
DATA_URL = (
    "https://raw.githubusercontent.com/dg2025121-dotcom/solar-energy/"
    "refs/heads/main/data/%ED%95%9C%EB%B9%9B_%EC%8B%9C%EA%B0%84%EB%B3%84_"
    "%ED%83%9C%EC%96%91%EA%B4%91%EB%B0%9C%EC%A0%84%EB%9F%89_%EA%B8%B0%EC%98%A8.csv"
)

REQUIRED_COLUMNS = ["거래일", "시간", "총계량전력량", "기온"]


# ---------------------------------------------------------
# 1. 데이터 불러오기
# ---------------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv(DATA_URL, encoding="utf-8-sig")


try:
    raw_df = load_data()
except Exception as error:
    st.error("GitHub에서 CSV 파일을 불러오지 못했습니다.")
    st.write("CSV 파일이 공개 저장소에 있는지, 주소가 맞는지 확인해 주세요.")
    st.code(str(error))
    st.stop()

missing_columns = [column for column in REQUIRED_COLUMNS if column not in raw_df.columns]
if missing_columns:
    st.error("CSV 파일에 필요한 열이 없습니다: " + ", ".join(missing_columns))
    st.write("현재 CSV 열 이름:", list(raw_df.columns))
    st.stop()


# ---------------------------------------------------------
# 2. 데이터 정리
# ---------------------------------------------------------
df = raw_df.copy()

df["거래일"] = pd.to_datetime(df["거래일"], errors="coerce")
for column in ["시간", "총계량전력량", "기온"]:
    df[column] = pd.to_numeric(df[column], errors="coerce")

original_rows = len(df)
missing_rows = int(df[REQUIRED_COLUMNS].isna().any(axis=1).sum())
duplicate_rows = int(df.duplicated(subset=["거래일", "시간"]).sum())

# 날짜, 시간, 발전량, 기온이 없는 행 제거
df = df.dropna(subset=REQUIRED_COLUMNS).copy()

# 시간은 1~24시, 발전량은 0 이상인 데이터만 사용
invalid_rows = int(
    ((df["시간"] < 1) | (df["시간"] > 24) | (df["총계량전력량"] < 0)).sum()
)
df = df[
    (df["시간"] >= 1)
    & (df["시간"] <= 24)
    & (df["총계량전력량"] >= 0)
].copy()

# 같은 날짜와 시간에 여러 행이 있으면 평균으로 묶어 시간별 데이터 생성
# 지역 열이 있는 원본 데이터에서도 날짜·시간 단위 분석이 가능하도록 처리
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
    st.error("데이터 정리 후 남은 행이 없습니다. CSV 파일을 확인해 주세요.")
    st.stop()


# ---------------------------------------------------------
# 3. 데이터 기본 정보
# ---------------------------------------------------------
st.header("1. 데이터 기본 정보")

col1, col2, col3 = st.columns(3)
col1.metric("분석 데이터 행 수", f"{len(df):,}개")
col2.metric("분석 데이터 열 수", "4개")
col3.metric(
    "분석 기간",
    f"{df['거래일'].min():%Y-%m-%d} ~ {df['거래일'].max():%Y-%m-%d}"
)

st.write("**사용한 열**")
st.write("- 거래일: 발전량이 기록된 날짜")
st.write("- 시간: 하루 중 시간(1~24시)")
st.write("- 총계량전력량: 해당 시간의 태양광 발전량(kWh)")
st.write("- 기온: 해당 시간의 기온(℃)")

with st.expander("데이터 정리 과정 보기"):
    st.write(f"- 원본 행 수: {original_rows:,}개")
    st.write(f"- 필수 항목 결측 행 수: {missing_rows:,}개")
    st.write(f"- 날짜·시간 중복 행 수(정리 전): {duplicate_rows:,}개")
    st.write(f"- 시간 범위 또는 음수 발전량 오류 행 수: {invalid_rows:,}개")
    st.write(f"- 정리 후 행 수: {len(df):,}개")
    st.caption(
        "발전량이 큰 값은 실제 높은 발전량일 수 있으므로, "
        "통계적으로 크다는 이유만으로 임의 삭제하지 않았습니다."
    )

with st.expander("정리된 데이터 미리 보기"):
    st.dataframe(df.head(20), use_container_width=True)


# ---------------------------------------------------------
# 4. 기존 그래프 분석
# ---------------------------------------------------------
st.header("2. 시간대별 태양광 발전량과 기온 그래프")

hourly = (
    df.groupby("시간", as_index=False)
    .agg({
        "총계량전력량": "mean",
        "기온": "mean"
    })
    .sort_values("시간")
)

fig1, ax1 = plt.subplots(figsize=(10, 4))
ax1.plot(
    hourly["시간"],
    hourly["총계량전력량"],
    marker="o",
    linewidth=2
)
ax1.set_title("시간대별 평균 태양광 발전량")
ax1.set_xlabel("시간")
ax1.set_ylabel("평균 발전량 (kWh)")
ax1.set_xticks(range(1, 25))
ax1.grid(True, alpha=0.3)
st.pyplot(fig1)
plt.close(fig1)

fig2, ax2 = plt.subplots(figsize=(10, 4))
ax2.plot(
    hourly["시간"],
    hourly["기온"],
    marker="o",
    linewidth=2
)
ax2.set_title("시간대별 평균 기온")
ax2.set_xlabel("시간")
ax2.set_ylabel("평균 기온 (℃)")
ax2.set_xticks(range(1, 25))
ax2.grid(True, alpha=0.3)
st.pyplot(fig2)
plt.close(fig2)

st.header("3. 기온과 태양광 발전량의 관계")

# 야간에는 발전량이 거의 0이므로 낮 시간대(7~19시)를 따로 분석
day_df = df[(df["시간"] >= 7) & (df["시간"] <= 19)].copy()

fig3, ax3 = plt.subplots(figsize=(8, 5))
ax3.scatter(
    day_df["기온"],
    day_df["총계량전력량"],
    alpha=0.25,
    s=12
)
ax3.set_title("낮 시간대 기온과 태양광 발전량의 관계")
ax3.set_xlabel("기온 (℃)")
ax3.set_ylabel("태양광 발전량 (kWh)")
ax3.grid(True, alpha=0.3)
st.pyplot(fig3)
plt.close(fig3)

if len(day_df) >= 2:
    correlation = day_df["기온"].corr(day_df["총계량전력량"])
    st.metric("낮 시간대 기온과 발전량의 피어슨 상관계수", f"{correlation:.3f}")    
    st.caption(
        "상관계수는 두 변수의 함께 변하는 정도를 나타냅니다. "
        "상관관계만으로 기온이 발전량을 직접 변화시킨다고 단정할 수는 없습니다."
    )

max_generation = hourly.loc[hourly["총계량전력량"].idxmax()]
max_temperature = hourly.loc[hourly["기온"].idxmax()]

st.subheader("그래프에서 확인한 결과")
result_col1, result_col2 = st.columns(2)
result_col1.metric(
    "평균 발전량이 가장 높은 시간",
    f"{int(max_generation['시간'])}시",
    f"{max_generation['총계량전력량']:,.2f} kWh"
)
result_col2.metric(
    "평균 기온이 가장 높은 시간",
    f"{int(max_temperature['시간'])}시",
    f"{max_temperature['기온']:.2f} ℃"
)

if int(max_generation["시간"]) == int(max_temperature["시간"]):
    st.write("평균 발전량과 평균 기온이 가장 높은 시간대가 같습니다.")
else:
    st.write(
        f"평균 발전량이 가장 높은 시간은 {int(max_generation['시간'])}시이고, "
        f"평균 기온이 가장 높은 시간은 {int(max_temperature['시간'])}시입니다. "
        "두 최고 시간대가 반드시 일치하지는 않습니다."
    )


# ---------------------------------------------------------
# 5. 머신러닝용 데이터 만들기
# ---------------------------------------------------------
st.header("4. 머신러닝 모델 3개로 태양광 발전량 예측")

st.write(
    "세 모델 모두 기온, 시간대, 계절 정보를 사용해 태양광 발전량을 예측합니다. "
    "데이터를 날짜순으로 정렬한 뒤 앞의 80%는 학습용, 뒤의 20%는 테스트용으로 나눕니다."
)

model_df = df.copy()
hour = model_df["시간"].astype(float)
day_of_year = model_df["거래일"].dt.dayofyear.astype(float)

# 시간과 계절은 반복되는 주기이므로 사인·코사인 값으로 변환
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

# 같은 날짜의 데이터가 학습과 테스트에 섞이지 않도록 날짜 단위로 분리
unique_dates = np.array(sorted(model_df["거래일"].dt.normalize().unique()))

if len(unique_dates) < 2:
    st.error("날짜가 2일 이상인 데이터가 있어야 학습용과 테스트용으로 나눌 수 있습니다.")
    st.stop()

split_index = int(len(unique_dates) * 0.8)
split_index = min(max(split_index, 1), len(unique_dates) - 1)
split_date = pd.Timestamp(unique_dates[split_index])

train_df = model_df[model_df["거래일"].dt.normalize() < split_date].copy()
test_df = model_df[model_df["거래일"].dt.normalize() >= split_date].copy()

X_train = train_df[features]
y_train = train_df[target]
X_test = test_df[features]
y_test = test_df[target]

st.write(f"- 학습 데이터: {len(train_df):,}행")
st.write(f"- 테스트 데이터: {len(test_df):,}행")
st.write(f"- 테스트 데이터 시작일: {split_date:%Y-%m-%d}")

# ---------------------------------------------------------
# 6. 모델 3개 학습 및 성능 비교
# ---------------------------------------------------------
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

with st.spinner("모델 3개를 학습하고 성능을 평가하고 있습니다..."):
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = np.maximum(model.predict(X_test), 0)
        predictions[model_name] = y_pred

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        results.append({
            "모델": model_name,
            "MAE": mae,
            "RMSE": rmse,
            "R²": r2
        })

results_df = pd.DataFrame(results).sort_values("RMSE").reset_index(drop=True)

st.subheader("모델 성능 비교")
st.dataframe(
    results_df.style.format({
        "MAE": "{:,.2f}",
        "RMSE": "{:,.2f}",
        "R²": "{:.4f}"
    }),
    use_container_width=True
)

best_model_name = results_df.iloc[0]["모델"]
st.success(f"RMSE 기준으로 가장 성능이 좋은 모델은 **{best_model_name}**입니다.")

st.write("**평가 지표 뜻**")
st.write("- MAE: 실제 발전량과 예측 발전량 차이의 평균입니다. 낮을수록 좋습니다.")
st.write("- RMSE: 예측 오차를 제곱해 평균한 뒤 제곱근을 구한 값입니다. 낮을수록 좋습니다.")
st.write("- R²: 모델이 데이터의 변화를 얼마나 설명하는지 나타냅니다. 일반적으로 1에 가까울수록 좋습니다.")

# RMSE 비교 그래프
fig4, ax4 = plt.subplots(figsize=(9, 4))
ax4.bar(results_df["모델"], results_df["RMSE"])
ax4.set_title("모델별 RMSE 비교 (낮을수록 좋음)")
ax4.set_ylabel("RMSE (kWh)")
ax4.tick_params(axis="x", rotation=12)
ax4.grid(axis="y", alpha=0.3)
st.pyplot(fig4)
plt.close(fig4)


# ---------------------------------------------------------
# 7. 선택 모델의 실제값과 예측값 비교
# ---------------------------------------------------------
st.subheader("실제 발전량과 예측 발전량 비교")

selected_model = st.selectbox(
    "비교할 모델 선택",
    list(models.keys()),
    index=list(models.keys()).index(best_model_name)
)

plot_count = min(300, len(test_df))
compare_df = pd.DataFrame({
    "실제 발전량": y_test.iloc[:plot_count].to_numpy(),
    "예측 발전량": predictions[selected_model][:plot_count]
})

fig5, ax5 = plt.subplots(figsize=(12, 4))
ax5.plot(compare_df.index, compare_df["실제 발전량"], label="실제 발전량", linewidth=1.5)
ax5.plot(compare_df.index, compare_df["예측 발전량"], label="예측 발전량", linewidth=1.5)
ax5.set_title(f"{selected_model}: 실제값과 예측값 비교 (테스트 데이터 일부)")
ax5.set_xlabel("테스트 데이터 순서")
ax5.set_ylabel("발전량 (kWh)")
ax5.legend()
ax5.grid(True, alpha=0.3)
st.pyplot(fig5)
plt.close(fig5)

# 랜덤 포레스트 변수 중요도
st.subheader("랜덤 포레스트의 변수 중요도")
rf_model = models["랜덤 포레스트"]
importance_df = pd.DataFrame({
    "변수": features,
    "중요도": rf_model.feature_importances_
}).sort_values("중요도", ascending=True)

fig6, ax6 = plt.subplots(figsize=(8, 4))
ax6.barh(importance_df["변수"], importance_df["중요도"])
ax6.set_title("랜덤 포레스트 변수 중요도")
ax6.set_xlabel("중요도")
ax6.grid(axis="x", alpha=0.3)
st.pyplot(fig6)
plt.close(fig6)

st.subheader("모델 분석 시 주의할 점")
st.write(
    "이번 모델은 기온과 시간·계절 정보만 사용합니다. "
    "실제 태양광 발전량은 일사량, 구름량, 강수량, 패널 방향과 상태 등에도 영향을 받습니다. "
    "따라서 모델의 예측 결과는 변수 사이의 관계를 탐색하는 자료이며, "
    "기온이 발전량의 원인이라고 증명하는 결과는 아닙니다."
)

st.caption("데이터 출처: 한국수력원자력 한빛 시간별 태양광 발전량·기온 데이터")
