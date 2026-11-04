import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import re
import os
from io import BytesIO

# ==========================================
# 1. 기본 설정
# ==========================================

st.set_page_config(
    page_title="태양광 발전량 분석 프로젝트",
    page_icon="☀️",
    layout="wide"
)

st.title("☀️ 태양광 발전량 분석 프로젝트")
st.subheader("태양광 발전량은 정말 온도가 가장 높은 시간대에 가장 많을까?")
st.caption("한국남동발전 시간대별 태양광 발전실적 데이터 분석")

st.markdown("""
이 프로젝트에서는 태양광 발전소의 시간대별 발전량을 분석하여
발전량이 가장 많은 시간대가 언제인지 알아봅니다.
""")

# ==========================================
# 2. 데이터 불러오기
# ==========================================

FILE_NAME = "한국남동발전_시간대별_태양광_발전실적(2).xls"

@st.cache_data
def load_data(file_bytes=None, file_url=None):

    if file_bytes is not None:
        data = pd.read_excel(BytesIO(file_bytes))

    elif file_url:
        if file_url.endswith(".csv"):
            data = pd.read_csv(file_url, encoding="utf-8-sig")
        else:
            data = pd.read_excel(file_url)

    elif os.path.exists(FILE_NAME):
        data = pd.read_excel(FILE_NAME)

    else:
        return None

    return data


# ==========================================
# 3. 사이드바
# ==========================================

st.sidebar.title("⚙️ 분석 설정")

uploaded_file = st.sidebar.file_uploader(
    "태양광 발전 데이터 업로드",
    type=["xls", "xlsx", "csv"]
)

file_url = st.sidebar.text_input(
    "또는 데이터 URL 입력",
    placeholder="GitHub Raw 데이터 주소"
)

try:
    if uploaded_file is not None:
        raw = load_data(uploaded_file.getvalue())

    elif file_url.strip():
        raw = load_data(file_url=file_url.strip())

    else:
        raw = load_data()

except Exception as e:
    st.error(f"데이터를 불러오는 중 오류가 발생했습니다: {e}")
    st.info("XLS 파일을 읽으려면 requirements.txt에 xlrd가 포함되어야 합니다.")
    st.stop()

if raw is None:
    st.warning("데이터 파일을 업로드하거나 GitHub Raw URL을 입력해 주세요.")
    st.stop()

# 원본 데이터 보관
df = raw.copy()

# ==========================================
# 4. 데이터 개요
# ==========================================

st.divider()
st.header("📊 1. 데이터 개요")

row_count = len(df)
col_count = len(df.columns)
missing_total = int(df.isna().sum().sum())

c1, c2, c3 = st.columns(3)

c1.metric("전체 행 개수", f"{row_count:,}개")
c2.metric("전체 열 개수", f"{col_count:,}개")
c3.metric("전체 결측치", f"{missing_total:,}개")

st.markdown("#### 🔎 열별 자료형 및 결측치 확인")

info = pd.DataFrame({
    "열 이름": df.columns,
    "자료형": [str(x) for x in df.dtypes],
    "빈 값 개수": df.isna().sum().values,
    "고유값 개수": df.nunique().values
})

st.dataframe(info, use_container_width=True, hide_index=True)

with st.expander("원본 데이터 미리 보기"):
    st.dataframe(df.head(20), use_container_width=True)

# ==========================================
# 5. 데이터 전처리
# ==========================================

st.divider()
st.header("🧹 2. 데이터 전처리")

before_rows = len(df)

# 시간대별 발전량 열 찾기
hour_cols = [
    c for c in df.columns
    if re.match(r"^\d+시 발전량", str(c))
]

if not hour_cols:
    st.error("시간대별 발전량 열을 찾을 수 없습니다.")
    st.stop()

# 발전소 이름과 날짜 열 확인
plant_col = "발전구분"
date_col = "일자"

if plant_col not in df.columns or date_col not in df.columns:
    st.error("발전구분 또는 일자 열을 찾을 수 없습니다.")
    st.stop()

# 날짜 변환
df[date_col] = pd.to_datetime(
    df[date_col], errors="coerce"
)

# 발전량 열 숫자형 변환
for col in hour_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# 필수 정보가 없는 행 제거
df = df.dropna(subset=[plant_col, date_col])

# 결측치 처리
numeric_cols = df.select_dtypes(include="number").columns.tolist()

for col in numeric_cols:
    if df[col].isna().sum() > 0:
        df[col] = df.groupby(plant_col)[col].transform(
            lambda x: x.fillna(x.median())
        )

        df[col] = df[col].fillna(df[col].median())

# 전처리 후 행 수
after_rows = len(df)

# ==========================================
# 6. 이상치 확인 (IQR 방식)
# ==========================================

outlier_count = 0

for plant, group in df.groupby(plant_col):
    for col in hour_cols:

        q1 = group[col].quantile(0.25)
        q3 = group[col].quantile(0.75)
        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        outlier_count += int(
            ((group[col] < lower) | (group[col] > upper)).sum()
        )

# 이상치 후보는 자동 삭제하지 않음
# 발전소별 설비 규모 차이를 고려하여 원본 값 유지

st.markdown("#### 🔍 전처리 기준")

st.markdown("""
| 항목 | 처리 방법 | 처리 기준 |
|---|---|---|
| 결측치 | 필수 정보가 없는 행 제거 | 발전소명과 날짜가 없는 경우 |
| 수치형 결측치 | 발전소별 중앙값으로 대체 | 발전량 등의 값이 비어 있는 경우 |
| 이상치 | IQR 방식으로 후보 확인 | 발전소별·시간대별 1.5×IQR 기준 |
| 야간 발전량 0 | 유지 | 실제 발전량이 없는 정상적인 값 |
| 큰 발전량 | 유지 | 발전소별 설비 규모 차이 고려 |
""")

st.caption(
    "IQR로 발견한 값은 이상치 후보이며, 실제 오류라고 단정할 수 없습니다. "
    "따라서 임의로 삭제하지 않았습니다."
)

st.markdown("#### 📈 전처리 전후 비교")

a, b, c = st.columns(3)

a.metric("전처리 전 행 수", f"{before_rows:,}개")
b.metric("전처리 후 행 수", f"{after_rows:,}개")
c.metric("IQR 이상치 후보", f"{outlier_count:,}개")

comparison = pd.DataFrame({
    "구분": ["전처리 전", "전처리 후"],
    "행 개수": [before_rows, after_rows]
})

fig_compare = px.bar(
    comparison,
    x="구분",
    y="행 개수",
    text="행 개수",
    color="구분",
    title="전처리 전후 행 개수 비교"
)

fig_compare.update_layout(
    xaxis_title="전처리 단계",
    yaxis_title="행 개수",
    showlegend=False
)

st.plotly_chart(fig_compare, use_container_width=True)

# ==========================================
# 7. 분석 기간 및 발전소 선택
# ==========================================

st.divider()
st.header("📅 3. 분석 조건 설정")

min_date = df[date_col].min().date()
max_date = df[date_col].max().date()

selected_dates = st.date_input(
    "분석할 날짜 범위",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

if isinstance(selected_dates, tuple) and len(selected_dates) == 2:
    start_date, end_date = selected_dates

    filtered = df[
        (df[date_col].dt.date >= start_date) &
        (df[date_col].dt.date <= end_date)
    ].copy()
else:
    filtered = df.copy()

plants = sorted(filtered[plant_col].dropna().unique())

selected_plants = st.multiselect(
    "비교할 발전소 선택",
    plants,
    default=plants
)

if not selected_plants:
    st.warning("발전소를 하나 이상 선택해 주세요.")
    st.stop()

filtered = filtered[
    filtered[plant_col].isin(selected_plants)
].copy()

if filtered.empty:
    st.warning("선택한 조건에 해당하는 데이터가 없습니다.")
    st.stop()

# ==========================================
# 8. 시간대별 전체 발전량 분석
# ==========================================

st.divider()
st.header("📈 4. 그래프 1 - 시간대별 전체 발전량")

hour_labels = [
    int(re.search(r"(\d+)시", c).group(1))
    for c in hour_cols
]

hourly_total = filtered[hour_cols].sum()

hourly_data = pd.DataFrame({
    "시간": hour_labels,
    "발전량": hourly_total.values
})

hourly_data = hourly_data.sort_values("시간")

peak_row = hourly_data.loc[
    hourly_data["발전량"].idxmax()
]

peak_hour = int(peak_row["시간"])
peak_value = float(peak_row["발전량"])

fig1 = px.line(
    hourly_data,
    x="시간",
    y="발전량",
    markers=True,
    title="☀️ 시간대별 전체 태양광 발전량",
    text=None
)

fig1.add_trace(
    go.Scatter(
        x=[peak_hour],
        y=[peak_value],
        mode="markers+text",
        name="최대 발전 시간",
        marker=dict(size=15, color="red"),
        text=[f"최대 {peak_hour}시"],
        textposition="top center"
    )
)

fig1.update_layout(
    xaxis_title="시간",
    yaxis_title="전체 발전량 (kWh)",
    xaxis=dict(dtick=1),
    hovermode="x unified"
)

st.plotly_chart(fig1, use_container_width=True)

st.success(
    f"💡 발견한 사실: {peak_hour}시에 전체 발전량이 가장 많았다."
)

st.markdown(
    f"**근거 수치:** {peak_hour}시 발전량은 "
    f"약 {peak_value:,.2f}kWh로, 분석한 시간대 중 가장 높았다."
)

# ==========================================
# 9. 발전소별 시간대 발전량 비교
# ==========================================

st.divider()
st.header("📊 5. 그래프 2 - 발전소별 발전량 비교")

plant_hourly = filtered.groupby(plant_col)[hour_cols].mean()

plant_long = plant_hourly.reset_index().melt(
    id_vars=plant_col,
    value_vars=hour_cols,
    var_name="시간",
    value_name="평균 발전량"
)

plant_long["시간"] = plant_long["시간"].str.extract(
    r"(\d+)"
).astype(int)

fig2 = px.line(
    plant_long,
    x="시간",
    y="평균 발전량",
    color=plant_col,
    markers=True,
    title="🏭 발전소별 시간대 평균 발전량 비교",
    hover_data={
        plant_col: True,
        "시간": True,
        "평균 발전량": ":,.2f"
    }
)

fig2.update_layout(
    xaxis_title="시간",
    yaxis_title="평균 발전량 (kWh)",
    xaxis=dict(dtick=1),
    hovermode="x unified",
    legend_title="발전소"
)

st.plotly_chart(fig2, use_container_width=True)

# 발전소별 최대 평균 발전 시간
plant_peak = plant_long.loc[
    plant_long.groupby(plant_col)["평균 발전량"].idxmax()
]

st.success(
    "💡 발견한 사실: 발전소마다 평균 발전량의 크기와 "
    "최대 발전 시간대가 다르게 나타날 수 있다."
)

st.markdown("**근거 수치: 발전소별 최대 평균 발전 시간**")

st.dataframe(
    plant_peak[
        [plant_col, "시간", "평균 발전량"]
    ].rename(columns={
        plant_col: "발전소",
        "시간": "최대 발전 시간",
        "평균 발전량": "최대 평균 발전량(kWh)"
    }).sort_values(
        "최대 평균 발전량",
        ascending=False
    ).reset_index(drop=True),
    use_container_width=True,
    hide_index=True
)

# ==========================================
# 10. 시간대별 발전량 순위
# ==========================================

st.divider()
st.header("🏆 6. 시간대별 발전량 순위")

rank_data = hourly_data.sort_values(
    "발전량",
    ascending=False
).copy()

rank_data["순위"] = range(1, len(rank_data) + 1)

fig3 = px.bar(
    rank_data,
    x="발전량",
    y="시간",
    orientation="h",
    color="발전량",
    text="발전량",
    title="시간대별 발전량 TOP 10"
)

fig3.update_layout(
    xaxis_title="전체 발전량 (kWh)",
    yaxis_title="시간",
    yaxis=dict(
        autorange="reversed",
        dtick=1
    )
)

st.plotly_chart(fig3, use_container_width=True)

top3 = rank_data.head(3)

st.info(
    f"💡 발견한 사실: 발전량이 가장 많은 상위 3개 시간대는 "
    f"{', '.join(str(x) + '시' for x in top3['시간'])}이다."
)

# ==========================================
# 11. 최종 분석 및 주의점
# ==========================================

st.divider()
st.header("🔍 7. 최종 분석 결과")

st.markdown(f"""
### ☀️ 우리 팀이 발견한 사실

- 분석 기간의 최대 발전 시간대는 **{peak_hour}시**이다.
- 이 시간의 전체 발전량은 **{peak_value:,.2f}kWh**이다.
- 발전소마다 발전량의 크기와 시간대별 변화 양상이 다르게 나타난다.
- 발전량이 높은 시간대가 반드시 기온이 가장 높은 시간대와 일치한다고 볼 수는 없다.

### ⚠️ 분석의 한계

현재 데이터에는 시간대별 기온 정보가 포함되어 있지 않다.

따라서 이번 분석으로는 발전량이 가장 많은 시간대를 확인할 수 있지만,
기온이 가장 높은 시간대와 정확히 일치하는지는 판단할 수 없다.

이를 확인하려면 같은 날짜와 지역의 시간대별 기온 데이터를 추가하여
발전량과 기온을 직접 비교해야 한다.
""")

st.divider()

st.markdown("""
### 📝 프로젝트 결론

태양광 발전량은 햇빛의 양, 날씨, 발전소의 설비 규모 등
여러 요인의 영향을 받는다.

이번 데이터에서는 시간대별 발전량의 차이를 확인할 수 있었으며,
기온이 가장 높은 시간대와 발전량이 가장 많은 시간대가
일치하는지 확인하기 위해서는 추가적인 기온 데이터 분석이 필요하다.
""")

st.caption("한국남동발전 시간대별 태양광 발전실적 데이터 분석 프로젝트")
