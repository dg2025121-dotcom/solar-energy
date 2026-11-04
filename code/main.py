st.markdown("**전처리 전후 행 수**")
process_table = pd.DataFrame({
    "처리 단계": [
        "원본 데이터",
        "필수 열 자료형 변환 후",
        "결측치 제거 후",
        "중복 기록 제거 후",
        "비정상 값 제거 후(최종 분석 데이터)",
    ],
    "행 수": [before, after_conversion, after_missing, after_duplicates, after_cleaning],
    "제거된 행 수": [
        0,
        0,
        missing_removed,
        duplicate_count,
        invalid_count,
    ],
})
st.dataframe(process_table, use_container_width=True, hide_index=True)

st.markdown(
    f"""
    **처리 방법**
    - 결측치: 필수 분석 열에 빈 값이 있는 행을 제거했습니다. 제거 행 수는 **{missing_removed:,}행**입니다.
    - 중복치: 날짜·지역·시간이 모두 같은 중복 기록을 제거했습니다. 제거 행 수는 **{duplicate_count:,}행**입니다.
    - 비정상 값: 시간 1~24 범위 밖, 음수 발전량, 기온 -50℃ 미만 또는 60℃ 초과인 행을 제거했습니다. 제거 행 수는 **{invalid_count:,}행**입니다.
    - 통계적 이상치: IQR 기준으로 발전량 **{outlier_counts['총계량전력량']:,}개**, 기온 **{outlier_counts['기온']:,}개**가 탐지되었습니다. 다만 발전량의 높은 값은 실제 발전량 최고치일 수 있어 삭제하지 않고 분석에 유지했습니다.
    """
)

if clean.empty:
    st.error("전처리 후 분석할 데이터가 없습니다.")
    st.stop()

# ── 3. 시간대별 그래프 ────────────────────────────────────────
st.header("3. 탐구 질문과 연결된 그래프")

hourly = (
    clean.groupby("시간", as_index=False)
    .agg(
        평균_발전량=("총계량전력량", "mean"),
        평균_기온=("기온", "mean"),
        자료_개수=("시간", "size"),
    )
    .sort_values("시간")
)

power_peak = hourly.loc[hourly["평균_발전량"].idxmax()]
temp_peak = hourly.loc[hourly["평균_기온"].idxmax()]

left, right = st.columns(2)
with left:
    st.subheader("그래프 1. 시간대별 평균 태양광 발전량")
    fig_power = px.line(
        hourly, x="시간", y="평균_발전량", markers=True,
        labels={"시간": "시간(시)", "평균_발전량": "평균 발전량(kWh)"},
        title="시간대별 평균 태양광 발전량"
    )
    fig_power.update_layout(xaxis=dict(dtick=1))
    st.plotly_chart(fig_power, use_container_width=True)
    st.info(
        f"**발견:** 평균 발전량은 {int(power_peak['시간'])}시에 가장 높았습니다. "
        f"근거: 시간대 평균 {power_peak['평균_발전량']:,.1f} kWh."
    )

with right:
    st.subheader("그래프 2. 시간대별 평균 기온")
    fig_temp = px.line(
        hourly, x="시간", y="평균_기온", markers=True,
        labels={"시간": "시간(시)", "평균_기온": "평균 기온(℃)"},
        title="시간대별 평균 기온"
    )
    fig_temp.update_layout(xaxis=dict(dtick=1))
    st.plotly_chart(fig_temp, use_container_width=True)
    st.info(
        f"**발견:** 평균 기온은 {int(temp_peak['시간'])}시에 가장 높았습니다. "
        f"근거: 시간대 평균 {temp_peak['평균_기온']:.2f}℃."
    )

st.subheader("그래프 3. 기온과 태양광 발전량의 관계")
# 야간의 발전량 0이 상관관계 해석을 왜곡하지 않도록 7~19시 자료를 별도로 살펴봄
daytime = clean[clean["시간"].between(7, 19)].copy()
corr = daytime["기온"].corr(daytime["총계량전력량"]) if len(daytime) > 1 else np.nan

fig_scatter = px.scatter(
    daytime,
    x="기온",
    y="총계량전력량",
    opacity=0.35,
    labels={"기온": "기온(℃)", "총계량전력량": "태양광 발전량(kWh)"},
    title="주간 시간대(7~19시)의 기온과 태양광 발전량"
)
st.plotly_chart(fig_scatter, use_container_width=True)
if pd.notna(corr):
    st.info(
        f"**발견:** 주간 시간대 기온과 발전량의 피어슨 상관계수는 **{corr:.3f}**입니다. "
        "상관계수는 두 값의 선형 관계를 나타내며, 기온이 발전량의 원인이라고 단정할 수는 없습니다."
    )

# ── 4. 질문에 대한 결론 ───────────────────────────────────────
st.header("4. 탐구 질문에 대한 결과")
difference = abs(int(power_peak["시간"]) - int(temp_peak["시간"]))
if int(power_peak["시간"]) == int(temp_peak["시간"]):
    st.success(
        f"이 자료에서는 평균 발전량 최고 시간과 평균 기온 최고 시간이 모두 "
        f"{int(power_peak['시간'])}시로 일치합니다."
    )
else:
    st.warning(
        f"이 자료에서는 평균 발전량 최고 시간은 {int(power_peak['시간'])}시, "
        f"평균 기온 최고 시간은 {int(temp_peak['시간'])}시로 "
        f"{difference}시간 차이가 납니다. 따라서 두 최고 시간은 일치하지 않습니다."
    )

st.markdown(
    """
    **해석할 때 주의할 점:** 태양광 발전량은 기온뿐 아니라 일사량, 구름량, 계절,
    태양의 고도 등 여러 요인의 영향을 받습니다. 따라서 이 분석은 기온과 발전량의
    관계를 살펴보는 것이며, 기온만으로 발전량을 설명하거나 인과관계를 증명하는 것은 아닙니다.
    """
)

st.caption("자료 출처: 한국수력원자력 한빛 시간별 태양광 발전량·기온 CSV (GitHub 원본 링크)")
