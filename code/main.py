"""
☀️ 태양광 발전량은 정말 기온이 가장 높은 시간대에 가장 많을까?
- 데이터: 한국남동발전 시간대별 태양광 발전실적 CSV
- 기온: Open-Meteo 자동 조회 또는 기상청 CSV 업로드 (발전 CSV에는 기온이 없음)

실행:  pip install streamlit pandas altair requests
       streamlit run solar_temp_app.py
"""
import io
import re
from datetime import timedelta
from pathlib import Path

import altair as alt
import pandas as pd
import requests
import streamlit as st

DEFAULT_CSV = "한국남동발전_시간대별_태양광_발전실적.csv"

# 기온 조회용 대표 지점 (위도, 경도)
LOCATIONS = {
    "구미": (36.12, 128.34),
    "예천": (36.65, 128.45),
    "창원 (두산엔진MG)": (35.23, 128.68),
    "영동": (36.17, 127.78),
    "사천 (삼천포)": (34.93, 128.07),
    "광양": (34.94, 127.70),
    "영흥 (인천)": (37.25, 126.45),
    "여수": (34.76, 127.66),
    "진주 (경상대)": (35.15, 128.10),
    "고흥": (34.60, 127.28),
    "서울": (37.57, 126.98),
}
PLANT_TO_LOC = [
    ("두산", "창원 (두산엔진MG)"), ("구미", "구미"), ("예천", "예천"), ("영동", "영동"),
    ("삼천포", "사천 (삼천포)"), ("광양", "광양"), ("영흥", "영흥 (인천)"),
    ("여수", "여수"), ("경상대", "진주 (경상대)"), ("고흥", "고흥"),
]


# ───────────────────────── 데이터 로딩 ─────────────────────────
def read_csv_bytes(raw: bytes) -> pd.DataFrame:
    for enc in ("utf-8-sig", "cp949"):
        try:
            return pd.read_csv(io.BytesIO(raw), encoding=enc, skipinitialspace=True, index_col=False)
        except UnicodeDecodeError:
            continue
    raise ValueError("CSV 인코딩을 읽을 수 없습니다 (UTF-8 / CP949 지원).")


@st.cache_data(show_spinner=False)
def load_generation(raw: bytes) -> pd.DataFrame:
    """가로형(1시~24시) → 세로형(발전구분, 호기, 일자, hour, kwh)"""
    df = read_csv_bytes(raw)
    df.columns = [str(c).strip() for c in df.columns]
    hour_cols = {c: int(re.match(r"^(\d+)시", c).group(1))
                 for c in df.columns if re.match(r"^\d+시 발전량", c)}
    if not hour_cols:
        raise ValueError("'1시 발전량(KWh)' 형태의 시간대별 컬럼을 찾지 못했습니다.")
    df["일자"] = pd.to_datetime(df["일자"], errors="coerce").dt.normalize()
    df["발전구분"] = df["발전구분"].astype(str).str.strip()
    long = df.melt(id_vars=["발전구분", "호기", "일자"], value_vars=list(hour_cols),
                   var_name="col", value_name="kwh")
    long["hour"] = long["col"].map(hour_cols)
    long["kwh"] = pd.to_numeric(long["kwh"], errors="coerce").fillna(0.0)
    return long.drop(columns="col").dropna(subset=["일자"])


def parse_temp_csv(raw: bytes) -> pd.DataFrame:
    """기상청 등에서 받은 시간별 기온 CSV → (ts, temp)"""
    df = read_csv_bytes(raw)
    df.columns = [str(c).strip() for c in df.columns]
    time_col = next((c for c in df.columns
                     if any(k in c.lower() for k in ("일시", "시간", "날짜", "time", "date"))), None)
    temp_col = next((c for c in df.columns
                     if "기온" in c and not any(k in c for k in ("최고", "최저", "체감"))), None) \
        or next((c for c in df.columns if "temp" in c.lower()), None)
    if not time_col or not temp_col:
        raise ValueError(f"시각/기온 컬럼을 찾지 못했습니다. 현재 컬럼: {list(df.columns)}")
    out = pd.DataFrame({"ts": pd.to_datetime(df[time_col], errors="coerce"),
                        "temp": pd.to_numeric(df[temp_col], errors="coerce")}).dropna()
    return out.groupby("ts", as_index=False)["temp"].mean()  # 지점이 여러 개면 평균


@st.cache_data(show_spinner=False, ttl=3600)
def fetch_open_meteo(lat: float, lon: float, start: str, end: str) -> pd.DataFrame:
    r = requests.get(
        "https://archive-api.open-meteo.com/v1/archive",
        params=dict(latitude=lat, longitude=lon, start_date=start, end_date=end,
                    hourly="temperature_2m,shortwave_radiation", timezone="Asia/Seoul"),
        timeout=30,
    )
    r.raise_for_status()
    h = r.json()["hourly"]
    return pd.DataFrame({"ts": pd.to_datetime(h["time"]),
                         "temp": h["temperature_2m"],
                         "rad": h["shortwave_radiation"]})


# ───────────────────────── 분석 함수 ─────────────────────────
def aggregate(long: pd.DataFrame, plants: list, start, end):
    """선택 발전소를 합산. 선택한 발전소가 모두 자료를 가진 날짜만 사용."""
    sub = long[long["발전구분"].isin(plants) & long["일자"].between(pd.Timestamp(start), pd.Timestamp(end))]
    cnt = sub.groupby("일자")["발전구분"].nunique()
    full_days = cnt[cnt == len(plants)].index
    sub = sub[sub["일자"].isin(full_days)]
    hourly = sub.groupby(["일자", "hour"], as_index=False)["kwh"].sum()
    # 'N시' = (N-1)~N시 구간 발전량으로 보고 N:00 시점 기온과 매칭
    hourly["ts"] = hourly["일자"] + pd.to_timedelta(hourly["hour"], unit="h")
    return hourly, len(full_days)


def peak_hours(w: pd.DataFrame, col: str) -> pd.Series:
    w = w.dropna(subset=[col])
    return w.loc[w.groupby("일자")[col].idxmax()].set_index("일자")["hour"]


def corr_safe(a: pd.Series, b: pd.Series, method="pearson"):
    return a.corr(b, method=method) if len(a) > 2 and a.std() > 0 and b.std() > 0 else float("nan")


# ───────────────────────── 화면 ─────────────────────────
def main():
    st.set_page_config(page_title="태양광 발전량 vs 기온", page_icon="☀️", layout="wide")
    st.title("☀️ 태양광 발전량은 정말 기온이 가장 높은 시간대에 가장 많을까?")
    st.caption("한국남동발전 시간대별 태양광 발전실적 · 발전량이 가장 많은 시각과 기온이 가장 높은 시각을 비교합니다.")

    # ── 사이드바: 데이터 선택
    with st.sidebar:
        st.header("1. 발전 데이터")
        up = st.file_uploader("발전실적 CSV", type="csv")
        local = Path(__file__).parent / DEFAULT_CSV
        if up is not None:
            raw = up.getvalue()
        elif local.exists():
            raw = local.read_bytes()
            st.caption(f"기본 파일 사용: {DEFAULT_CSV}")
        else:
            st.info("발전실적 CSV를 업로드하세요.")
            st.stop()
        try:
            long = load_generation(raw)
        except Exception as e:
            st.error(f"발전 데이터를 읽지 못했습니다: {e}")
            st.stop()

        days_total = long["일자"].nunique()
        coverage = long.groupby("발전구분")["일자"].nunique().sort_values(ascending=False)
        default_plants = list(coverage[coverage == days_total].index) or list(coverage.index[:1])
        plants = st.multiselect(
            "발전소 (여러 개 선택 시 합산)", list(coverage.index), default=default_plants,
            format_func=lambda p: f"{p} ({coverage[p]}일)",
            help="일부 발전소는 9/27~9/30 4일치만 있습니다. 기본값은 전체 기간 자료가 있는 발전소입니다.",
        )
        dmin, dmax = long["일자"].min().date(), long["일자"].max().date()
        dr = st.date_input("기간", (dmin, dmax), min_value=dmin, max_value=dmax)
        start, end = (dr[0], dr[1]) if isinstance(dr, tuple) and len(dr) == 2 else (dmin, dmax)
        h0, h1 = st.slider("분석 시간대 (시)", 1, 24, (8, 18),
                           help="야간·새벽에 0이 아닌 값이 일부 있어(계측/집계 이상 추정) 낮 시간대만 비교하는 것이 안전합니다.")

        st.header("2. 기온 데이터")
        src = st.radio("기온 가져오기", ["Open-Meteo 자동 조회", "기상청 CSV 업로드", "사용 안 함"])
        temp_df, temp_note = None, None
        if src == "Open-Meteo 자동 조회":
            names = list(LOCATIONS)
            guess = next((loc for key, loc in PLANT_TO_LOC if plants and key in plants[0]), "구미")
            loc = st.selectbox("기온 기준 지점", names, index=names.index(guess))
        elif src == "기상청 CSV 업로드":
            tup = st.file_uploader("시간별 기온 CSV (일시, 기온 컬럼 포함)", type="csv", key="temp")
            st.caption("기상청 기상자료개방포털 → 종관기상관측(ASOS) → 시간자료")

    if not plants:
        st.warning("발전소를 하나 이상 선택하세요.")
        st.stop()

    hourly, n_days = aggregate(long, plants, start, end)
    if hourly.empty:
        st.warning("선택한 조건에 해당하는 데이터가 없습니다. 발전소/기간을 바꿔 보세요.")
        st.stop()

    # ── 기온 불러오기
    if src == "Open-Meteo 자동 조회":
        lat, lon = LOCATIONS[loc]
        try:
            temp_df = fetch_open_meteo(lat, lon, start.isoformat(), (end + timedelta(days=1)).isoformat())
            temp_note = f"Open-Meteo 재분석 자료 · {loc} 인근 격자 · 시간별 2m 기온"
        except Exception as e:
            st.sidebar.error(f"기온 조회 실패: {e}\n\n기상청 CSV 업로드를 이용해 주세요.")
    elif src == "기상청 CSV 업로드" and tup is not None:
        try:
            temp_df = parse_temp_csv(tup.getvalue())
            temp_note = "업로드한 기온 CSV (지점이 여러 개면 평균)"
        except Exception as e:
            st.sidebar.error(f"기온 파일 오류: {e}")

    merged = hourly.merge(temp_df, on="ts", how="left") if temp_df is not None else hourly.assign(temp=pd.NA)
    if temp_df is not None:
        merged["temp"] = pd.to_numeric(merged["temp"])
    w = merged[merged["hour"].between(h0, h1)].copy()
    has_temp = temp_df is not None and w["temp"].notna().any()

    st.markdown(f"**분석 대상**: {', '.join(plants)} · {n_days}일 · {h0}~{h1}시"
                + (f" · 기온: {temp_note}" if has_temp else ""))

    tab1, tab2, tab3 = st.tabs(["① 시간대별 발전 패턴", "② 기온 vs 발전량", "③ 데이터 확인"])

    # ── 탭1: 기온 없이도 가능한 발전 패턴
    with tab1:
        avg = merged.groupby("hour", as_index=False)["kwh"].mean().rename(columns={"kwh": "평균 발전량(kWh)"})
        gp = peak_hours(w, "kwh")
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("시간대별 평균 발전량")
            st.altair_chart(
                alt.Chart(avg).mark_bar().encode(
                    x=alt.X("hour:O", title="시"), y=alt.Y("평균 발전량(kWh):Q"),
                    tooltip=["hour", alt.Tooltip("평균 발전량(kWh):Q", format=",.1f")]),
                theme="streamlit")
        with c2:
            st.subheader("일별 발전 최대 시각 분포")
            dist = gp.value_counts().sort_index().rename_axis("hour").reset_index(name="일수")
            st.altair_chart(
                alt.Chart(dist).mark_bar(color="#f59e0b").encode(
                    x=alt.X("hour:O", title="발전량이 가장 많았던 시(시)"), y="일수:Q",
                    tooltip=["hour", "일수"]),
                theme="streamlit")
        st.metric("발전량 최대 시각 (중앙값)", f"{gp.median():.0f}시")

    # ── 탭2: 기온 비교
    with tab2:
        if not has_temp:
            st.info("사이드바에서 기온 데이터를 불러오면 비교 결과가 표시됩니다. "
                    "(발전실적 CSV에는 기온 컬럼이 없습니다.)")
        else:
            wt = w.dropna(subset=["temp"])
            gp, tp = peak_hours(wt, "kwh"), peak_hours(wt, "temp")
            pk = pd.concat([gp.rename("gen"), tp.rename("temp")], axis=1).dropna()
            n = len(pk)
            same = int((pk["gen"] == pk["temp"]).sum())
            near = int(((pk["gen"] - pk["temp"]).abs() <= 1).sum())
            lag_med = float((pk["temp"] - pk["gen"]).median())

            m = st.columns(4)
            m[0].metric("발전량 최대 시각(중앙값)", f"{pk['gen'].median():.0f}시")
            m[1].metric("기온 최고 시각(중앙값)", f"{pk['temp'].median():.0f}시")
            m[2].metric("같은 시각에 일치한 날", f"{same}/{n}일", f"{same / n:.0%}", delta_color="off")
            m[3].metric("±1시간 이내 일치", f"{near}/{n}일", f"{near / n:.0%}", delta_color="off")

            if same / n >= 0.7:
                verdict = "대부분의 날에 기온이 가장 높은 시각과 발전량이 가장 많은 시각이 겹칩니다."
            elif lag_med >= 1:
                verdict = (f"기온 최고 시각이 발전량 최대 시각보다 중앙값 기준 {lag_med:.0f}시간 늦습니다. "
                           "즉 '기온이 가장 높은 시간대에 발전량이 가장 많다'고 보기는 어렵습니다.")
            else:
                verdict = "두 피크 시각이 일치하는 날의 비율이 낮아, 발전량 최대가 기온 최고와 같은 시간대라고 보기 어렵습니다."
            st.success(f"**결과 요약** — {verdict}")

            # 정규화 프로파일
            prof = wt.groupby("hour", as_index=False).agg(kwh=("kwh", "mean"), temp=("temp", "mean"))
            rng = prof["temp"].max() - prof["temp"].min()
            prof["발전량 (최대=1)"] = prof["kwh"] / prof["kwh"].max() if prof["kwh"].max() > 0 else 0
            prof["기온 (최저=0, 최고=1)"] = (prof["temp"] - prof["temp"].min()) / rng if rng > 0 else 0
            pl = prof.melt("hour", ["발전량 (최대=1)", "기온 (최저=0, 최고=1)"], "지표", "값")
            st.subheader("시간대별 평균 패턴 (0~1로 정규화해 모양만 비교)")
            st.altair_chart(
                alt.Chart(pl).mark_line(point=True).encode(
                    x=alt.X("hour:O", title="시"), y=alt.Y("값:Q", scale=alt.Scale(domain=[0, 1.05])),
                    color="지표:N", tooltip=["hour", "지표", alt.Tooltip("값:Q", format=".2f")]),
                theme="streamlit")

            st.subheader("피크 시각 차이 (기온 최고 시각 − 발전 최대 시각)")
            diff = (pk["temp"] - pk["gen"]).value_counts().sort_index().rename_axis("차이(시간)").reset_index(name="일수")
            st.altair_chart(
                alt.Chart(diff).mark_bar(color="#ef4444").encode(
                    x=alt.X("차이(시간):O"), y="일수:Q", tooltip=["차이(시간)", "일수"]),
                theme="streamlit")

            # 상관관계
            st.subheader("기온과 발전량의 관계")
            r_p = corr_safe(wt["kwh"], wt["temp"])
            r_s = corr_safe(wt["kwh"], wt["temp"], "spearman")
            cc = st.columns(3 if "rad" in wt else 2)
            cc[0].metric("피어슨 상관계수 (시간 단위)", f"{r_p:.2f}")
            cc[1].metric("스피어만 상관계수", f"{r_s:.2f}")
            if "rad" in wt and wt["rad"].notna().any():
                cc[2].metric("(참고) 일사량과의 상관", f"{corr_safe(wt['kwh'], wt['rad']):.2f}",
                             help="태양광의 직접적인 원인은 기온이 아니라 일사량입니다.")

            sc = alt.Chart(wt).mark_circle(size=60, opacity=0.6).encode(
                x=alt.X("temp:Q", title="기온(℃)", scale=alt.Scale(zero=False)),
                y=alt.Y("kwh:Q", title="발전량(kWh)"),
                color=alt.Color("hour:O", title="시", scale=alt.Scale(scheme="turbo")),
                tooltip=["일자:T", "hour", alt.Tooltip("temp:Q", format=".1f"), alt.Tooltip("kwh:Q", format=",.0f")])
            trend = sc.transform_regression("temp", "kwh").mark_line(color="black")
            st.altair_chart(sc + trend, theme="streamlit")

            cl, cr = st.columns(2)
            with cl:
                st.markdown("**기온 구간별 평균 발전량** (낮 시간대 전체)")
                try:
                    bins = pd.qcut(wt["temp"], 5, duplicates="drop")
                    b = wt.assign(bin=bins).groupby("bin", observed=True).agg(
                        평균발전량=("kwh", "mean"), 시간수=("kwh", "size")).reset_index()
                    b["구간"] = b["bin"].apply(lambda i: f"{i.left:.1f}~{i.right:.1f}℃")
                    st.altair_chart(
                        alt.Chart(b).mark_bar().encode(
                            x=alt.X("구간:N", sort=list(b["구간"])), y=alt.Y("평균발전량:Q", title="평균 발전량(kWh)"),
                            tooltip=["구간", alt.Tooltip("평균발전량:Q", format=",.1f"), "시간수"]),
                        theme="streamlit")
                except ValueError:
                    st.caption("기온 값의 변동이 적어 구간을 나눌 수 없습니다.")
            with cr:
                st.markdown("**같은 시각끼리 비교한 상관계수** (날짜별 기온 ↔ 발전량)")
                rows = [{"hour": h, "상관계수": corr_safe(g["kwh"], g["temp"])} for h, g in wt.groupby("hour")]
                hr = pd.DataFrame(rows).dropna()
                if hr.empty:
                    st.caption("계산할 데이터가 부족합니다.")
                else:
                    st.altair_chart(
                        alt.Chart(hr).mark_bar().encode(
                            x=alt.X("hour:O", title="시"),
                            y=alt.Y("상관계수:Q", scale=alt.Scale(domain=[-1, 1])),
                            tooltip=["hour", alt.Tooltip("상관계수:Q", format=".2f")]),
                        theme="streamlit")

            # 일 단위
            st.subheader("일 단위: 낮 평균기온 ↔ 하루 발전량")
            daily = wt.groupby("일자", as_index=False).agg(발전량=("kwh", "sum"), 평균기온=("temp", "mean"))
            ds = alt.Chart(daily).mark_circle(size=70, opacity=0.7).encode(
                x=alt.X("평균기온:Q", title="낮 평균기온(℃)", scale=alt.Scale(zero=False)),
                y=alt.Y("발전량:Q", title="분석 시간대 발전량 합계(kWh)"),
                tooltip=["일자:T", alt.Tooltip("평균기온:Q", format=".1f"), alt.Tooltip("발전량:Q", format=",.0f")])
            st.altair_chart(ds + ds.transform_regression("평균기온", "발전량").mark_line(color="black"),
                            theme="streamlit")
            st.caption(f"일 단위 상관계수: {corr_safe(daily['발전량'], daily['평균기온']):.2f}")

            with st.expander("해석할 때 주의할 점"):
                st.markdown(
                    "- **상관관계는 인과관계가 아닙니다.** 태양광 발전량은 기본적으로 일사량에 비례하고, "
                    "기온은 일사로 데워진 공기가 만든 결과라 함께 움직일 뿐입니다.\n"
                    "- 일반적으로 하루 중 기온 최고점은 태양이 가장 높은 시각(정오 무렵)보다 늦게 나타나므로 "
                    "두 피크가 어긋나는 것이 자연스럽습니다. 또한 패널 온도가 오르면 변환 효율은 오히려 "
                    "조금 떨어집니다.\n"
                    "- 발전량은 날씨(구름), 설비 용량, 출력 제한 등의 영향을 받습니다. "
                    "발전소를 합산하면 설비 용량이 큰 곳이 결과를 좌우합니다.\n"
                    "- 조회한 기온은 발전소 위치가 아닌 대표 지점(또는 격자) 값이며, "
                    "'N시' 발전량은 N:00 시점의 기온과 맞춰 비교했습니다."
                )

    # ── 탭3: 데이터 확인
    with tab3:
        st.subheader("분석에 사용된 시간별 데이터")
        show = merged.drop(columns=["ts"]).rename(columns={"hour": "시", "kwh": "발전량(kWh)", "temp": "기온(℃)"})
        st.dataframe(show)
        st.download_button("CSV로 내려받기", show.to_csv(index=False).encode("utf-8-sig"),
                           "태양광_기온_분석데이터.csv", "text/csv")
        st.subheader("발전소별 자료 일수")
        st.dataframe(coverage.rename("자료 일수").reset_index())
        st.caption("데이터 참고: 여수·탑선 일부 행의 24시 값, 영동·삼천포 일부 행의 19~20시 값처럼 "
                   "야간에 0이 아닌 값이 있습니다. 기본 분석 시간대(8~18시)에는 포함되지 않습니다.")


if __name__ == "__main__":
    main()
