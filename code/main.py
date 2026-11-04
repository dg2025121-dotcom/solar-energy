"""
☀️ 태양광 발전량은 정말 기온이 가장 높은 시간대에 가장 많을까?
- 데이터: 한국남동발전 시간대별 태양광 발전실적 CSV (GitHub 원본 주소에서 자동 로드, 업로드도 가능)
- 기온: Open-Meteo 자동 조회 또는 기상청 CSV 업로드 (발전 CSV에는 기온이 없음)

실행:  pip install streamlit pandas altair requests
       streamlit run solar_temp_app.py
"""
import io
import re
from datetime import timedelta
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import requests
import streamlit as st

DATA_URL = (
    "https://raw.githubusercontent.com/dg2025121-dotcom/solar-energy/refs/heads/main/data/"
    "%ED%95%9C%EA%B5%AD%EB%82%A8%EB%8F%99%EB%B0%9C%EC%A0%84_%EC%8B%9C%EA%B0%84%EB%8C%80%EB%B3%84_"
    "%ED%83%9C%EC%96%91%EA%B4%91_%EB%B0%9C%EC%A0%84%EC%8B%A4%EC%A0%81.csv"
)
DEFAULT_CSV = "한국남동발전_시간대별_태양광_발전실적.csv"  # 같은 폴더에 있으면 GitHub 실패 시 대체 사용
NIGHT_HOURS = {1, 2, 3, 4, 5, 21, 22, 23, 24}  # 태양광 발전이 있을 수 없는 시간대(9~10월 기준)

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


# ───────────────────────── 데이터 로딩·전처리 ─────────────────────────
def read_csv_bytes(raw: bytes) -> pd.DataFrame:
    for enc in ("utf-8-sig", "cp949"):
        try:
            return pd.read_csv(io.BytesIO(raw), encoding=enc, skipinitialspace=True, index_col=False)
        except UnicodeDecodeError:
            continue
    raise ValueError("CSV 인코딩을 읽을 수 없습니다 (UTF-8 / CP949 지원).")


@st.cache_data(show_spinner="GitHub에서 발전 데이터를 불러오는 중...", ttl=3600)
def fetch_url(url: str) -> bytes:
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.content


@st.cache_data(show_spinner=False)
def prepare(raw: bytes):
    """원본 → (원본 표, 세로형 정제 데이터, 전처리 기록)"""
    df = read_csv_bytes(raw)
    df.columns = [str(c).strip() for c in df.columns]
    raw_df = df.copy()
    hour_cols = {c: int(re.match(r"^(\d+)시", c).group(1))
                 for c in df.columns if re.match(r"^\d+시 발전량", c)}
    if not hour_cols:
        raise ValueError("'1시 발전량(KWh)' 형태의 시간대별 컬럼을 찾지 못했습니다.")
    cols = list(hour_cols)
    log = {"n_raw": len(df), "n_cols": df.shape[1]}

    # 검증: 시간별 합 = 총량 인지 (원자료 정합성)
    tot_col = next((c for c in df.columns if c.startswith("총량")), None)
    if tot_col:
        s = df[cols].apply(pd.to_numeric, errors="coerce").fillna(0).sum(axis=1)
        log["n_total_mismatch"] = int(((s - pd.to_numeric(df[tot_col], errors="coerce")).abs() > 1).sum())
    else:
        log["n_total_mismatch"] = None

    # 1) 일자 오류 행 제거 → 2) 중복 행 제거
    df["일자"] = pd.to_datetime(df["일자"], errors="coerce").dt.normalize()
    df["발전구분"] = df["발전구분"].astype(str).str.strip()
    n_a = len(df)
    df = df.dropna(subset=["일자"])
    log["n_bad_date"] = n_a - len(df)
    log["n_after_date"] = len(df)
    df = df.drop_duplicates(["발전구분", "호기", "일자"], keep="first")
    log["n_dup"] = log["n_after_date"] - len(df)
    log["n_wide"] = len(df)

    # 3) 셀 단위 처리
    vals = df[cols].apply(pd.to_numeric, errors="coerce")
    log["n_cells"] = int(vals.size)
    log["n_missing"] = int(vals.isna().sum().sum())
    vals = vals.fillna(0.0)
    log["n_neg"] = int((vals < 0).sum().sum())
    vals = vals.clip(lower=0.0)
    night_cols = [c for c, h in hour_cols.items() if h in NIGHT_HOURS]
    night_mask = vals[night_cols] > 0
    log["n_night"] = int(night_mask.sum().sum())
    log["night_plants"] = df.loc[night_mask.any(axis=1), "발전구분"].value_counts().head(5).to_dict()
    vals[night_cols] = vals[night_cols].where(~night_mask, 0.0)

    clean = pd.concat([df[["발전구분", "호기", "일자"]], vals], axis=1)
    long = clean.melt(id_vars=["발전구분", "호기", "일자"], value_vars=cols,
                      var_name="col", value_name="kwh")
    long["hour"] = long["col"].map(hour_cols)
    long = long.drop(columns="col")

    # 4) 통계적 이상치 탐지(제거하지 않고 표시만): 발전기별 양수값의 Tukey 3×IQR 초과
    pos = long[long["kwh"] > 0]
    g = pos.groupby(["발전구분", "호기"])["kwh"]
    q1, q3 = g.transform(lambda x: x.quantile(0.25)), g.transform(lambda x: x.quantile(0.75))
    log["n_pos"] = len(pos)
    log["n_iqr"] = int((pos["kwh"] > q3 + 3 * (q3 - q1)).sum())
    log["n_long"] = len(long)
    return raw_df, long, log


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
    out["ts"] = out["ts"].astype("datetime64[ns]")
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
    return pd.DataFrame({"ts": pd.to_datetime(h["time"]).astype("datetime64[ns]"),
                         "temp": h["temperature_2m"],
                         "rad": h["shortwave_radiation"]})


# ───────────────────────── 분석 함수 ─────────────────────────
def aggregate(long: pd.DataFrame, plants: list, start, end):
    """선택 발전소를 합산. 선택한 발전소가 모두 자료를 가진 날짜만 사용."""
    sub = long[long["발전구분"].isin(plants) & long["일자"].between(pd.Timestamp(start), pd.Timestamp(end))]
    cnt = sub.groupby("일자")["발전구분"].nunique()
    full_days = cnt[cnt == len(plants)].index
    sub_full = sub[sub["일자"].isin(full_days)]
    hourly = sub_full.groupby(["일자", "hour"], as_index=False)["kwh"].sum()
    # 'N시' = (N-1)~N시 구간 발전량으로 보고 N:00 시점 기온과 매칭
    hourly["ts"] = (hourly["일자"] + pd.to_timedelta(hourly["hour"], unit="h")).astype("datetime64[ns]")
    info = {"n_sel": len(sub), "n_sel_full": len(sub_full)}
    return hourly, len(full_days), info


def peak_hours(w: pd.DataFrame, col: str) -> pd.Series:
    w = w.dropna(subset=[col])
    return w.loc[w.groupby("일자")[col].idxmax()].set_index("일자")["hour"]


def corr_safe(a: pd.Series, b: pd.Series, method="pearson"):
    return a.corr(b, method=method) if len(a) > 2 and a.std() > 0 and b.std() > 0 else float("nan")


def describe_r(r: float) -> str:
    if np.isnan(r):
        return "상관을 계산할 수 없습니다."
    a = abs(r)
    sign = "양(+)" if r > 0 else "음(−)"
    if a < 0.1:
        return "상관이 거의 없습니다."
    return f"{'약한' if a < 0.3 else '보통 수준의' if a < 0.6 else '강한'} {sign}의 상관이 있습니다."


def fmt(n) -> str:
    return f"{int(n):,}"


def ax(field, typ, title, kind="x", **kw):
    """축 인코딩 (열 이름을 shorthand 문자열로 해석하지 않도록 field/type을 명시)"""
    cls = alt.Y if kind == "y" else alt.X
    return cls(field=field, type=typ, title=title, **kw)


def tip(field, typ, title, fmt_=None):
    kw = {"format": fmt_} if fmt_ else {}
    return alt.Tooltip(field=field, type=typ, title=title, **kw)


def show(chart, title, height=300):
    """그래프 하나가 실패해도 앱 전체가 멈추지 않도록 보호"""
    try:
        st.altair_chart(chart.properties(title=title, height=height, width="container"))
    except Exception as e:  # noqa: BLE001
        st.error(f"'{title}' 그래프를 그리지 못했습니다: {type(e).__name__}: {e}")


def finding(text: str, evidence: str):
    st.info(f"🔎 **찾아낸 것** · {text}\n\n📊 **근거 수치** · {evidence}")


# ───────────────────────── 화면 ─────────────────────────
def main():
    st.set_page_config(page_title="태양광 발전량 vs 기온", page_icon="☀️", layout="wide")
    st.title("☀️ 태양광 발전량은 정말 기온이 가장 높은 시간대에 가장 많을까?")
    st.caption("한국남동발전 시간대별 태양광 발전실적 · 발전량이 가장 많은 시각과 기온이 가장 높은 시각을 비교합니다.")

    # ── 사이드바
    with st.sidebar:
        st.header("1. 발전 데이터")
        origin = st.radio("데이터 출처", ["GitHub (기본)", "CSV 업로드"])
        local = Path(__file__).parent / DEFAULT_CSV
        raw = None
        if origin == "CSV 업로드":
            up = st.file_uploader("발전실적 CSV", type="csv")
            if up is not None:
                raw = up.getvalue()
        else:
            try:
                raw = fetch_url(DATA_URL)
                st.caption("GitHub 원본 CSV를 사용 중입니다.")
            except Exception as e:
                if local.exists():
                    raw = local.read_bytes()
                    st.warning(f"GitHub 로딩 실패({e}). 같은 폴더의 {DEFAULT_CSV}로 대체합니다.")
                else:
                    st.error(f"GitHub 로딩 실패: {e}\n'CSV 업로드'를 선택해 파일을 올려 주세요.")
        if raw is None:
            st.stop()
        try:
            raw_df, long, log = prepare(raw)
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
                           help="낮 시간대만 비교하는 것이 안전합니다. (야간 값은 전처리에서도 0으로 처리됩니다.)")

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

    hourly, n_days, info = aggregate(long, plants, start, end)
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

    merged = hourly.merge(temp_df, on="ts", how="left") if temp_df is not None else hourly.assign(temp=np.nan)
    merged["temp"] = pd.to_numeric(merged["temp"])
    w = merged[merged["hour"].between(h0, h1)].copy()
    has_temp = temp_df is not None and w["temp"].notna().any()
    wt = w.dropna(subset=["temp"]) if has_temp else w

    # 분석용 표: 일자 · 시 · 발전량 · 기온 (선택한 발전소 합산, 1~24시 전체)
    analysis = (merged[["일자", "hour", "kwh", "temp"]]
                .rename(columns={"hour": "시", "kwh": "발전량(kWh)", "temp": "기온(℃)"})
                .sort_values(["일자", "시"]).reset_index(drop=True))

    st.markdown(f"**분석 대상**: {', '.join(plants)} · {n_days}일 · {h0}~{h1}시"
                + (f" · 기온: {temp_note}" if has_temp else ""))

    tab0, tab1, tab2, tab3 = st.tabs(
        ["① 데이터 개요·전처리", "② 시간대별 발전 패턴", "③ 기온 vs 발전량", "④ 분석용 데이터"])

    # ═════════ 탭0: 데이터 개요 + 전처리 ═════════
    with tab0:
        st.subheader("데이터 개요 (원본 CSV)")
        c = st.columns(4)
        c[0].metric("행 개수", fmt(raw_df.shape[0]))
        c[1].metric("열 개수", fmt(raw_df.shape[1]))
        c[2].metric("발전소(발전구분) 수", fmt(raw_df["발전구분"].nunique()))
        c[3].metric("기간", f"{dmin:%m/%d} ~ {dmax:%m/%d}")
        ov = pd.DataFrame({
            "열 이름": raw_df.columns,
            "자료형": raw_df.dtypes.astype(str).values,
            "빈 값 개수": raw_df.isna().sum().values,
            "빈 값 비율(%)": (raw_df.isna().mean() * 100).round(1).values,
        })
        st.dataframe(ov, hide_index=True)
        st.caption(f"빈 값이 있는 열: {int((ov['빈 값 개수'] > 0).sum())}개 / 전체 {len(ov)}개 열. "
                   "행 하나는 '발전기 1대 × 하루'이고, 1시~24시 열이 시간대별 발전량(kWh)입니다.")

        st.subheader("분석용 데이터 개요 (일자 · 시 · 발전량 · 기온)")
        st.caption("사이드바에서 고른 발전소를 합산해 '어느 날의 몇 시'를 한 행으로 만든 표입니다. 이후 모든 그래프의 입력입니다.")
        ca = st.columns(3)
        ca[0].metric("행 개수", fmt(analysis.shape[0]))
        ca[1].metric("열 개수", fmt(analysis.shape[1]))
        ca[2].metric("기온이 있는 행", f"{fmt(analysis['기온(℃)'].notna().sum())} / {fmt(len(analysis))}")
        st.dataframe(pd.DataFrame({
            "열 이름": analysis.columns,
            "자료형": analysis.dtypes.astype(str).values,
            "빈 값 개수": analysis.isna().sum().values,
            "설명": ["날짜", "시각(1~24시, N시 = (N-1)~N시 구간)", "선택 발전소 합산 발전량",
                     "같은 시각의 기온 (기온 데이터를 불러오지 않으면 비어 있음)"],
        }), hide_index=True)

        st.subheader("결측치·이상치 처리 내용")
        mm = log["n_total_mismatch"]
        cell = pd.DataFrame([
            ["일자 오류 행", log["n_bad_date"], "행 삭제", "날짜로 읽을 수 없으면 시간 매칭이 불가능"],
            ["중복 행 (발전구분·호기·일자 동일)", log["n_dup"], "첫 행만 유지", "같은 발전기·날짜가 두 번 집계되는 것을 방지"],
            [f"시간별 값 결측 (전체 {fmt(log['n_cells'])}칸 중)", log["n_missing"], "0으로 대체",
             "발전 기록이 없는 칸은 발전이 없던 것으로 간주(대부분 야간)"],
            ["음수 발전량", log["n_neg"], "0으로 보정", "태양광 발전량은 음수가 될 수 없음"],
            [f"야간({'·'.join(str(h) for h in sorted(NIGHT_HOURS))}시) 양수 값 — 이상치", log["n_night"], "0으로 처리",
             "일몰~일출 사이에 발전이 있을 수 없어 계측/집계 오류로 판단"
             + (f" (발생: {', '.join(f'{k} {v}행' for k, v in log['night_plants'].items())})" if log["night_plants"] else "")],
            [f"낮 시간 통계적 이상치 (발전기별 양수값 Tukey 3×IQR 초과, 양수 {fmt(log['n_pos'])}개 중)",
             log["n_iqr"], "유지(표시만)", "일사량이 큰 날의 정상 변동일 수 있어 삭제하지 않음"],
            ["(검증) 시간별 합계 ≠ 총량 컬럼인 행", mm if mm is not None else "총량 컬럼 없음", "확인만",
             "0이면 시간별 값과 총량이 모두 일치 = 원자료 내부 정합성 양호"],
        ], columns=["항목", "해당 개수", "처리 방법", "이유"])
        cell["해당 개수"] = cell["해당 개수"].map(lambda x: fmt(x) if isinstance(x, (int, np.integer)) else x)
        st.dataframe(cell, hide_index=True)

        st.subheader("처리 전후 행 수")
        steps = [
            ["원본 CSV", "—", "행(발전기×일자)", log["n_raw"], log["n_raw"], True],
            ["일자 오류 행 제거", "일자를 날짜로 변환하지 못한 행 삭제", "행(발전기×일자)", log["n_raw"], log["n_after_date"], True],
            ["중복 행 제거", "발전구분·호기·일자가 같은 행은 첫 행만 유지", "행(발전기×일자)", log["n_after_date"], log["n_wide"], True],
            ["세로 변환", "1시~24시 열을 행으로 풀기 (결측·음수·야간 이상치 처리 포함)", "행 → 시간값", log["n_wide"], log["n_long"], False],
            ["발전소·기간 선택", "사이드바에서 고른 발전소와 기간만", "시간값", log["n_long"], info["n_sel"], True],
            ["공통 일자만 사용", "선택한 발전소가 모두 자료를 가진 날짜만", "시간값", info["n_sel"], info["n_sel_full"], True],
            ["발전소 합산", "같은 날짜·시각의 발전량을 합산", "행(일자×시)", info["n_sel_full"], len(hourly), False],
            [f"분석 시간대({h0}~{h1}시)", "분석 시간대 밖의 시각 제외", "행(일자×시)", len(hourly), len(w), True],
        ]
        if has_temp:
            steps.append(["기온 매칭 결측 제거", "같은 시각의 기온이 없는 행 제외", "행(일자×시)", len(w), len(wt), True])
        fun = pd.DataFrame([{
            "단계": s_[0], "처리 내용": s_[1], "단위": s_[2],
            "처리 전": fmt(s_[3]), "처리 후": fmt(s_[4]),
            "제거된 수": fmt(s_[3] - s_[4]) if s_[5] else "— (변환/합산)",
        } for s_ in steps])
        st.dataframe(fun, hide_index=True)
        st.caption("※ 결측·음수·야간 이상치는 행을 지우지 않고 값만 0으로 바꾸기 때문에 '세로 변환' 단계의 행 수는 유지됩니다.")

    # ═════════ 탭1: 발전 패턴 (기온 없이도 가능) ═════════
    with tab1:
        avg = merged.groupby("hour", as_index=False)["kwh"].mean().rename(columns={"kwh": "avg_kwh"})
        gp = peak_hours(w, "kwh")
        total_all, total_win = merged["kwh"].sum(), w["kwh"].sum()
        c1, c2 = st.columns(2)
        with c1:
            show(alt.Chart(avg).mark_bar().encode(
                x=ax("hour", "ordinal", "시각(시)"),
                y=ax("avg_kwh", "quantitative", "평균 발전량 (kWh)", kind="y"),
                tooltip=[tip("hour", "ordinal", "시각(시)"),
                         tip("avg_kwh", "quantitative", "평균 발전량(kWh)", ",.1f")]),
                "그래프 1. 시간대별 평균 발전량")
            ph = int(avg.loc[avg["avg_kwh"].idxmax(), "hour"])
            pv = float(avg["avg_kwh"].max())
            finding(f"발전량은 {ph}시 무렵에 가장 많고, 선택한 낮 시간대({h0}~{h1}시)에 집중됩니다.",
                    f"{ph}시 평균 {pv:,.0f} kWh (24시간 평균 {avg['avg_kwh'].mean():,.0f} kWh의 "
                    f"{pv / max(avg['avg_kwh'].mean(), 1e-9):.1f}배) · {h0}~{h1}시 구간이 하루 발전량의 "
                    f"{total_win / max(total_all, 1e-9):.1%}")
        with c2:
            dist = gp.value_counts().sort_index().rename_axis("hour").reset_index(name="days")
            show(alt.Chart(dist).mark_bar(color="#f59e0b").encode(
                x=ax("hour", "ordinal", "그날 발전량이 가장 많았던 시각(시)"),
                y=ax("days", "quantitative", "일수 (일)", kind="y"),
                tooltip=[tip("hour", "ordinal", "시각(시)"), tip("days", "quantitative", "일수")]),
                "그래프 2. 일별 발전량 최대 시각의 분포")
            mode_h = int(dist.loc[dist["days"].idxmax(), "hour"])
            mode_n = int(dist["days"].max())
            finding(f"날마다 발전 최대 시각은 {mode_h}시가 가장 흔하지만 날씨에 따라 달라집니다.",
                    f"{len(gp)}일 중 {mode_n}일({mode_n / len(gp):.0%})이 {mode_h}시 · "
                    f"범위 {int(gp.min())}~{int(gp.max())}시 · 중앙값 {gp.median():.0f}시")

    # ═════════ 탭2: 기온 비교 ═════════
    with tab2:
        if not has_temp:
            st.info("사이드바에서 기온 데이터를 불러오면 비교 그래프와 결과가 표시됩니다. "
                    "(발전실적 CSV에는 기온 컬럼이 없습니다.)")
        else:
            gp, tp = peak_hours(wt, "kwh"), peak_hours(wt, "temp")
            pk = pd.concat([gp.rename("gen"), tp.rename("temp")], axis=1).dropna()
            n = len(pk)
            same = int((pk["gen"] == pk["temp"]).sum())
            near = int(((pk["gen"] - pk["temp"]).abs() <= 1).sum())
            lag_med = float((pk["temp"] - pk["gen"]).median())
            # 그래프용 단순 복사본 (영문 열 이름, 날짜는 문자열)
            cd = pd.DataFrame({"date": wt["일자"].dt.strftime("%Y-%m-%d").values,
                               "hour": wt["hour"].astype(int).values,
                               "temp": wt["temp"].astype(float).values,
                               "kwh": wt["kwh"].astype(float).values})

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

            # 그래프 3: 정규화 일주기 패턴
            prof = cd.groupby("hour", as_index=False).agg(kwh=("kwh", "mean"), temp=("temp", "mean"))
            rng = prof["temp"].max() - prof["temp"].min()
            kmax = prof["kwh"].max()
            prof["gen_n"] = prof["kwh"] / kmax if kmax > 0 else 0.0
            prof["temp_n"] = (prof["temp"] - prof["temp"].min()) / rng if rng > 0 else 0.0
            pl = pd.concat([
                pd.DataFrame({"hour": prof["hour"], "series": "발전량 (최대=1)", "value": prof["gen_n"]}),
                pd.DataFrame({"hour": prof["hour"], "series": "기온 (최저=0, 최고=1)", "value": prof["temp_n"]}),
            ], ignore_index=True)
            show(alt.Chart(pl).mark_line(point=True).encode(
                x=ax("hour", "ordinal", "시각(시)"),
                y=ax("value", "quantitative", "정규화 값 (0~1)", kind="y", scale=alt.Scale(domain=[0, 1.05])),
                color=alt.Color(field="series", type="nominal", title="지표"),
                tooltip=[tip("hour", "ordinal", "시각(시)"), tip("series", "nominal", "지표"),
                         tip("value", "quantitative", "값", ".2f")]),
                "그래프 3. 하루 중 평균 발전량과 평균 기온의 모양 비교", height=320)
            g_h = int(prof.loc[prof["kwh"].idxmax(), "hour"])
            t_h = int(prof.loc[prof["temp"].idxmax(), "hour"])
            t_at_g = float(prof.loc[prof["hour"] == g_h, "temp_n"].iloc[0])
            gap = t_h - g_h
            rel = (f"기온이 발전 피크보다 {gap}시간 늦게 최고가 됩니다" if gap > 0 else
                   "기온과 발전량이 같은 시각에 최고가 됩니다" if gap == 0 else
                   f"기온이 발전 피크보다 {-gap}시간 일찍 최고가 됩니다")
            finding(f"평균 하루 패턴에서 {rel}.",
                    f"평균 발전 최대 {g_h}시 · 평균 기온 최고 {t_h}시(평균 {prof['temp'].max():.1f}℃) · "
                    f"발전 최대 시각의 기온은 하루 중 최저=0~최고=1 기준 {t_at_g:.2f}")

            # 그래프 4: 피크 시각 차이
            diff = ((pk["temp"] - pk["gen"]).astype(int).value_counts().sort_index()
                    .rename_axis("gap").reset_index(name="days"))
            show(alt.Chart(diff).mark_bar(color="#ef4444").encode(
                x=ax("gap", "ordinal", "기온 최고 시각 − 발전 최대 시각 (시간)"),
                y=ax("days", "quantitative", "일수 (일)", kind="y"),
                tooltip=[tip("gap", "ordinal", "차이(시간)"), tip("days", "quantitative", "일수")]),
                "그래프 4. 두 피크 시각의 차이 분포 (0이면 같은 시각)")
            top = diff.loc[diff["days"].idxmax()]
            finding(f"가장 흔한 차이는 {int(top['gap']):+d}시간이며, 두 피크가 정확히 같은 날은 {same / n:.0%}입니다.",
                    f"같은 시각 {same}/{n}일 · ±1시간 이내 {near}/{n}일 · 차이 중앙값 {lag_med:+.0f}시간 · "
                    f"최빈 차이 {int(top['gap']):+d}시간({int(top['days'])}일)")

            # 그래프 5: 산점도 + 추세선
            r_p, r_s = corr_safe(cd["kwh"], cd["temp"]), corr_safe(cd["kwh"], cd["temp"], "spearman")
            slope = float(np.polyfit(cd["temp"], cd["kwh"], 1)[0]) if cd["temp"].std() > 0 else float("nan")
            base = alt.Chart(cd).encode(
                x=ax("temp", "quantitative", "기온 (℃)", scale=alt.Scale(zero=False)),
                y=ax("kwh", "quantitative", "발전량 (kWh)", kind="y"))
            pts = base.mark_circle(size=60, opacity=0.6).encode(
                color=alt.Color(field="hour", type="ordinal", title="시각(시)", scale=alt.Scale(scheme="turbo")),
                tooltip=[tip("date", "nominal", "날짜"), tip("hour", "ordinal", "시각(시)"),
                         tip("temp", "quantitative", "기온(℃)", ".1f"), tip("kwh", "quantitative", "발전량(kWh)", ",.0f")])
            trend = base.transform_regression("temp", "kwh").mark_line(color="black")
            show(pts + trend, "그래프 5. 기온과 시간별 발전량의 관계 (점 = 하루 중 한 시각, 검은 선 = 추세선)", height=340)
            finding(f"기온과 발전량(시간 단위)은 {describe_r(r_p)} 다만 기온이 발전량을 만든다는 뜻은 아닙니다.",
                    f"피어슨 r = {r_p:.2f} · 스피어만 ρ = {r_s:.2f} · 추세선 기울기 {slope:+,.0f} kWh/℃ · 표본 {fmt(len(cd))}개 시각")

            cl, cr = st.columns(2)
            with cl:
                try:
                    bins = pd.qcut(cd["temp"], 5, duplicates="drop")
                    b = cd.assign(bin=bins).groupby("bin", observed=True).agg(
                        avg_kwh=("kwh", "mean"), n=("kwh", "size")).reset_index()
                    b["label"] = b["bin"].apply(lambda i: f"{i.left:.1f}~{i.right:.1f}℃")
                    b = b[["label", "avg_kwh", "n"]]
                    show(alt.Chart(b).mark_bar().encode(
                        x=ax("label", "nominal", "기온 구간 (℃, 5분위)", sort=list(b["label"])),
                        y=ax("avg_kwh", "quantitative", "평균 발전량 (kWh)", kind="y"),
                        tooltip=[tip("label", "nominal", "기온 구간"),
                                 tip("avg_kwh", "quantitative", "평균 발전량(kWh)", ",.1f"),
                                 tip("n", "quantitative", "시각 수")]),
                        "그래프 6. 기온 구간별 평균 발전량")
                    v0, v1 = float(b["avg_kwh"].iloc[0]), float(b["avg_kwh"].iloc[-1])
                    mono = bool(np.all(np.diff(b["avg_kwh"].values) > 0))
                    if v1 > v0 * 1.1:
                        txt = "기온 구간이 높을수록 평균 발전량이 많습니다" + (" (구간마다 계속 증가)." if mono else ".")
                    elif v1 < v0 * 0.9:
                        txt = "기온 구간이 높을수록 평균 발전량이 오히려 적습니다."
                    else:
                        txt = "기온 구간에 따른 평균 발전량 차이가 크지 않습니다."
                    finding(txt + " (일사량·시각 효과가 섞인 값)",
                            f"최저 구간 {b['label'].iloc[0]} {v0:,.0f} kWh → 최고 구간 {b['label'].iloc[-1]} {v1:,.0f} kWh"
                            + (f" ({v1 / v0:.1f}배)" if v0 > 0 else ""))
                except ValueError:
                    st.caption("기온 값의 변동이 적어 구간을 나눌 수 없습니다.")
            with cr:
                rows = [{"hour": int(h), "r": corr_safe(g["kwh"], g["temp"])} for h, g in cd.groupby("hour")]
                hr = pd.DataFrame(rows).dropna()
                if hr.empty:
                    st.caption("같은 시각끼리의 상관을 계산할 데이터가 부족합니다.")
                else:
                    show(alt.Chart(hr).mark_bar().encode(
                        x=ax("hour", "ordinal", "시각(시)"),
                        y=ax("r", "quantitative", "날짜별 기온↔발전량 상관계수", kind="y",
                             scale=alt.Scale(domain=[-1, 1])),
                        tooltip=[tip("hour", "ordinal", "시각(시)"), tip("r", "quantitative", "상관계수", ".2f")]),
                        "그래프 7. 같은 시각끼리 비교한 기온↔발전량 상관")
                    mean_r = float(hr["r"].mean())
                    pos_n = int((hr["r"] > 0).sum())
                    finding(f"같은 시각끼리 비교하면 기온과 발전량은 {describe_r(mean_r)} "
                            "(같은 시각 비교에서 나타나는 관계는 대체로 맑은 날씨 효과입니다).",
                            f"시각별 평균 r = {mean_r:.2f} · 양(+)인 시각 {pos_n}/{len(hr)}개 · "
                            f"범위 {hr['r'].min():.2f} ~ {hr['r'].max():.2f}")

            # 그래프 8: 일 단위
            daily = cd.groupby("date", as_index=False).agg(kwh=("kwh", "sum"), temp=("temp", "mean"))
            dbase = alt.Chart(daily).encode(
                x=ax("temp", "quantitative", "분석 시간대 평균기온 (℃)", scale=alt.Scale(zero=False)),
                y=ax("kwh", "quantitative", f"{h0}~{h1}시 발전량 합계 (kWh)", kind="y"))
            dpts = dbase.mark_circle(size=70, opacity=0.7).encode(
                tooltip=[tip("date", "nominal", "날짜"), tip("temp", "quantitative", "평균기온(℃)", ".1f"),
                         tip("kwh", "quantitative", "발전량(kWh)", ",.0f")])
            dtrend = dbase.transform_regression("temp", "kwh").mark_line(color="black")
            show(dpts + dtrend, "그래프 8. 날짜별 평균기온과 하루 발전량 (점 = 하루)", height=320)
            r_d = corr_safe(daily["kwh"], daily["temp"])
            hot, cold = daily.loc[daily["temp"].idxmax()], daily.loc[daily["temp"].idxmin()]
            finding(f"날짜 단위로 보면 평균기온과 하루 발전량은 {describe_r(r_d)}",
                    f"일 단위 r = {r_d:.2f} (n = {len(daily)}일) · 가장 따뜻한 날 {hot['date'][5:].replace('-', '/')}({hot['temp']:.1f}℃) "
                    f"{hot['kwh']:,.0f} kWh · 가장 쌀쌀한 날 {cold['date'][5:].replace('-', '/')}({cold['temp']:.1f}℃) {cold['kwh']:,.0f} kWh")

            with st.expander("해석할 때 주의할 점"):
                st.markdown(
                    "- **상관관계는 인과관계가 아닙니다.** 태양광 발전량은 기본적으로 일사량에 비례하고, "
                    "기온은 일사로 데워진 공기가 만든 결과라 함께 움직일 뿐입니다.\n"
                    "- 일반적으로 하루 중 기온 최고점은 태양이 가장 높은 시각(정오 무렵)보다 늦게 나타나므로 "
                    "두 피크가 어긋나는 것이 자연스럽습니다. 또한 패널 온도가 오르면 변환 효율은 오히려 조금 떨어집니다.\n"
                    "- 발전량은 날씨(구름), 설비 용량, 출력 제한 등의 영향을 받습니다. "
                    "발전소를 합산하면 설비 용량이 큰 곳이 결과를 좌우합니다.\n"
                    "- 조회한 기온은 발전소 위치가 아닌 대표 지점(또는 격자) 값이며, "
                    "'N시' 발전량은 N:00 시점의 기온과 맞춰 비교했습니다."
                )

    # ═════════ 탭3: 분석용 데이터 ═════════
    with tab3:
        st.subheader("분석용 데이터 (일자 · 시 · 발전량 · 기온)")
        table = analysis.assign(일자=analysis["일자"].dt.strftime("%Y-%m-%d"))
        st.dataframe(table, hide_index=True)
        st.download_button("CSV로 내려받기", table.to_csv(index=False).encode("utf-8-sig"),
                           "태양광_일자_시_발전량_기온.csv", "text/csv")
        st.subheader("발전소별 자료 일수")
        st.dataframe(coverage.rename("자료 일수").reset_index(), hide_index=True)


if __name__ == "__main__":
    main()
