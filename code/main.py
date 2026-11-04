# -*- coding: utf-8 -*-
"""태양광 발전량은 정말 기온이 높은 시간대에 가장 많을까? (Streamlit)"""
import io
import re
import urllib.request

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

DATA_URL = (
    "https://raw.githubusercontent.com/dg2025121-dotcom/solar-energy/refs/heads/main/data/"
    "%ED%95%9C%EA%B5%AD%EB%82%A8%EB%8F%99%EB%B0%9C%EC%A0%84_%EC%8B%9C%EA%B0%84%EB%8C%80%EB%B3%84_"
    "%ED%83%9C%EC%96%91%EA%B4%91_%EB%B0%9C%EC%A0%84%EC%8B%A4%EC%A0%81.csv"
)
NONE = "(없음)"

st.set_page_config(page_title="태양광 발전량 vs 기온", page_icon="☀️", layout="wide")
st.title("☀️ 태양광 발전량은 정말 기온이 높은 시간대에 가장 많을까?")
st.caption("한국남동발전 시간대별 태양광 발전실적 · 분석 속성: 발전량, 시, 기온")


# ───────────────────────── 데이터 읽기 ─────────────────────────
def decode_bytes(raw: bytes) -> pd.DataFrame:
    last = None
    for enc in ("utf-8-sig", "cp949", "euc-kr"):
        try:
            df = pd.read_csv(io.BytesIO(raw), encoding=enc)
            df.columns = [str(c).strip() for c in df.columns]
            return df
        except Exception as e:  # noqa: BLE001
            last = e
    raise last


@st.cache_data(show_spinner="CSV 불러오는 중...")
def load_url(url: str) -> pd.DataFrame:
    with urllib.request.urlopen(url, timeout=30) as r:
        return decode_bytes(r.read())


def guess(cols, keys, default=NONE):
    for k in keys:
        for c in cols:
            if k in str(c).lower():
                return c
    return default


def hour_columns(df):
    """'1시', '01', '13h' 처럼 시각별로 펼쳐진 열(와이드 형식) 찾기"""
    found = []
    for c in df.columns:
        m = re.fullmatch(r"\s*(\d{1,2})\s*(시|h|H|:00)?\s*", str(c))
        if m and 0 <= int(m.group(1)) <= 24:
            found.append((c, int(m.group(1))))
    return found if len(found) >= 12 else []


def split_datetime_or_hour(s: pd.Series):
    """시 열이 숫자면 그대로, '2024-07-01 13:00' 같은 일시면 일자와 시로 분리"""
    num = pd.to_numeric(s, errors="coerce")
    if num.notna().mean() > 0.9:
        return None, num
    dt = pd.to_datetime(s, errors="coerce")
    if dt.notna().mean() > 0.9:
        return dt.dt.normalize(), dt.dt.hour.astype("float")
    ext = s.astype(str).str.extract(r"(\d{1,2})")[0]
    return None, pd.to_numeric(ext, errors="coerce")


# ───────────────────────── 사이드바: 데이터 선택 ─────────────────────────
st.sidebar.header("⚙️ 설정")
up = st.sidebar.file_uploader("다른 CSV로 분석하려면 업로드 (선택)", type="csv")
try:
    raw = decode_bytes(up.getvalue()) if up else load_url(DATA_URL)
except Exception as e:  # noqa: BLE001
    st.error(f"데이터를 불러오지 못했습니다: {e}")
    st.stop()

# ───────────────────────── 1. 데이터 개요 ─────────────────────────
st.header("1. 데이터 개요")
c1, c2 = st.columns(2)
c1.metric("행 개수", f"{raw.shape[0]:,}")
c2.metric("열 개수", f"{raw.shape[1]:,}")
overview = pd.DataFrame(
    {
        "열 이름": raw.columns,
        "자료형": [str(t) for t in raw.dtypes],
        "빈 값 개수": raw.isna().sum().values,
        "빈 값 비율(%)": (raw.isna().mean() * 100).round(2).values,
    }
)
st.dataframe(overview, use_container_width=True, hide_index=True)
with st.expander("원본 데이터 미리보기 (상위 10행)"):
    st.dataframe(raw.head(10), use_container_width=True)

# ───────────────────────── 2. 속성(발전량·시·기온) 만들기 ─────────────────────────
st.header("2. 분석 속성 만들기 (발전량 · 시 · 기온)")
hours = hour_columns(raw)
work = None
date_series = None

if hours:
    st.info(f"시각별로 펼쳐진 형식(열 {len(hours)}개)을 감지해 세로 형태로 변환했습니다.")
    hcols = [c for c, _ in hours]
    idcols = [c for c in raw.columns if c not in hcols]

    with st.sidebar.expander("행 필터 (선택)", expanded=False):
        sel_col = st.selectbox("필터 기준 열", [NONE] + [c for c in idcols if raw[c].nunique() <= 40])
        sub = raw
        if sel_col != NONE:
            vals = sorted(raw[sel_col].dropna().astype(str).unique())
            keep = st.multiselect("남길 값", vals, default=vals)
            sub = raw[raw[sel_col].astype(str).isin(keep)]

    date_col = st.sidebar.selectbox("일자 열", [NONE] + idcols, index=([NONE] + idcols).index(guess(idcols, ["일자", "날짜", "date"])))
    temp_col = st.sidebar.selectbox("기온 열 (원본에 있으면)", [NONE] + idcols, index=([NONE] + idcols).index(guess(idcols, ["기온", "temp"])))

    keep_cols = [c for c in (date_col, temp_col) if c != NONE]
    long = sub.melt(id_vars=keep_cols, value_vars=hcols, var_name="_h", value_name="발전량")
    hmap = {c: h for c, h in hours}
    long["시"] = long["_h"].map(hmap)
    work = long.drop(columns="_h")
    if date_col != NONE:
        date_series = pd.to_datetime(work[date_col], errors="coerce").dt.normalize()
    work["기온"] = pd.to_numeric(work[temp_col], errors="coerce") if temp_col != NONE else np.nan
else:
    cols = list(raw.columns)
    opts = [NONE] + cols
    g_p = st.sidebar.selectbox("발전량 열", opts, index=opts.index(guess(cols, ["발전량", "kwh", "발전"])))
    g_h = st.sidebar.selectbox("시 열", opts, index=opts.index(guess(cols, ["시간", "hour", "시", "일시"])))
    g_t = st.sidebar.selectbox("기온 열", opts, index=opts.index(guess(cols, ["기온", "temp"])))
    if NONE in (g_p, g_h):
        st.warning("사이드바에서 발전량 열과 시 열을 선택해 주세요.")
        st.stop()
    d_from, h_vals = split_datetime_or_hour(raw[g_h])
    work = pd.DataFrame(
        {"발전량": pd.to_numeric(raw[g_p], errors="coerce"), "시": h_vals,
         "기온": pd.to_numeric(raw[g_t], errors="coerce") if g_t != NONE else np.nan}
    )
    date_series = d_from
    if date_series is None:
        dc = guess(cols, ["일자", "날짜", "date"])
        if dc != NONE:
            date_series = pd.to_datetime(raw[dc], errors="coerce").dt.normalize()

# 기온이 원본에 없으면 외부 기온 파일과 합치기
if work["기온"].isna().all():
    st.warning("원본 데이터에서 기온 열을 찾지 못했습니다. 기온 CSV(일자·시·기온)를 올리면 일자+시 기준으로 합쳐서 분석합니다.")
    wf = st.file_uploader("기온 CSV 업로드", type="csv", key="weather")
    if wf is None or date_series is None:
        if date_series is None:
            st.error("기온을 합치려면 원본에 일자 열이 필요합니다. 사이드바에서 일자 열을 지정해 주세요.")
        st.stop()
    w = decode_bytes(wf.getvalue())
    wc = list(w.columns)
    wo = [NONE] + wc
    a, b, c = st.columns(3)
    w_dt = a.selectbox("기온 파일의 일자/일시 열", wc, index=wc.index(guess(wc, ["일시", "일자", "날짜", "date"], wc[0])))
    w_hr = b.selectbox("기온 파일의 시 열 (일시에 시각이 있으면 없음)", wo, index=wo.index(guess(wc, ["시간", "hour"])))
    w_tp = c.selectbox("기온 열", wc, index=wc.index(guess(wc, ["기온", "temp"], wc[-1])))
    shift = st.checkbox("발전 데이터의 시가 1~24 표기라서 -1 하여 0~23과 맞추기", value=bool(work["시"].min() == 1 and work["시"].max() == 24))

    wdt = pd.to_datetime(w[w_dt], errors="coerce")
    wd = pd.DataFrame({"_d": wdt.dt.normalize(), "기온_w": pd.to_numeric(w[w_tp], errors="coerce")})
    wd["_h"] = pd.to_numeric(w[w_hr], errors="coerce") if w_hr != NONE else wdt.dt.hour
    wd = wd.dropna().groupby(["_d", "_h"], as_index=False)["기온_w"].mean()
    key = work[["시"]].copy()
    key["_d"] = date_series.values
    key["_h"] = key["시"] - 1 if shift else key["시"]
    work["기온"] = key.merge(wd, on=["_d", "_h"], how="left")["기온_w"].values

work = work[["발전량", "시", "기온"]].reset_index(drop=True)
st.write("분석에 쓰는 표 (속성: 발전량, 시, 기온)")
st.dataframe(work.head(10), use_container_width=True, hide_index=True)

# ───────────────────────── 3. 결측치·이상치 처리 ─────────────────────────
st.header("3. 결측치와 이상치 처리")
k = st.sidebar.slider("이상치 기준 (IQR 배수)", 1.5, 5.0, 3.0, 0.5,
                      help="값이 클수록 더 극단적인 값만 이상치로 제거합니다.")

d = work.copy()
for col in d.columns:
    d[col] = pd.to_numeric(d[col], errors="coerce")
log = [("처리 전 (표준화된 데이터)", len(d), "-")]


def step(name, mask_keep, note):
    global d
    before = len(d)
    d = d[mask_keep(d)]
    log.append((name, len(d), f"{before - len(d):,}행 제거 · {note}"))


miss = d.isna().sum()
step("① 결측치 제거", lambda x: x.notna().all(axis=1),
     f"발전량 {miss['발전량']:,} / 시 {miss['시']:,} / 기온 {miss['기온']:,}개 비어 있어 해당 행 삭제")
d["시"] = d["시"].astype(int)
step("② 시 범위 확인", lambda x: x["시"].between(0, 24), "0~24 밖의 시 삭제")
step("③ 발전량 음수 제거", lambda x: x["발전량"] >= 0, "발전량은 0 이상이어야 하므로 음수 삭제")
step("④ 기온 물리 범위", lambda x: x["기온"].between(-40, 50), "-40~50℃ 밖의 기온(센서 오류) 삭제")

# 발전량 이상치: 시간대마다 발전 패턴이 달라서 시간대별 IQR로 판단 (밤 0값이 낮 발전량을 이상치로 만들지 않게)
grp = d.groupby("시")["발전량"]
q1, q3 = grp.transform(lambda s: s.quantile(0.25)), grp.transform(lambda s: s.quantile(0.75))
iqr = q3 - q1
step("⑤ 발전량 이상치(시간대별 IQR)", lambda x: ~((iqr.loc[x.index] > 0) & (x["발전량"] > q3.loc[x.index] + k * iqr.loc[x.index])),
     f"시간대별 Q3 + {k}×IQR 초과 삭제 (IQR=0인 시간대는 유지)")
qa, qb = d["기온"].quantile([0.25, 0.75])
step("⑥ 기온 이상치(전체 IQR)", lambda x: x["기온"].between(qa - k * (qb - qa), qb + k * (qb - qa)),
     f"{qa - k * (qb - qa):.1f}~{qb + k * (qb - qa):.1f}℃ 밖 삭제")

log.append(("처리 후 (최종)", len(d), f"총 {log[0][1] - len(d):,}행 제거 ({(log[0][1] - len(d)) / max(log[0][1], 1) * 100:.2f}%)"))
cc1, cc2, cc3 = st.columns(3)
cc1.metric("처리 전 행 수", f"{log[0][1]:,}")
cc2.metric("처리 후 행 수", f"{len(d):,}", delta=f"{len(d) - log[0][1]:,}")
cc3.metric("제거 비율", f"{(log[0][1] - len(d)) / max(log[0][1], 1) * 100:.2f}%")
st.dataframe(pd.DataFrame(log, columns=["단계", "남은 행 수", "처리 내용"]), use_container_width=True, hide_index=True)

if len(d) < 30:
    st.error("처리 후 남은 데이터가 너무 적어 분석할 수 없습니다. 기온 열/합치기 설정을 확인해 주세요.")
    st.stop()

# ───────────────────────── 4. 그래프 ─────────────────────────
st.header("4. 질문 분석: 발전량은 기온이 가장 높은 시간대에 가장 많을까?")
lo, hi = st.sidebar.slider("'낮 시간' 범위 (시)", 0, 24, (6, 18))
day = d[d["시"].between(lo, hi)]

# 그래프 1: 시간대별 평균 발전량 & 평균 기온
hourly = d.groupby("시", as_index=False).agg(발전량=("발전량", "mean"), 기온=("기온", "mean"))
hp = hourly.loc[hourly["발전량"].idxmax()]
ht = hourly.loc[hourly["기온"].idxmax()]
fig1 = make_subplots(specs=[[{"secondary_y": True}]])
fig1.add_trace(go.Bar(x=hourly["시"], y=hourly["발전량"], name="평균 발전량", opacity=0.75), secondary_y=False)
fig1.add_trace(go.Scatter(x=hourly["시"], y=hourly["기온"], name="평균 기온", mode="lines+markers"), secondary_y=True)
fig1.update_layout(title="시간대별 평균 발전량과 평균 기온", xaxis_title="시", legend=dict(orientation="h", y=-0.2))
fig1.update_yaxes(title_text="평균 발전량", secondary_y=False)
fig1.update_yaxes(title_text="평균 기온 (℃)", secondary_y=True)
st.plotly_chart(fig1, use_container_width=True)
gap = int(ht["시"] - hp["시"])
same = "같은 시간대" if gap == 0 else (f"기온 최고 시각이 {abs(gap)}시간 {'늦음' if gap > 0 else '빠름'}")
st.markdown(
    f"**찾아낸 것:** 발전량이 가장 많은 시각과 기온이 가장 높은 시각은 {same}입니다.  \n"
    f"**근거:** 평균 발전량 최대 {int(hp['시'])}시({hp['발전량']:,.1f}), 평균 기온 최고 {int(ht['시'])}시({ht['기온']:.1f}℃)."
)

# 그래프 2: 기온 vs 발전량 산점도 (낮 시간)
r_all = d["발전량"].corr(d["기온"])
r_day = day["발전량"].corr(day["기온"]) if len(day) > 2 else np.nan
per_h = day.groupby("시").apply(lambda g: g["발전량"].corr(g["기온"]) if g["기온"].nunique() > 1 and g["발전량"].nunique() > 1 else np.nan)
r_h = per_h.mean()
samp = day.sample(min(len(day), 15000), random_state=0)
fig2 = go.Figure(go.Scatter(x=samp["기온"], y=samp["발전량"], mode="markers", marker=dict(size=4, opacity=0.35), name="관측값"))
if len(samp) > 2:
    m, b0 = np.polyfit(samp["기온"], samp["발전량"], 1)
    xs = np.array([samp["기온"].min(), samp["기온"].max()])
    fig2.add_trace(go.Scatter(x=xs, y=m * xs + b0, mode="lines", name="추세선", line=dict(width=3)))
fig2.update_layout(title=f"기온과 발전량의 관계 ({lo}~{hi}시)", xaxis_title="기온 (℃)", yaxis_title="발전량")
st.plotly_chart(fig2, use_container_width=True)
strength = lambda r: "거의 없음" if abs(r) < 0.2 else ("약함" if abs(r) < 0.4 else ("보통" if abs(r) < 0.6 else "강함"))
st.markdown(
    f"**찾아낸 것:** 낮 시간 기준 기온과 발전량의 상관은 {strength(r_day)}"
    f"({'양' if r_day > 0 else '음'}의 관계)입니다.  \n"
    f"**근거:** 전체 시간 r={r_all:.2f}, 낮({lo}~{hi}시) r={r_day:.2f}, 같은 시각끼리 비교한 평균 r={r_h:.2f}."
)

# 그래프 3: 기온 구간별 평균 발전량 (낮 시간)
bins = np.arange(np.floor(day["기온"].min() / 5) * 5, np.ceil(day["기온"].max() / 5) * 5 + 5, 5)
cut = pd.cut(day["기온"], bins=bins, right=False)
band = day.groupby(cut, observed=True)["발전량"].agg(["mean", "count"]).reset_index()
band["구간"] = band["기온"].astype(str).str.replace("[", "", regex=False).str.replace(")", "", regex=False).str.replace(", ", "~")
band = band[band["count"] >= 5]
fig3 = go.Figure(go.Bar(x=band["구간"], y=band["mean"], text=band["count"].map(lambda v: f"n={v:,}"), textposition="outside"))
fig3.update_layout(title=f"기온 구간별 평균 발전량 ({lo}~{hi}시)", xaxis_title="기온 구간 (℃)", yaxis_title="평균 발전량")
st.plotly_chart(fig3, use_container_width=True)
best = band.loc[band["mean"].idxmax()]
top = band.iloc[-1]
st.markdown(
    f"**찾아낸 것:** 평균 발전량이 가장 큰 기온 구간은 {best['구간']}℃이며, "
    f"{'가장 따뜻한 구간과 같습니다' if best['구간'] == top['구간'] else '가장 따뜻한 구간이 아닙니다'}.  \n"
    f"**근거:** {best['구간']}℃ 평균 {best['mean']:,.1f}(n={int(best['count']):,}) vs "
    f"가장 따뜻한 {top['구간']}℃ 평균 {top['mean']:,.1f}(n={int(top['count']):,})."
)

# ───────────────────────── 5. 결론 ─────────────────────────
st.header("5. 결론")
peak_match = abs(gap) <= 1
if peak_match and r_h > 0.2:
    verdict = "대체로 그렇다고 볼 수 있습니다."
elif peak_match:
    verdict = "시각은 비슷하게 겹치지만, 같은 시각끼리 비교하면 기온 자체의 영향은 크지 않습니다."
else:
    verdict = "그렇다고 보기 어렵습니다."
st.success(
    f"**{verdict}** 발전량 최대 {int(hp['시'])}시 vs 기온 최고 {int(ht['시'])}시, "
    f"낮 시간 상관 r={r_day:.2f}, 같은 시각 내 평균 r={r_h:.2f}."
)
st.caption("※ 낮에는 일사량이 기온과 발전량을 동시에 올리므로, 상관이 있어도 기온이 원인이라고 단정할 수는 없습니다. "
           "(일사량이 같을 때는 오히려 고온에서 패널 효율이 떨어질 수 있습니다.)")
