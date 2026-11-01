"""Графики для веб-интерфейса (Altair)."""
import altair as alt
import pandas as pd

from .tools import PARAM_NAMES, _load, _substation_id

ACTUAL = "#1f6f8b"
PLAN = "#9aa5ad"
EVENT = "#e8673a"


def _series(sid, start, end, params):
    df = _load(sid, start, end).loc[start:end].reset_index()
    rows = []
    for p in params:
        rows.append(pd.DataFrame({"ts": df["ts"], "value": df[p], "series": PARAM_NAMES[p], "kind": "факт"}))
        if p in ("t1", "t2"):
            rows.append(pd.DataFrame({"ts": df["ts"], "value": df[f"{p}_plan"],
                                      "series": f"{PARAM_NAMES[p]} — график", "kind": "план"}))
    return pd.concat(rows, ignore_index=True)


PALETTE = {"t_outdoor": "#7f8c8d", "t1": "#e4572e", "t2": "#2e86de", "t_dhw": "#9b59b6",
           "p1": "#16a085", "p2": "#f39c12", "flow": "#27ae60"}


def _lines(data, height):
    domain = list(dict.fromkeys(data["series"]))
    by_name = {v: k for k, v in PARAM_NAMES.items()}
    rng = [PALETTE[by_name[s.replace(" — график", "")]] for s in domain]
    return alt.Chart(data).mark_line(strokeWidth=1.6).encode(
        x=alt.X("ts:T", title=None, axis=alt.Axis(format="%d.%m", tickCount="day", labelOverlap=True)),
        y=alt.Y("value:Q", title=None, scale=alt.Scale(zero=False)),
        color=alt.Color("series:N", title=None, scale=alt.Scale(domain=domain, range=rng),
                        legend=alt.Legend(orient="bottom", columns=2, labelLimit=320)),
        strokeDash=alt.StrokeDash("kind:N", legend=None,
                                  scale=alt.Scale(domain=["факт", "план"], range=[[1, 0], [5, 4]])),
        tooltip=[alt.Tooltip("ts:T", format="%d.%m %H:%M", title="время"),
                 alt.Tooltip("series:N", title="ряд"), alt.Tooltip("value:Q", format=".1f", title="значение")],
    ).properties(height=height)


def event_chart(substation, parameter, start, end, pad_days=2, height=230):
    """Параметр события с запасом по времени, плановой линией и подсветкой интервала."""
    sid = _substation_id(substation)
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    params = {"*": ["t1", "t2"], "p1,p2": ["p1", "p2"]}.get(parameter, [parameter])
    params = [p for p in params if p in PARAM_NAMES] or ["t1"]
    if params == ["t1"] or params == ["t2"]:
        params = ["t1", "t2"]          # подача и обратка рядом — видно, датчик это или режим
    data = _series(sid, s - pd.Timedelta(days=pad_days), e + pd.Timedelta(days=pad_days), params)
    band = alt.Chart(pd.DataFrame({"a": [s], "b": [e + pd.Timedelta(minutes=59)]})).mark_rect(
        color=EVENT, opacity=0.15).encode(x=alt.X("a:T", axis=None), x2="b:T")
    return band + _lines(data, height)


PERIOD_COLORS = ["#2a78d6", "#eb6834"]           # первый и второй период
CORRIDOR = {"t1": "±3 % от графика", "t2": "до +5 % к графику", "t_dhw": "60–75 °C"}


def period_frame(substation, start, end, param):
    """Ряд параметра за период с коридором нормы (lo, hi) — для сравнения периодов."""
    sid = _substation_id(substation)
    s, e = pd.Timestamp(start), pd.Timestamp(end) + pd.Timedelta(hours=23)
    df = _load(sid, s, e).loc[s:e].reset_index()
    out = pd.DataFrame({"ts": df["ts"], "value": df[param]})
    if param == "t1":
        out["lo"], out["hi"], out["plan"] = df["t1_plan"] * 0.97, df["t1_plan"] * 1.03, df["t1_plan"]
    elif param == "t2":
        out["lo"], out["hi"], out["plan"] = df["t2_plan"], df["t2_plan"] * 1.05, df["t2_plan"]
    elif param == "t_dhw":
        out["lo"], out["hi"] = 60.0, 75.0
    return out


def period_chart(frame, title, color, y_domain, height=260):
    base = alt.Chart(frame).encode(x=alt.X("ts:T", title=None, axis=alt.Axis(format="%d.%m", labelOverlap=True)))
    y = alt.Y("value:Q", title=None, scale=alt.Scale(domain=y_domain))
    layers = []
    if "lo" in frame:
        layers.append(base.mark_area(color="#1baf7a", opacity=0.22).encode(y=alt.Y("lo:Q", scale=alt.Scale(
            domain=y_domain), title=None), y2="hi:Q"))
    if "plan" in frame:
        layers.append(base.mark_line(color="#898781", strokeDash=[5, 4], strokeWidth=1).encode(
            y=alt.Y("plan:Q", scale=alt.Scale(domain=y_domain), title=None)))
    layers.append(base.mark_line(color=color, strokeWidth=1.5).encode(
        y=y, tooltip=[alt.Tooltip("ts:T", format="%d.%m %H:%M", title="время"),
                      alt.Tooltip("value:Q", format=".1f", title="значение")]))
    return alt.layer(*layers).properties(title=title, height=height)


def compare_bars(rows, height=240):
    """Парные столбцы «первый период / второй период» по показателям. rows: [{metric, period, value}]."""
    data = pd.DataFrame(rows)
    periods = list(dict.fromkeys(data["period"]))
    top = max(int(data["value"].max()), 1)
    step = max(1, round(top / 5))
    ticks = list(range(0, top + step, step))              # часы и события — целые деления
    # Нулевое значение — короткий штрих цвета периода: иначе «в январе 0» неотличимо от «январь не посчитан»
    data["bar"] = data["value"].where(data["value"] > 0, ticks[-1] * 0.008)
    data["label"] = data["value"].map(lambda v: f"{v:g}")
    scale_x = alt.Scale(domain=[0, ticks[-1] * 1.08])      # место для подписи у самой длинной полосы
    base = alt.Chart(data).encode(
        y=alt.Y("metric:N", title=None, sort=None, axis=alt.Axis(labelLimit=320)),
        yOffset=alt.YOffset("period:N", sort=periods),
        color=alt.Color("period:N", title=None, sort=periods,
                        scale=alt.Scale(domain=periods, range=PERIOD_COLORS), legend=alt.Legend(orient="top")),
        tooltip=[alt.Tooltip("metric:N", title="показатель"), alt.Tooltip("period:N", title="период"),
                 alt.Tooltip("value:Q", title="значение")],
    )
    bars = base.mark_bar().encode(x=alt.X("bar:Q", title=None, scale=scale_x,
                                          axis=alt.Axis(values=ticks, format="d")))
    labels = base.mark_text(align="left", dx=4, fontSize=11, fontWeight="bold").encode(
        x=alt.X("bar:Q", scale=scale_x), text="label:N")
    return (bars + labels).properties(height=height)


def overview_chart(substation, start, end, params, height=320):
    sid = _substation_id(substation)
    return _lines(_series(sid, pd.Timestamp(start), pd.Timestamp(end) + pd.Timedelta(hours=23), params), height)
