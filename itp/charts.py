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


def overview_chart(substation, start, end, params, height=320):
    sid = _substation_id(substation)
    return _lines(_series(sid, pd.Timestamp(start), pd.Timestamp(end) + pd.Timedelta(hours=23), params), height)
