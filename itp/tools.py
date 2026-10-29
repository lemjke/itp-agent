"""Инструменты агента. Все вычисления — здесь, LLM числа не считает.

Каждая функция возвращает компактный JSON-совместимый словарь: агенту передаются
агрегаты и найденные события, а не сырые часовые ряды.
"""
import re
import warnings

import numpy as np
import pandas as pd
import psycopg

from .config import DATA_END, DATA_START, DB_DSN
from .norms import BY_CODE
from .schedule import graph

warnings.filterwarnings("ignore", message="pandas only supports SQLAlchemy")

PARAMS = ["t_outdoor", "t1", "t2", "t_dhw", "p1", "p2", "flow"]
PARAM_NAMES = {
    "t_outdoor": "наружная температура, °C",
    "t1": "температура подачи, °C",
    "t2": "температура обратки, °C",
    "t_dhw": "температура ГВС, °C",
    "p1": "давление в подающем, бар",
    "p2": "давление в обратном, бар",
    "flow": "расход теплоносителя, т/ч",
}
MAX_DAYS = 62
CONTEXT_DAYS = 3          # история до периода — для базовой линии и сглаживания


class ToolError(ValueError):
    """Ошибка входных данных: текст уходит агенту, чтобы он исправил вызов."""


# ---------- вспомогательное ----------

def _substation_id(substation):
    m = re.search(r"\d+", str(substation))
    if not m:
        raise ToolError(f"Не понял номер ИТП: {substation!r}. Пример: 'ИТП-3' или 3.")
    sid = int(m.group())
    with psycopg.connect(DB_DSN) as con:
        if not con.execute("SELECT 1 FROM substations WHERE id = %s", (sid,)).fetchone():
            raise ToolError(f"ИТП с номером {sid} нет. Вызовите list_substations.")
    return sid


def _period(start, end):
    try:
        s = pd.Timestamp(start)
        e = pd.Timestamp(end)
    except (ValueError, TypeError):
        raise ToolError("Даты в формате 'YYYY-MM-DD' или 'YYYY-MM-DD HH:MM'.")
    if len(str(end).strip()) <= 10:          # дата без времени — до конца суток
        e = e + pd.Timedelta(hours=23)
    if e < s:
        raise ToolError("Конец периода раньше начала.")
    if (e - s).days > MAX_DAYS:
        raise ToolError(f"Период не длиннее {MAX_DAYS} дней.")
    lo, hi = pd.Timestamp(DATA_START), pd.Timestamp(DATA_END)
    if e < lo or s > hi:
        hint = ""
        try:
            s2, e2 = s.replace(year=hi.year), e.replace(year=hi.year)
            if e2 >= lo and s2 <= hi:
                hint = f" Вероятно, ошибка в годе — повторите вызов с start='{s2:%Y-%m-%d}', end='{e2:%Y-%m-%d}'."
        except ValueError:
            pass
        raise ToolError(f"За период {s:%Y-%m-%d} — {e:%Y-%m-%d} данных нет (данные есть с {lo:%Y-%m-%d} "
                        f"по {hi:%Y-%m-%d}).{hint or ' Проверьте даты.'}")
    return s, e


ALL_WORDS = {"все", "всех", "all", "*", "любой"}


def _is_all(substation):
    return str(substation).strip().lower() in ALL_WORDS


def _load(sid, start, end):
    """Часовые данные с запасом истории; плановые значения по графику."""
    with psycopg.connect(DB_DSN) as con:
        df = pd.read_sql(
            "SELECT * FROM measurements WHERE substation_id = %s AND ts BETWEEN %s AND %s ORDER BY ts",
            con, params=(sid, start - pd.Timedelta(days=CONTEXT_DAYS), end),
        )
    df = df.set_index("ts")
    t_smooth = df["t_outdoor"].rolling(6, min_periods=1).mean()
    df["t1_plan"], df["t2_plan"] = graph(t_smooth.values)
    return df


def _fmt(ts):
    return pd.Timestamp(ts).strftime("%Y-%m-%d %H:%M")


def _runs(mask, min_len=1, max_gap=0):
    """Непрерывные участки True (с допуском разрывов max_gap часов) → [(start, end)]."""
    idx = mask.index[mask.values]
    if len(idx) == 0:
        return []
    runs, s, prev = [], idx[0], idx[0]
    for t in idx[1:]:
        if (t - prev) > pd.Timedelta(hours=1 + max_gap):
            runs.append((s, prev))
            s = t
        prev = t
    runs.append((s, prev))
    return [(a, b) for a, b in runs if (b - a) / pd.Timedelta(hours=1) + 1 >= min_len]


def _clip(runs, start, end):
    return [(max(a, start), min(b, end)) for a, b in runs if b >= start and a <= end]


def _signed(v):
    return f"{v:+.1f}".replace(".", ",")


def _num(v):
    return f"{v:.1f}".replace(".", ",")


def _describe(kind, param, ev):
    """Краткое текстовое описание события и поведения связанных параметров (для LLM)."""
    name = PARAM_NAMES.get(param, param)
    if kind == "data_gap":
        return "Нет данных по всем параметрам."
    if kind == "frozen_value":
        return f"Показание «{name}» не менялось, держится на {ev['value']}."
    if kind == "spike":
        return f"Одиночный выброс «{name}»: {ev['value']} при обычном уровне около {ev['neighbours_median']}."
    if kind in ("t1_off_schedule", "t2_above_schedule"):
        main = (f"Подача {ev['direction']}: в среднем {_signed(ev['t1_mean_deviation_pct'])} % от графика "
                f"(макс. {_num(ev['max_deviation_pct'])} %)." if kind == "t1_off_schedule" else
                f"Обратка выше графика: в среднем {_signed(ev['t2_mean_deviation_pct'])} % "
                f"(макс. {_num(ev['max_deviation_pct'])} %, норма — не более +5 %).")
        if kind == "t1_off_schedule":
            t1d, t2d = ev["t1_mean_deviation_pct"], ev["t2_mean_deviation_pct"]
            if abs(t2d) < 3:
                state = "почти в норме"
            elif np.sign(t2d) == np.sign(t1d):
                state = "тоже отклонилась в ту же сторону, что и подача"
            else:
                state = "отклонилась в противоположную сторону"
            related = f" Обратка в это время {_signed(t2d)} % от графика — {state}."
        else:
            related = f" Подача в это время {_signed(ev['t1_mean_deviation_pct'])} % от графика."
        flow = f" Расход {_signed(ev['flow_change_vs_before_pct'])} % к предыдущим дням."
        trend = ""
        if kind == "t1_off_schedule":
            tr = ev["deviation_trend_pct_per_day"]
            trend = (f" Отклонение растёт на {_num(abs(tr))} % в сутки (похоже на постепенный дрейф)."
                     if abs(tr) >= 0.8 else " Отклонение стабильное по величине (не растёт).")
        return main + related + flow + trend
    if kind == "dhw_out_of_range":
        return f"Температура ГВС от {ev['min_value']} до {ev['max_value']} °C при норме 60–75 °C."
    if kind == "low_pressure_drop":
        return f"Перепад давлений снижался до {ev['min_dp']} бар."
    return ""


# ---------- инструменты ----------

def list_substations(**_ignored):
    """Список ИТП и период, за который есть данные."""
    with psycopg.connect(DB_DSN) as con:
        rows = con.execute("SELECT id, name, address, building_type FROM substations ORDER BY id").fetchall()
        lo, hi = con.execute("SELECT min(ts), max(ts) FROM measurements").fetchone()
    return {
        "substations": [{"id": r[0], "name": r[1], "address": r[2], "type": r[3]} for r in rows],
        "data_period": {"start": _fmt(lo), "end": _fmt(hi)},
    }


def query_sensors(substation, start, end, parameters=None):
    """Посуточная сводка параметров ИТП за период + плановые значения по графику."""
    sid = _substation_id(substation)
    s, e = _period(start, end)
    df = _load(sid, s, e).loc[s:e]
    params = [p for p in (parameters or PARAMS) if p in PARAMS] or PARAMS
    daily = df.resample("D")
    out = []
    for day, g in daily:
        row = {"date": day.strftime("%Y-%m-%d"), "hours_with_data": int(g["t1"].notna().sum())}
        for p in params:
            row[p] = round(float(g[p].mean()), 1) if g[p].notna().any() else None
        if "t1" in params:
            row["t1_plan"] = round(float(g["t1_plan"].mean()), 1)
        if "t2" in params:
            row["t2_plan"] = round(float(g["t2_plan"].mean()), 1)
        out.append(row)
    return {"substation": f"ИТП-{sid}", "units": {p: PARAM_NAMES[p] for p in params}, "daily": out}


def detect_anomalies(substation, start, end):
    """Детекторы: пропуски, «замёрзшие» значения, выбросы, отклонения от графика и норм.

    Для каждого события возвращается контекст — как в это время вели себя связанные
    параметры. По нему агент отличает неисправность датчика от нарушения режима.
    substation='все' — проверить все ИТП одним вызовом.
    """
    if _is_all(substation):
        with psycopg.connect(DB_DSN) as con:
            ids = [r[0] for r in con.execute("SELECT id FROM substations ORDER BY id")]
        parts = [detect_anomalies(i, start, end) for i in ids]
        events = sorted((ev for p in parts for ev in p["events"]), key=lambda ev: (ev["start"], ev["substation"]))
        return {"substation": "все", "substations_checked": [p["substation"] for p in parts],
                "period": parts[0]["period"], "events_found": len(events), "events": events}

    sid = _substation_id(substation)
    s, e = _period(start, end)
    df = _load(sid, s, e)
    events = []

    def add(kind, param, a, b, requirement, **evidence):
        req = BY_CODE[requirement]
        events.append({
            "substation": f"ИТП-{sid}", "type": kind, "parameter": param, "start": _fmt(a), "end": _fmt(b),
            "duration_h": int((b - a) / pd.Timedelta(hours=1)) + 1,
            "summary": _describe(kind, param, evidence),
            "requirement": {"code": requirement, "rule": req["rule"], "source": req["source"]},
            "evidence": evidence,
        })

    # 1. Пропуски данных
    gap = df[["t1", "t2", "t_dhw", "p1", "p2", "flow"]].isna().any(axis=1)
    for a, b in _clip(_runs(gap), s, e):
        add("data_gap", "*", a, b, "DATA_CONTINUITY")

    # 2. «Замёрзшие» значения: одно и то же показание 6 часов подряд и дольше
    for p in ["t1", "t2", "t_dhw", "p1", "p2", "flow"]:
        x = df[p]
        group = (x != x.shift()).cumsum()
        size = x.groupby(group).transform("size")
        frozen = (size >= 6) & x.notna()
        for a, b in _clip(_runs(frozen, min_len=6), s, e):
            add("frozen_value", p, a, b, "DATA_VALIDITY", value=float(df.at[a, p]))

    # 3. Одиночные выбросы: точка резко отличается и от предыдущей, и от следующей
    #    в одну сторону (ступенька режима выбросом не считается)
    spikes = pd.Series(False, index=df.index)
    for p in ["t1", "t2", "t_dhw", "p1", "p2", "flow"]:
        x = df[p]
        d_prev, d_next = x - x.shift(1), x - x.shift(-1)
        mad = (d_prev - d_prev.median()).abs().median() * 1.4826
        thr = max(8 * mad, 1.0 if p in ("p1", "p2") else 4.0)
        flag = (d_prev.abs() > thr) & (d_next.abs() > thr) & (np.sign(d_prev) == np.sign(d_next))
        spikes |= flag
        for a, b in _clip(_runs(flag), s, e):
            add("spike", p, a, b, "DATA_VALIDITY",
                value=round(float(x.at[a]), 1),
                neighbours_median=round(float(x.loc[a - pd.Timedelta(hours=3):a - pd.Timedelta(hours=1)].median()), 1))

    # Сглаженные ряды без выбросов — для режимных проверок
    clean = df.copy()
    clean.loc[spikes, ["t1", "t2", "t_dhw", "p1", "p2", "flow"]] = np.nan
    sm = clean.rolling(6, center=True, min_periods=3).mean()
    baseline = clean.loc[: s - pd.Timedelta(hours=1)]
    base_flow = baseline["flow"].mean() if len(baseline) else clean["flow"].mean()

    def context(a, b):
        w = clean.loc[a:b]
        t2_dev = (w["t2"] - w["t2_plan"]).mean() / w["t2_plan"].mean() * 100
        t1_dev = (w["t1"] - w["t1_plan"]).mean() / w["t1_plan"].mean() * 100
        days = max((b - a) / pd.Timedelta(days=1), 1 / 24)
        return dict(
            t1_mean_deviation_pct=round(float(t1_dev), 1),
            t2_mean_deviation_pct=round(float(t2_dev), 1),
            flow_change_vs_before_pct=round(float((w["flow"].mean() / base_flow - 1) * 100), 1),
            outdoor_mean=round(float(w["t_outdoor"].mean()), 1),
            days=round(days, 1),
        )

    def trend_per_day(series, a, b):
        y = series.loc[a:b].dropna()
        if len(y) < 6:
            return 0.0
        x = (y.index - y.index[0]) / pd.Timedelta(days=1)
        return round(float(np.polyfit(x, y.values, 1)[0]), 2)

    # 4. Подача вне графика ±3 % (держится ≥ 6 ч)
    t1_dev = (sm["t1"] - sm["t1_plan"]) / sm["t1_plan"]
    for a, b in _clip(_runs(t1_dev.abs() > 0.03, min_len=6, max_gap=2), s, e):
        dev = t1_dev.loc[a:b]
        add("t1_off_schedule", "t1", a, b, "T1_SCHEDULE",
            direction="выше графика" if dev.mean() > 0 else "ниже графика",
            max_deviation_pct=round(float(dev.abs().max() * 100), 1),
            deviation_trend_pct_per_day=trend_per_day(t1_dev * 100, a, b),
            **context(a, b))

    # 5. Обратка выше графика более чем на 5 % (норма — по среднесуточной, поэтому окно 24 ч)
    sm24 = clean[["t2", "t2_plan"]].rolling(24, center=True, min_periods=12).mean()
    t2_dev = (sm24["t2"] - sm24["t2_plan"]) / sm24["t2_plan"]
    for a, b in _clip(_runs(t2_dev > 0.05, min_len=12, max_gap=2), s, e):
        add("t2_above_schedule", "t2", a, b, "T2_SCHEDULE",
            max_deviation_pct=round(float(t2_dev.loc[a:b].max() * 100), 1),
            **context(a, b))

    # 6. ГВС вне 60–75 °C
    dhw = clean["t_dhw"].rolling(3, center=True, min_periods=2).median()
    for a, b in _clip(_runs((dhw < 60) | (dhw > 75), min_len=2, max_gap=1), s, e):
        add("dhw_out_of_range", "t_dhw", a, b, "DHW_RANGE",
            min_value=round(float(clean["t_dhw"].loc[a:b].min()), 1),
            max_value=round(float(clean["t_dhw"].loc[a:b].max()), 1))

    # 7. Перепад давлений
    dp = sm["p1"] - sm["p2"]
    for a, b in _clip(_runs(dp < 1.0, min_len=3), s, e):
        add("low_pressure_drop", "p1,p2", a, b, "DP_MIN", min_dp=round(float(dp.loc[a:b].min()), 2))

    events.sort(key=lambda ev: ev["start"])
    return {"substation": f"ИТП-{sid}", "period": {"start": _fmt(s), "end": _fmt(e)},
            "events_found": len(events), "events": events}


def get_requirement(code):
    """Текст нормативного требования по коду (коды приходят из detect_anomalies)."""
    req = BY_CODE.get(str(code).strip().upper())
    if not req:
        raise ToolError(f"Нет требования {code!r}. Доступны: {', '.join(BY_CODE)}.")
    return req


TOOLS = {
    "list_substations": list_substations,
    "query_sensors": query_sensors,
    "detect_anomalies": detect_anomalies,
    "get_requirement": get_requirement,
}
