"""Сравнение двух периодов одного ИТП: «как этот месяц выглядит на фоне прошлого».

Часы вне нормы считаются тем же сглаживанием и по тем же порогам, что у детекторов (tools.detect_events):
подача — 6 ч, ±3 % от графика; обратка — 24 ч, выше графика более чем на 5 %; ГВС — вне 60–75 °C.
Тепло сравнивается на градусо-сутки (по 20 °C внутри): иначе тёплый месяц всегда «экономнее» холодного.
"""
import numpy as np
import pandas as pd

from . import ml_detect
from .tools import _fmt, _load, _period, _substation_id, detect_events

T_INSIDE = 20.0
EVENT_RU = {"data_gap": "пропуски данных", "frozen_value": "«замёрзшие» показания", "spike": "выбросы",
            "t1_off_schedule": "подача вне графика", "t2_above_schedule": "обратка выше графика",
            "dhw_out_of_range": "ГВС вне 60–75 °C", "low_pressure_drop": "низкий перепад давлений"}
METRICS = {   # ключ: (подпись, единица, сколько знаков)
    "t_outdoor_mean": ("Средняя наружная температура", "°C", 1),
    "t1_dev_pct": ("Подача: среднее отклонение от графика", "%", 1),
    "t2_dev_pct": ("Обратка: среднее отклонение от графика", "%", 1),
    "t_dhw_mean": ("Средняя температура ГВС", "°C", 1),
    "hours_t1_out": ("Часов: подача вне ±3 % от графика", "ч", 0),
    "hours_t2_out": ("Часов: обратка выше графика более 5 %", "ч", 0),
    "hours_dhw_out": ("Часов: ГВС вне 60–75 °C", "ч", 0),
    "hours_missing": ("Часов без данных", "ч", 0),
    "heat_gcal": ("Тепло за период", "Гкал", 1),
    "heat_per_degree_day": ("Тепло на градусо-сутки", "Гкал/(°C·сут)", 3),
    "events": ("Событий детекторов", "", 0),
    "suspicions": ("Статистических подозрений", "", 0),
}


def period_stats(sid, s, e):
    full = _load(sid, s, e)
    df = full.loc[s:e]
    t1_sm = df[["t1", "t1_plan"]].rolling(6, center=True, min_periods=3).mean()
    t2_sm = df[["t2", "t2_plan"]].rolling(24, center=True, min_periods=12).mean()
    dhw_sm = df["t_dhw"].rolling(3, center=True, min_periods=2).median()
    events = detect_events(full, sid, s, e)
    degree_days = float(np.maximum(T_INSIDE - df["t_outdoor"], 0).sum() / 24)
    heat = float((df["flow"] * (df["t1"] - df["t2"])).sum() / 1000)         # т/ч · °C = Мкал/ч → Гкал
    by_type = {}
    for ev in events:
        by_type[ev["type"]] = by_type.get(ev["type"], 0) + 1
    stats = {
        "start": _fmt(s), "end": _fmt(e), "hours": len(df),
        "t_outdoor_mean": float(df["t_outdoor"].mean()),
        "t1_dev_pct": float(((df["t1"] - df["t1_plan"]) / df["t1_plan"]).mean() * 100),
        "t2_dev_pct": float(((df["t2"] - df["t2_plan"]) / df["t2_plan"]).mean() * 100),
        "t_dhw_mean": float(df["t_dhw"].mean()),
        "hours_t1_out": int(((t1_sm["t1"] - t1_sm["t1_plan"]).abs() / t1_sm["t1_plan"] > 0.03).sum()),
        "hours_t2_out": int(((t2_sm["t2"] - t2_sm["t2_plan"]) / t2_sm["t2_plan"] > 0.05).sum()),
        "hours_dhw_out": int(((dhw_sm < 60) | (dhw_sm > 75)).sum()),
        "hours_missing": int(df["t1"].isna().sum()),
        "heat_gcal": heat, "degree_days": degree_days,
        "heat_per_degree_day": heat / degree_days if degree_days else float("nan"),
        "events": len(events), "events_by_type": {EVENT_RU.get(k, k): v for k, v in by_type.items()},
        "suspicions": len(ml_detect.suspicions(full, events, s, e)),
    }
    return {k: (round(v, METRICS[k][2]) if k in METRICS and isinstance(v, float) else v) for k, v in stats.items()}


def compare_periods(substation, start_1, end_1, start_2, end_2):
    """Сравнить два периода одного ИТП: отклонения от графика, часы вне нормы, тепло, события."""
    sid = _substation_id(substation)
    s1, e1, note1 = _period(start_1, end_1)
    s2, e2, note2 = _period(start_2, end_2)
    p1, p2 = period_stats(sid, s1, e1), period_stats(sid, s2, e2)
    change = {}
    for k in METRICS:
        a, b = p1[k], p2[k]
        if a is None or b is None or (isinstance(a, float) and np.isnan(a)):
            continue
        change[k] = {"from": a, "to": b, "diff": round(b - a, METRICS[k][2]),
                     **({"pct": round((b - a) / abs(a) * 100)} if a and k.startswith(("heat", "hours", "events"))
                        else {})}
    res = {"substation": f"ИТП-{sid}", "period_1": p1, "period_2": p2, "change": change,
           "units": {k: v[1] for k, v in METRICS.items()}}
    notes = [n for n in (note1, note2) if n]
    if notes:
        res["note"] = " ".join(notes)
    return res
