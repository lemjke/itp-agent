"""Статистические и ML-детекторы в дополнение к правилам (itp/tools.py: detect_events).

Правила сравнивают параметр с нормой (подача ±3 % от графика и т. п.) и поэтому не видят того, что
норму не нарушает: смещение датчика подачи на 1–2 °C неотличимо от допустимого отклонения режима.
Здесь — проверка согласованности параметров между собой и с собственной нормой ряда:

- признаки по часам: отклонения подачи/обратки от графика, тепловой баланс
  (расход × (подача − обратка) на единицу нагрузки по погоде — должен быть постоянным для ИТП),
  ГВС, перепад давлений, скачки относительно соседних часов, «застывание» (малый разброс);
- statistical — устойчивые z-оценки признаков (медиана и MAD ряда), событие — часы, где хоть один
  признак выходит за свой порог (у скачков и «застывания» хвосты тяжелее, чем у уровней);
- iforest — Isolation Forest на тех же z-оценках, событие — часы с аномальностью выше порога.

Пороги подбирает eval/eval_ml.py на отдельных чистых рядах и сохраняет в itp/ml_thresholds.json.
События без категории и нормы: это «подозрение — проверить», а не нарушение.
Пропуски данных — только правила (ML их не видит).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from .schedule import load_ratio

THRESHOLDS_FILE = Path(__file__).with_name("ml_thresholds.json")

LEVEL = ["r1", "r2", "heat", "t_dhw", "dp"]                    # уровни — сглаживаются за 6 ч
JUMP = ["j_t1", "j_t2", "j_t_dhw", "j_flow"]                    # скачки — по часам
STILL = ["s_t1", "s_t2", "s_t_dhw", "s_flow"]                   # «застывание» — разброс за 6 ч
FEATURES = LEVEL + JUMP + STILL
FEATURE_RU = {
    "r1": "подача относительно графика", "r2": "обратка относительно графика",
    "heat": "тепловой баланс (расход × перепад температур к нагрузке)", "t_dhw": "температура ГВС",
    "dp": "перепад давлений", "j_t1": "скачок подачи", "j_t2": "скачок обратки", "j_t_dhw": "скачок ГВС",
    "j_flow": "скачок расхода", "s_t1": "застывшая подача", "s_t2": "застывшая обратка",
    "s_t_dhw": "застывшая ГВС", "s_flow": "застывший расход",
}

MIN_GAP_H = 3          # часы-кандидаты ближе этого сливаются в одно событие
MIN_STILL_H = 3        # событие только из признаков «застывания» — не короче этого
IF_TREES = 100


def thresholds():
    """{"z": {признак: порог |z|}, "iforest": порог} — из калибровки (eval/eval_ml.py)."""
    return json.loads(THRESHOLDS_FILE.read_text(encoding="utf-8"))


def features(df):
    """Признаки по часам. df — с индексом ts и плановыми t1_plan/t2_plan (tools.with_plan)."""
    x = df.interpolate(limit=3).ffill().bfill()                 # ML не работает с пропусками
    t_smooth = x["t_outdoor"].rolling(6, min_periods=1).mean()
    load = np.maximum(load_ratio(t_smooth.values), 0.15)
    f = pd.DataFrame(index=df.index)
    f["r1"] = x["t1"] - x["t1_plan"]
    f["r2"] = x["t2"] - x["t2_plan"]
    f["heat"] = np.log(np.maximum(x["flow"] * (x["t1"] - x["t2"]), 1e-3) / load)
    f["t_dhw"] = x["t_dhw"]
    f["dp"] = x["p1"] - x["p2"]
    f[LEVEL] = f[LEVEL].rolling(6, min_periods=1, center=True).mean()
    for p in ("t1", "t2", "t_dhw", "flow"):
        neighbours = (x[p].shift(1) + x[p].shift(-1)) / 2
        f[f"j_{p}"] = (x[p] - neighbours).abs()
        # только полное окно: на краю ряда разброс по 3–4 точкам ложно выглядит «застывшим»
        f[f"s_{p}"] = -np.log(x[p].rolling(6, min_periods=6, center=True).std() + 1e-3)
    return f.ffill().bfill()[FEATURES]


def zscores(f):
    """Устойчивые z-оценки относительно собственной нормы ряда (медиана, MAD)."""
    med = f.median()
    mad = (f - med).abs().median() * 1.4826
    return ((f - med) / mad.replace(0, np.nan)).fillna(0.0).clip(-50, 50)


def _runs(mask, index):
    """Подряд идущие часы (с промежутками меньше MIN_GAP_H) → [(начало, конец)]."""
    hours = index[mask]
    runs = []
    for t in hours:
        if runs and t - runs[-1][1] <= pd.Timedelta(hours=MIN_GAP_H):
            runs[-1][1] = t
        else:
            runs.append([t, t])
    return [tuple(r) for r in runs]


def _shift_text(p, f, a, b):
    """«подача выше обычного на 1,8 °C» — насколько уровень признака в событии отличается от обычного."""
    d = float(f[p].loc[a:b].mean() - f[p].median())
    if p == "heat":
        return f"тепловой баланс {'выше' if d > 0 else 'ниже'} обычного на {abs(np.expm1(d)) * 100:.0f} %"
    unit = "бар" if p == "dp" else "°C"
    return f"{FEATURE_RU[p]} {'выше' if d > 0 else 'ниже'} обычного на {abs(d):.1f} {unit}".replace(".", ",")


def _events(kind, mask, z, index, score=None, f=None):
    out = []
    for a, b in _runs(mask, index):
        zz = z.loc[a:b].abs()
        top = zz.max().sort_values(ascending=False)
        main = [p for p in top.index[:2] if top[p] >= 3] or [top.index[0]]
        # «Застывший» датчик по определению держится часами; один час с малым разбросом — шум
        if all(p in STILL for p in main) and b - a < pd.Timedelta(hours=MIN_STILL_H - 1):
            continue
        parts = [_shift_text(p, f, a, b) if f is not None and p in LEVEL else FEATURE_RU[p] for p in main]
        out.append({
            "type": f"ml_{kind}", "detector": kind, "start": a, "end": b, "features": main,
            "summary": "Необычно для этого ИТП: " + "; ".join(parts) + f" (до {top.iloc[0]:.0f} σ). Норму это может "
                       "не нарушать — стоит проверить датчики.",
            **({"score": round(float(score.loc[a:b].max()), 2)} if score is not None else {}),
        })
    return out


def statistical(df, limits=None, z=None):
    limits = limits or thresholds()["z"]
    f = features(df)
    z = zscores(f) if z is None else z
    over = z.abs() > pd.Series(limits)[z.columns]
    return _events("statistical", over.any(axis=1).values, z, df.index, f=f)


def suspicions(df, rule_events, s, e, tol=pd.Timedelta(hours=6)):
    """Подозрения для агента: события статистического детектора в периоде, которых нет среди правил."""
    out = []
    for ev in statistical(df):
        if ev["end"] < s or ev["start"] > e:
            continue
        if any(ev["start"] <= pd.Timestamp(r["end"]) + tol and ev["end"] >= pd.Timestamp(r["start"]) - tol
               for r in rule_events):
            continue
        out.append({"start": ev["start"].strftime("%Y-%m-%d %H:%M"), "end": ev["end"].strftime("%Y-%m-%d %H:%M"),
                    "summary": ev["summary"]})
    return out


def iforest_scores(df, seed=0, z=None):
    """Аномальность часа по Isolation Forest, 0…1 (выше — необычнее)."""
    z = zscores(features(df)) if z is None else z
    forest = IsolationForest(n_estimators=IF_TREES, random_state=seed).fit(z.values)
    return z, pd.Series(-forest.score_samples(z.values), index=df.index)


def iforest(df, threshold=None, seed=0, z=None):
    threshold = threshold or thresholds()["iforest"]
    z, score = iforest_scores(df, seed, z)
    return _events("iforest", (score > threshold).values, z, df.index, score)
