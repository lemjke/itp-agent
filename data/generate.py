"""Генератор синтетических данных ИТП с внесёнными аномалиями и эталонной разметкой.

Запуск:  python -m data.generate          (из корня проекта)
Создаёт БД itp_db (если её нет), схему и заполняет таблицы.
"""
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from itp.config import DB_ADMIN_DSN, DB_DSN, DATA_START, DATA_END  # noqa: E402
from itp.norms import REQUIREMENTS  # noqa: E402
from itp.schedule import graph, load_ratio  # noqa: E402

SEED = 42

SUBSTATIONS = [
    (1, "ИТП-1", "ул. Учебная, 12 (школа)", "школа"),
    (2, "ИТП-2", "ул. Солнечная, 5 (детский сад)", "детский сад"),
    (3, "ИТП-3", "пр. Северный, 48 (жилой дом)", "жилой дом"),
    (4, "ИТП-4", "ул. Больничная, 3 (поликлиника)", "поликлиника"),
    (5, "ИТП-5", "ул. Речная, 21 (жилой дом)", "жилой дом"),
]

PARAMS = ["t_outdoor", "t1", "t2", "t_dhw", "p1", "p2", "flow"]

# Сценарий аномалий: (ИТП, категория, вид, параметр, начало, конец, описание).
# Последняя неделя февраля насыщена событиями — на ней удобно показывать агента.
PLANNED = [
    (1, "sensor_fault", "shift", "t1", "2026-02-22 08:00", "2026-02-24 20:00",
     "Смещение показаний датчика t1 на +7 °C; t2 и расход в норме"),
    (1, "sensor_fault", "spike", "t2", "2026-01-12 14:00", "2026-01-12 14:00",
     "Одиночный выброс t2 +22 °C"),
    (2, "regime_violation", "underheat", "t1", "2026-02-25 00:00", "2026-02-26 12:00",
     "Недогрев теплоносителя от источника: t1 ниже графика на 12 °C, t2 снижена на 5 °C"),
    (2, "regime_violation", "dhw_low", "t_dhw", "2026-02-10 06:00", "2026-02-10 11:00",
     "Температура ГВС опустилась до ~55 °C"),
    (3, "regime_violation", "high_return", "t2", "2026-02-21 00:00", "2026-02-27 23:00",
     "Завышенная обратка: t2 выше графика на 7 °C, расход +10 % (разрегулировка системы отопления)"),
    (3, "sensor_fault", "frozen", "t_dhw", "2026-02-26 03:00", "2026-02-26 20:00",
     "«Замёрзшее» значение датчика ГВС"),
    (3, "sensor_fault", "gap", "*", "2026-02-15 10:00", "2026-02-15 16:00",
     "Пропуск всех измерений (обрыв связи)"),
    (4, "sensor_fault", "drift", "t1", "2026-02-18 00:00", "2026-02-23 23:00",
     "Линейный дрейф датчика t1 от 0 до +8 °C"),
    (4, "sensor_fault", "spike", "p1", "2026-01-28 09:00", "2026-01-28 09:00",
     "Одиночный выброс давления p1 +2,5 бар"),
    (5, "sensor_fault", "gap", "*", "2026-01-20 02:00", "2026-01-20 09:00",
     "Пропуск всех измерений (обрыв связи)"),
    (5, "sensor_fault", "spike", "t1", "2026-02-27 15:00", "2026-02-27 15:00",
     "Одиночный выброс t1 −25 °C"),
]


def outdoor_temperature(idx, rng):
    """Наружная температура: сезонный тренд + погодные волны + суточный ход."""
    days = (idx - idx[0]).total_seconds().values / 86400
    trend = -18 + 5 * days / days[-1]                     # конец февраля теплее
    n_days = int(np.ceil(days[-1])) + 2
    weather = np.zeros(n_days)
    for d in range(1, n_days):                            # AR(1): затяжные похолодания и оттепели
        weather[d] = 0.75 * weather[d - 1] + rng.normal(0, 3.5)
    weather_h = np.interp(days, np.arange(n_days), weather)
    diurnal = 3 * np.sin(2 * np.pi * (idx.hour.values - 9) / 24)
    return trend + weather_h + diurnal + rng.normal(0, 0.4, len(idx))


def simulate_substation(sid, idx, t_out, rng):
    t_smooth = pd.Series(t_out).rolling(6, min_periods=1).mean().values   # инерция здания
    t1_plan, t2_plan = graph(t_smooth)
    n = len(idx)
    bias1, bias2 = rng.normal(0, 0.5), rng.normal(0, 0.6)
    t1 = t1_plan + bias1 + rng.normal(0, 0.6, n)
    t2 = t2_plan + bias2 + rng.normal(0, 0.5, n)
    t_dhw = 64 + 1.0 * np.sin(2 * np.pi * (idx.hour.values - 19) / 24) + rng.normal(0, 0.6, n)
    p1 = 6.2 + rng.normal(0, 0.05, n)
    p2 = 4.1 + rng.normal(0, 0.05, n)
    q_design = rng.uniform(0.4, 1.2)                      # Гкал/ч
    q = q_design * np.maximum(load_ratio(t_smooth), 0.15)
    flow = q * 1000 / np.maximum(t1 - t2, 5) + rng.normal(0, 0.1, n)
    return pd.DataFrame({
        "ts": idx, "substation_id": sid, "t_outdoor": t_out,
        "t1": t1, "t2": t2, "t_dhw": t_dhw, "p1": p1, "p2": p2, "flow": flow,
    })


def apply_anomaly(df, kind, param, start, end, magnitude=None):
    """Вносит аномалию. magnitude — сила (°C или бар); None — значения по умолчанию из сценария.

    shift / drift / spike / high_return — добавка к параметру (для drift — к концу интервала);
    underheat — снижение t1 (t2 снижается на 5/12 от него); dhw_low — на сколько ГВС ниже 60 °C;
    frozen и gap силы не имеют — их «сила» задаётся длительностью интервала.
    """
    m = (df["ts"] >= start) & (df["ts"] <= end)
    k = int(m.sum())
    if kind == "shift":
        df.loc[m, param] += 7 if magnitude is None else magnitude
    elif kind == "drift":
        df.loc[m, param] += np.linspace(0, 8 if magnitude is None else magnitude, k)
    elif kind == "spike":
        df.loc[m, param] += ({"t1": -25, "t2": 22, "p1": 2.5}.get(param, 15) if magnitude is None else magnitude)
    elif kind == "frozen":
        df.loc[m, param] = df.loc[m, param].iloc[0]
    elif kind == "gap":
        df.loc[m, [p for p in PARAMS if p != "t_outdoor"]] = np.nan
    elif kind == "underheat":
        d = 12 if magnitude is None else magnitude
        df.loc[m, "t1"] -= d
        df.loc[m, "t2"] -= d * 5 / 12
    elif kind == "high_return":
        d = 7 if magnitude is None else magnitude
        df.loc[m, "t2"] += d
        df.loc[m, "flow"] *= 1 + d / 70
    elif kind == "dhw_low":
        depth = 5 if magnitude is None else magnitude
        df.loc[m, "t_dhw"] = 60 - depth + np.random.default_rng(1).normal(0, 0.5, k)


def build():
    rng = np.random.default_rng(SEED)
    idx = pd.date_range(DATA_START, DATA_END, freq="h")
    t_out = outdoor_temperature(idx, rng)

    frames = []
    for sid, *_ in SUBSTATIONS:
        df = simulate_substation(sid, idx, t_out, rng)
        for a_sid, _cat, kind, param, start, end, _ in PLANNED:
            if a_sid == sid:
                apply_anomaly(df, kind, param, pd.Timestamp(start), pd.Timestamp(end))
        frames.append(df)
    data = pd.concat(frames, ignore_index=True)
    data[PARAMS] = data[PARAMS].round(2)
    return data


def ensure_database():
    dbname = dict(p.split("=", 1) for p in DB_DSN.split())["dbname"]
    with psycopg.connect(DB_ADMIN_DSN, autocommit=True) as con:
        exists = con.execute("SELECT 1 FROM pg_database WHERE datname = %s", (dbname,)).fetchone()
        if not exists:
            con.execute(f'CREATE DATABASE "{dbname}"')


def save(data):
    ensure_database()
    schema = (ROOT / "db" / "schema.sql").read_text(encoding="utf-8")
    with psycopg.connect(DB_DSN) as con:
        con.execute(schema)
        with con.cursor() as cur:
            cur.executemany("INSERT INTO substations VALUES (%s, %s, %s, %s)", SUBSTATIONS)
            cur.executemany(
                "INSERT INTO requirements (code, parameter, title, rule, source) VALUES (%s, %s, %s, %s, %s)",
                [(r["code"], r["parameter"], r["title"], r["rule"], r["source"]) for r in REQUIREMENTS],
            )
            cur.executemany(
                "INSERT INTO anomalies_truth (substation_id, category, kind, parameter, start_ts, end_ts, description) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s)",
                PLANNED,
            )
            buf = io.StringIO()
            data.to_csv(buf, index=False, header=False, na_rep="")
            buf.seek(0)
            cols = ", ".join(data.columns)
            with cur.copy(f"COPY measurements ({cols}) FROM STDIN WITH (FORMAT csv, NULL '')") as copy:
                copy.write(buf.read())


if __name__ == "__main__":
    data = build()
    save(data)
    print(f"Сгенерировано {len(data)} строк измерений для {len(SUBSTATIONS)} ИТП, "
          f"{len(PLANNED)} аномалий в эталонной разметке.")
