"""Статистический детектор (itp/ml_detect.py) и сравнение периодов (itp/compare.py)."""
import pandas as pd
import pytest

from data.generate import apply_anomaly
from eval.eval_sensitivity import base_series, run_detectors
from itp import ml_detect
from itp.tools import ToolError, with_plan


def prep(df):
    return with_plan(df.set_index("ts").copy())


@pytest.mark.parametrize("seed", range(20_000, 20_005))
def test_calibration_series_have_no_statistical_events(seed):
    # пороги — максимум на калибровочных рядах с запасом, поэтому на них событий быть не должно
    assert ml_detect.statistical(prep(base_series(seed)[0])) == []


def test_small_sensor_shift_seen_by_statistics_but_not_rules():
    df, _ = base_series(30_001)
    a = df["ts"].iloc[600]
    b = a + pd.Timedelta(hours=47)
    apply_anomaly(df, "shift", "t1", a, b, 2)                 # +2 °C — меньше допуска ±3 % от графика
    assert run_detectors(df) == []
    d = prep(df)
    events = ml_detect.statistical(d)
    assert len(events) == 1 and events[0]["start"] <= b and events[0]["end"] >= a
    assert "heat" in events[0]["features"]                    # сдвиг виден по тепловому балансу
    assert "тепловой баланс выше обычного" in events[0]["summary"]


def test_suspicions_skip_what_rules_already_found():
    df, _ = base_series(30_002)
    a = df["ts"].iloc[600]
    apply_anomaly(df, "shift", "t1", a, a + pd.Timedelta(hours=47), 8)    # большое — найдут и правила
    d = prep(df)
    rules = run_detectors(df)
    assert rules
    assert ml_detect.suspicions(d, rules, d.index[0], d.index[-1]) == []


@pytest.mark.db
def test_compare_periods_shows_return_overheat_and_weather_correction():
    from itp.compare import compare_periods
    r = compare_periods("ИТП-3", "2026-01-01", "2026-01-31", "2026-02-01", "2026-02-28")
    ch = r["change"]
    assert ch["hours_t2_out"]["from"] == 0 and ch["hours_t2_out"]["to"] > 100
    assert ch["t_outdoor_mean"]["diff"] > 0                                     # февраль теплее
    assert abs(ch["heat_per_degree_day"]["pct"]) < abs(ch["heat_gcal"]["pct"])  # поправка на погоду
    assert r["period_2"]["events_by_type"]["обратка выше графика"] == 1


@pytest.mark.db
def test_compare_periods_validates_dates():
    from itp.compare import compare_periods
    with pytest.raises(ToolError):
        compare_periods("ИТП-3", "2023-01-01", "2023-01-31", "2026-02-01", "2026-02-28")
