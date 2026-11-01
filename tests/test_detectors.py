"""Детекторы: эталон из генератора и отсутствие ложных срабатываний на чистых рядах."""
import pytest

from eval.eval_detectors import run
from eval.eval_sensitivity import base_series, run_detectors


@pytest.mark.db
def test_all_planted_anomalies_found_without_extras():
    r = run()
    assert r["truth"] == 11
    assert r["tp"] == r["truth"], f"пропущено: {r['missed']}"
    assert not r["extra"], f"лишние события: {r['extra']}"


@pytest.mark.parametrize("seed", range(5))
def test_clean_series_has_no_events(seed):
    df, _ = base_series(seed)
    assert run_detectors(df) == []


def test_events_carry_requirement_and_summary():
    df, _ = base_series(42)
    df.loc[df.index[500], "t1"] += 12                 # одиночный выброс подачи
    events = run_detectors(df)
    assert any(ev["type"] == "spike" and ev["parameter"] == "t1" for ev in events)
    for ev in events:
        assert ev["summary"] and ev["requirement"]["code"] and ev["requirement"]["source"]
