"""Сопоставление найденных событий с эталонной разметкой (общие правила для детекторов и агента)."""
import pandas as pd
import psycopg

from itp.config import DB_DSN

TOLERANCE = pd.Timedelta(hours=6)   # допуск ±Δt по границам событий

# Вид аномалии в эталоне → какой параметр, норму и категорию ожидаем в ответе
EXPECTED = {
    "shift":       ("t1", "T1_SCHEDULE", "sensor_fault"),
    "drift":       ("t1", "T1_SCHEDULE", "sensor_fault"),
    "spike":       (None, "DATA_VALIDITY", "sensor_fault"),
    "frozen":      (None, "DATA_VALIDITY", "sensor_fault"),
    "gap":         ("*", "DATA_CONTINUITY", "sensor_fault"),
    "underheat":   ("t1", "T1_SCHEDULE", "regime_violation"),
    "high_return": ("t2", "T2_SCHEDULE", "regime_violation"),
    "dhw_low":     ("t_dhw", "DHW_RANGE", "regime_violation"),
}


def load_truth():
    with psycopg.connect(DB_DSN) as con:
        rows = con.execute(
            "SELECT id, substation_id, category, kind, parameter, start_ts, end_ts, description "
            "FROM anomalies_truth ORDER BY id").fetchall()
    out = []
    for r in rows:
        param, req, cat = EXPECTED[r[3]]
        out.append({"id": r[0], "substation": r[1], "category": cat, "kind": r[3],
                    "parameter": param or r[4], "requirement_code": req,
                    "start": pd.Timestamp(r[5]), "end": pd.Timestamp(r[6]), "description": r[7]})
    return out


def truth_in(truth, substation, start, end):
    s, e = pd.Timestamp(start), pd.Timestamp(end) + pd.Timedelta(hours=23)
    return [t for t in truth if t["substation"] == substation and t["end"] >= s and t["start"] <= e]


def _sid(value):
    digits = "".join(ch for ch in str(value) if ch.isdigit())
    return int(digits) if digits else None


def _ts(value):
    try:
        return pd.Timestamp(value)
    except (ValueError, TypeError):
        return None


def _param_ok(found, expected):
    found = str(found).replace(" ", "").lower()
    if expected == "*":
        return found in ("*", "all", "всепараметры", "") or "все" in found
    return expected in found.split(",") or found == expected


def matches(found, truth_item):
    """found — событие детектора или пункт findings агента (substation, parameter, start, end)."""
    if _sid(found.get("substation")) != truth_item["substation"]:
        return False
    if not _param_ok(found.get("parameter", ""), truth_item["parameter"]):
        return False
    s, e = _ts(found.get("start")), _ts(found.get("end"))
    if s is None or e is None:
        return False
    return s <= truth_item["end"] + TOLERANCE and e >= truth_item["start"] - TOLERANCE


def match_sets(found_list, truth_list):
    """Жадное сопоставление один-к-одному. Возвращает пары и несопоставленные элементы."""
    pairs, used = [], set()
    for t in truth_list:
        for i, f in enumerate(found_list):
            if i not in used and matches(f, t):
                pairs.append((f, t))
                used.add(i)
                break
    matched_truth = {id(t) for _, t in pairs}
    missed = [t for t in truth_list if id(t) not in matched_truth]
    extra = [f for i, f in enumerate(found_list) if i not in used]
    return pairs, missed, extra
