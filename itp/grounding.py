"""Проверка находок агента по результатам инструментов (guardrail после LLM).

Каждая находка в отчёте должна соответствовать событию, которое реально вернул
detect_anomalies: тот же ИТП, совместимый параметр, пересечение по времени.
Подтверждённые находки получают каноничные даты и норму из события; остальные отбрасываются.
"""
import re

import pandas as pd

TOLERANCE = pd.Timedelta(hours=6)


def _sid(value):
    m = re.search(r"\d+", str(value))
    return int(m.group()) if m else None


def _ts(value):
    try:
        return pd.Timestamp(value)
    except (ValueError, TypeError):
        return None


def _param_ok(found, event_param):
    f = str(found).replace(" ", "").lower()
    if event_param == "*":
        return f in ("*", "all", "") or "все" in f
    return f == event_param or event_param in f.split(",") or f in event_param.split(",")


def tool_events(trace):
    """Все события, которые вернул detect_anomalies за время работы агента."""
    events = []
    for step in trace:
        res = step.get("result") or {}
        if step.get("tool") == "detect_anomalies" and "events" in res:
            events += res["events"]
    return events


def find_event(finding, events):
    sid, s, e = _sid(finding.get("substation")), _ts(finding.get("start")), _ts(finding.get("end"))
    if sid is None or s is None or e is None:
        return None
    for ev in events:
        if (_sid(ev["substation"]) == sid and _param_ok(finding.get("parameter", ""), ev["parameter"])
                and s <= _ts(ev["end"]) + TOLERANCE and e >= _ts(ev["start"]) - TOLERANCE):
            return ev
    return None


def ground(report, trace):
    """Оставляет только подтверждённые находки. Возвращает (отчёт, число отброшенных)."""
    events = tool_events(trace)
    kept, used = [], set()
    for f in report.get("findings", []):
        ev = find_event(f, events)
        if ev is None or id(ev) in used:          # нет опоры или дубликат одного события
            continue
        used.add(id(ev))
        kept.append({**f, "substation": ev["substation"], "parameter": ev["parameter"],
                     "start": ev["start"], "end": ev["end"],
                     "requirement_code": ev["requirement"]["code"],
                     "requirement_source": ev["requirement"]["source"]})
    dropped = len(report.get("findings", [])) - len(kept)
    return {**report, "findings": kept}, dropped
