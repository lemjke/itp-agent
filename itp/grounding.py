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


def ground_sources(report, trace, question=""):
    """Источники норм — только те пункты, которые реально вернул search_norms (без выдуманных ссылок).

    Модель может сократить подпись («ПП № 354, п. 4»), поэтому источник засчитывается, если совпадает
    с найденным целиком или совпадают номер пункта и номер документа (511, 354, 1034, 3684).
    В отчёт попадает полная подпись найденного пункта.
    """
    results = [r for s in trace if s.get("tool") == "search_norms" for r in (s.get("result") or {}).get("results", [])]
    found = list(dict.fromkeys(r["source"] for r in results))
    kept = []
    for src in map(str, report.get("sources") or []):
        point = re.search(r"п\.\s*([\d()]+)\s*$", src)
        docs = set(re.findall(r"\b\d{3,4}\b", src[:point.start()] if point else src))
        for f in found:
            if src == f or (point and docs and f.endswith(f"п. {point.group(1)}")
                            and docs & set(re.findall(r"\b\d{3,4}\b", f))):
                kept.append(f)
                break
    if not kept and results:
        kept = _source_by_citation(report.get("answer", ""), found) or \
            _source_by_numbers(report.get("answer", ""), question, results)
    return {**report, "sources": list(dict.fromkeys(kept))}


def _source_by_citation(answer, found):
    """Источник назван в тексте ответа («приказ Минэнерго № 511, п. 128»), но не в sources.

    Номер пункта должен стоять рядом с номером документа; засчитывается, только если подходит один пункт.
    """
    out = []
    for m in re.finditer(r"(?:п\.|пункт[еау]?)\s*(\d+)", answer):
        docs = set(re.findall(r"\b\d{3,4}\b", answer[max(0, m.start() - 120):m.end() + 120])) - {m.group(1)}
        out += [f for f in found if f.endswith(f"п. {m.group(1)}") and docs & set(re.findall(r"\b\d{3,4}\b", f))]
    out = list(dict.fromkeys(out))
    return out if len(out) == 1 else []


def _nums(text):
    """Числа текста: запятая → точка, хвостовые нули дробной части убраны («0,050» = «0.05», «60» = «60»)."""
    out = set()
    for n in re.findall(r"\d+(?:[.,]\d+)?", str(text)):
        n = n.replace(",", ".")
        out.add(n.rstrip("0").rstrip(".") if "." in n else n)
    return out


def _source_by_numbers(answer, question, results):
    """Модель не указала источник: берём найденный пункт, в тексте которого есть числа из ответа.

    Числа из вопроса не считаются («55 градусов — это нарушение?»), номера пунктов тоже. Источник ставится,
    только если такой пункт один: при ничьей (одни и те же 0,05 МПа в двух пунктах) или без совпадений
    его нет — неверная ссылка хуже отсутствующей.
    """
    wanted = _nums(answer) - _nums(question)
    scores = {}
    for r in results:
        body = re.sub(r"^\s*[\d()]+\.\s*", "", r["text"])          # без номера пункта в начале
        scores[r["source"]] = max(scores.get(r["source"], 0), len(wanted & _nums(body)))
    top = max(scores.values(), default=0)
    best = [s for s, v in scores.items() if v == top]
    return best if top and len(best) == 1 else []


def ground(report, trace, question=""):
    """Оставляет только подтверждённые находки. Возвращает (отчёт, число отброшенных)."""
    report = ground_sources(report, trace, question)
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
