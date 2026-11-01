"""Проверки ответа модели (guardrails): находки, источники, числа."""
from itp.agent import PLACEHOLDER, _numbers, unverified_numbers
from itp.grounding import ground, ground_sources

EVENT = {"substation": "ИТП-3", "parameter": "t2", "start": "2026-02-21 00:00", "end": "2026-02-27 23:00",
         "requirement": {"code": "T2_SCHEDULE", "source": "ПТЭ, п. 128"}}
TRACE = [{"tool": "detect_anomalies", "args": {}, "result": {"events": [EVENT]}}]


def finding(**kw):
    return {"substation": "ИТП-3", "parameter": "t2", "start": "2026-02-21", "end": "2026-02-27",
            "what": "обратка выше графика", "category": "regime_violation", "hypothesis": "",
            "requirement_code": "ВЫДУМАНО", "requirement_source": "Приказ № 1234", **kw}


def test_grounded_finding_gets_canonical_dates_and_norm():
    report, dropped = ground({"findings": [finding()]}, TRACE)
    assert dropped == 0
    f = report["findings"][0]
    assert (f["start"], f["requirement_code"], f["requirement_source"]) == (EVENT["start"], "T2_SCHEDULE", "ПТЭ, п. 128")


def test_finding_without_detector_event_is_dropped():
    report, dropped = ground({"findings": [finding(substation="ИТП-123"), finding(parameter="t1")]}, TRACE)
    assert (report["findings"], dropped) == ([], 2)


def test_duplicate_of_one_event_is_dropped():
    report, dropped = ground({"findings": [finding(), finding(start="2026-02-22")]}, TRACE)
    assert (len(report["findings"]), dropped) == (1, 1)


def test_finding_within_tolerance_is_kept():
    report, _ = ground({"findings": [finding(start="2026-02-27 23:00", end="2026-02-28 04:00")]}, TRACE)
    assert len(report["findings"]) == 1


NORM_TRACE = [{"tool": "search_norms", "args": {}, "result": {"results": [
    {"source": "Правила предоставления коммунальных услуг (ПП № 354), приложение N 1, п. 4", "text": ""},
    {"source": "СанПиН 2.1.3684-21, п. 84", "text": ""}]}}]


def test_sources_only_from_search_results():
    out = ground_sources({"sources": ["ПП № 354, п. 4", "СанПиН 2.1.3684-21, п. 84", "ПП № 354, п. 7",
                                      "Приказ № 1234, п. 4"]}, NORM_TRACE)
    assert out["sources"] == ["Правила предоставления коммунальных услуг (ПП № 354), приложение N 1, п. 4",
                              "СанПиН 2.1.3684-21, п. 84"]


def test_no_sources_without_search():
    assert ground_sources({"sources": ["СанПиН 2.1.3684-21, п. 84"]}, TRACE)["sources"] == []


def test_numbers_are_normalized():
    # Одно число в разной записи (запятая/точка, лишние нули) — одно и то же значение
    # (_numbers хранит и исходную запись, и без хвостовых нулей — сравнение идёт по пересечению)
    assert _numbers("54,6 °C, 0,05 МПа и 5%") <= _numbers("54.6 °C, 0.050 МПа и 5.0%")
    assert not _numbers("54,6") & _numbers("54,7")


def test_unverified_numbers_are_those_absent_from_tools_and_question():
    trace = [{"tool": "query_sensors", "result": {"period_summary": {"t2": {"mean": 53.8}}}}]
    report = {"answer": "Средняя обратка на ИТП-3 — 53,8 °C, максимум 71,2 °C.", "findings": []}
    assert unverified_numbers(report, "Какая обратка на ИТП-3?", trace) == ["71.2"]


def test_placeholder_detected():
    assert PLACEHOLDER.search("Средняя — ${period_summary.t2.mean} °C")
    assert not PLACEHOLDER.search("Средняя — 53,8 °C")
