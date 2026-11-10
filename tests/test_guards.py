"""Проверки ответа модели (guardrails): находки, источники, числа."""
from itp.agent import PLACEHOLDER, _missing_highlights, _numbers, unverified_numbers
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


def test_missing_source_restored_by_numbers_from_answer():
    trace = [{"tool": "search_norms", "result": {"results": [
        {"source": "ПТЭ (приказ Минэнерго № 511), п. 128", "text": "128. … не более чем на 5%."},
        {"source": "СанПиН 2.1.3684-21, п. 84", "text": "84. … не ниже плюс 60 °C и не выше плюс 75 °C."}]}}]
    out = ground_sources({"answer": "Должна быть от 60 до 75 °C, 55 °C — нарушение.", "sources": []}, trace,
                         question="Горячая вода 55 градусов — это нарушение?")
    assert out["sources"] == ["СанПиН 2.1.3684-21, п. 84"]
    # чисел из выдачи в ответе нет — источник не придумываем
    assert ground_sources({"answer": "Нужно проверить оборудование.", "sources": []}, trace)["sources"] == []
    # ничья — тоже без источника
    tie = [{"tool": "search_norms", "result": {"results": [
        {"source": "п. 133", "text": "133. … 0,05 МПа (0,5 кгс/см2)"}, {"source": "п. 397", "text": "397. … 0,05 МПа"}]}}]
    assert ground_sources({"answer": "Не менее чем на 0,05 МПа.", "sources": []}, tie)["sources"] == []


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


def test_missing_source_restored_by_citation_in_answer():
    trace = [{"tool": "search_norms", "result": {"results": [
        {"source": "ПТЭ (приказ Минэнерго № 511), п. 127", "text": "127. … 3%; … 5%"},
        {"source": "ПТЭ (приказ Минэнерго № 511), п. 128", "text": "128. … не более чем на 5%."}]}}]
    answer = "Согласно приказу Минэнерго № 511, п. 128, обратка не должна быть выше графика более чем на 5%."
    out = ground_sources({"answer": answer, "sources": []}, trace, question="Обратка на 4% выше — допустимо?")
    assert out["sources"] == ["ПТЭ (приказ Минэнерго № 511), п. 128"]
    # пункт без документа рядом — не ссылка
    assert ground_sources({"answer": "См. п. 128.", "sources": []}, trace)["sources"] == []


def test_missing_highlight_appended_only_when_absent():
    trace = [{"tool": "compare_periods", "result": {"highlights": [
        "Обратка выше графика более 5 %: 0 → 175 ч", "Тепло на градусо-сутки: 0,232 → 0,223"]}}]
    assert _missing_highlights("Обратка была выше графика 175 часов в феврале.", trace) == ""
    added = _missing_highlights("Тепло на градусо-сутки снизилось на 4%.", trace)
    assert "Главные изменения" in added and "0 → 175 ч" in added
    assert _missing_highlights("Что-то", [{"tool": "detect_anomalies", "result": {}}]) == ""
