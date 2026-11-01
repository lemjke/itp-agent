"""Разбор вопроса без LLM: периоды, ИТП по адресу, самостоятельный ли вопрос, нарезка норм."""
import pandas as pd
import pytest

from data.norms_index import structural
from itp.agent import self_contained
from itp.rag import label
from itp.tools import ToolError, _period, _substation_id


def test_date_without_time_means_end_of_day():
    s, e, note = _period("2026-02-22", "2026-02-28")
    assert (s, e, note) == (pd.Timestamp("2026-02-22"), pd.Timestamp("2026-02-28 23:00"), None)


def test_period_partly_outside_data_is_clipped_with_note():
    s, e, note = _period("2025-12-01", "2026-02-28")
    assert s == pd.Timestamp("2026-01-01") and "сокращён" in note


def test_wrong_year_gets_hint():
    with pytest.raises(ToolError, match="ошибка в годе.*2026-02-01"):
        _period("2023-02-01", "2023-02-28")


def test_too_long_period_rejected():
    with pytest.raises(ToolError, match="не длиннее"):
        _period("2026-01-01", "2026-06-30")


@pytest.mark.db
@pytest.mark.parametrize("text, sid", [("ИТП-3", 3), ("итп 5", 5), ("школа на Учебной", 1),
                                       ("дом на Северном", 3), ("поликлиника на Больничной", 4)])
def test_substation_by_number_or_address(text, sid):
    assert _substation_id(text) == sid


@pytest.mark.db
def test_unknown_substation_rejected():
    with pytest.raises(ToolError):
        _substation_id("ИТП-123")


@pytest.mark.parametrize("q", ["Что не так с ИТП-3 за последнюю неделю?", "Проверь ИТП-1 и ИТП-5 в феврале",
                               "чо там с итп 2 10 февраля", "Any issues at substation 1 in January?"])
def test_complete_questions_are_self_contained(q):
    assert self_contained(q)


@pytest.mark.db          # без номера ИТП проверяется адрес — по таблице substations
@pytest.mark.parametrize("q", ["А за январь?", "А на ИТП-4 за тот же период?", "Подробнее про ИТП-3",
                               "А максимальная?", "Какой допуск по температуре подачи?"])
def test_follow_ups_and_norm_questions_go_to_model(q):
    assert not self_contained(q)


def test_structural_chunks_keep_points_and_sections():
    chunks = {c["point"]: c for c in structural("PTE511") if c["point"] in ("127", "128")}
    assert "3%" in chunks["127"]["text"] and "5%" in chunks["128"]["text"]
    assert "V. Требования к диспетчерскому управлению" in chunks["127"]["section"]


def test_appendix_points_are_distinguished():
    fives = [c for c in structural("PP354") if c["point"] == "5"]
    assert any(c["section"].startswith("Приложение N 1 ") for c in fives)
    assert len({c["section"] for c in fives}) > 1          # п. 5 есть в правилах и в нескольких приложениях


def test_source_label():
    assert label("PP354", "Приложение N 1 к Правилам …", "4") == \
        "Правила предоставления коммунальных услуг (ПП № 354), приложение N 1, п. 4"
    assert label("SANPIN3684", "Утверждены …", "84") == "СанПиН 2.1.3684-21, п. 84"
