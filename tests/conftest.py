"""Тесты без LLM: детерминированная часть агента (детекторы, проверки, разбор вопросов, API).

Тесты с меткой db нужны PostgreSQL с данными (python -m data.generate); без БД они пропускаются.
Качество ответов LLM проверяется отдельно — eval/ (долго, нужна модель).
"""
import os

import psycopg
import pytest

from itp.config import DB_DSN


def _db_ready():
    try:
        with psycopg.connect(DB_DSN, connect_timeout=3) as con:
            return con.execute("SELECT count(*) FROM measurements").fetchone()[0] > 0
    except psycopg.Error:
        return False


DB_READY = _db_ready()


def pytest_collection_modifyitems(items):
    if os.getenv("REQUIRE_DB") and not DB_READY:        # в CI тихий пропуск скрыл бы, что БД не поднялась
        raise pytest.UsageError("REQUIRE_DB задан, но PostgreSQL с данными недоступен")
    skip = pytest.mark.skip(reason="нет PostgreSQL с данными (python -m data.generate)")
    for item in items:
        if "db" in item.keywords and not DB_READY:
            item.add_marker(skip)
