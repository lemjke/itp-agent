"""Подготовка БД при запуске (Docker): данные и индекс норм — только если их ещё нет.

    python -m data.bootstrap
"""
import sys

import psycopg

from data import generate, norms_index
from itp.config import DB_DSN


def _has_rows(table):
    try:
        with psycopg.connect(DB_DSN) as con:
            return con.execute(f"SELECT EXISTS (SELECT 1 FROM {table})").fetchone()[0]
    except psycopg.Error:          # нет БД или таблицы
        return False


def main():
    if _has_rows("measurements"):
        print("Данные уже есть — генерация пропущена")
    else:
        generate.save(generate.build())
        print("Синтетические данные сгенерированы")
    if _has_rows("norm_chunks"):
        print("Индекс норм уже есть — пропущен")
    else:
        norms_index.main()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
