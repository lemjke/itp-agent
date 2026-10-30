"""Общие настройки прототипа. Значения можно переопределить переменными окружения.

Пароль PostgreSQL: переменная окружения PGPASSWORD или файл itp/local_config.py
со строкой DB_PASSWORD = "..." (файл в .gitignore).
"""
import os

try:
    from .local_config import DB_PASSWORD
except ImportError:
    DB_PASSWORD = os.getenv("PGPASSWORD", "")

_auth = f"user=postgres password={DB_PASSWORD}" if DB_PASSWORD else "user=postgres"
DB_DSN = os.getenv("ITP_DB_DSN", f"host=localhost port=5432 dbname=itp_db {_auth}")
DB_ADMIN_DSN = os.getenv("ITP_DB_ADMIN_DSN", f"host=localhost port=5432 dbname=postgres {_auth}")

LLM_MODEL = os.getenv("ITP_LLM_MODEL", "qwen2.5:7b")
LLM_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
EMBED_MODEL = os.getenv("ITP_EMBED_MODEL", "bge-m3")       # эмбеддинги для поиска норм

# Климат Новосибирска (СП 131.13330): расчётная температура наружного воздуха
# для проектирования отопления и расчётная внутренняя температура.
T_OUT_DESIGN = -37.0
T_IN_DESIGN = 20.0

# Упрощённый температурный график 130/70 с нижним изломом 70 °C (под ГВС).
# В реальной системе — утверждённый график теплоснабжающей организации.
T1_DESIGN = 130.0
T2_DESIGN = 70.0
T1_BREAK = 70.0

# Период синтетических данных и «текущий момент» для агента
DATA_START = "2026-01-01 00:00"
DATA_END = "2026-02-28 23:00"
NOW = "2026-02-28 23:00"
