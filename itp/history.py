"""История диалогов и оценки ответов (таблица chat_messages в той же БД).

Одна строка — пара «вопрос → ответ агента» целиком: отчёт, вызовы инструментов, время, модель.
Нужна не только чтобы история переживала обновление страницы: это журнал работы агента
(что он ответил и на каких данных) и источник реальных вопросов для новых сценариев оценки.
Таблица создаётся при первом обращении; data/generate.py её не трогает.
"""
import json
import uuid

import psycopg
from psycopg.types.json import Jsonb

from .config import DB_DSN

SCHEMA = """
CREATE TABLE IF NOT EXISTS chat_messages (
    id         serial PRIMARY KEY,
    chat_id    text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    question   text NOT NULL,
    result     jsonb NOT NULL,          -- ответ ask(): report, trace, seconds, проверки
    model      text,
    arch       text,
    feedback   smallint,                -- 1 — 👍, 0 — 👎, NULL — без оценки
    comment    text
);
CREATE INDEX IF NOT EXISTS chat_messages_chat ON chat_messages (chat_id, id);
"""

_ready = False


def _con():
    global _ready
    con = psycopg.connect(DB_DSN)
    if not _ready:
        con.execute(SCHEMA)
        con.commit()
        _ready = True
    return con


def new_chat_id():
    return uuid.uuid4().hex[:12]


def save(chat_id, question, result, model, arch="react"):
    """Сохранить пару вопрос–ответ, вернуть id строки (нужен для оценки ответа)."""
    # Через JSON-строку: в результатах инструментов бывают Timestamp и numpy-числа
    data = json.loads(json.dumps(result, ensure_ascii=False, default=str))
    with _con() as con:
        return con.execute("INSERT INTO chat_messages (chat_id, question, result, model, arch) "
                           "VALUES (%s, %s, %s, %s, %s) RETURNING id",
                           (chat_id, question, Jsonb(data), model, arch)).fetchone()[0]


def load(chat_id):
    with _con() as con:
        rows = con.execute("SELECT id, question, result, feedback FROM chat_messages "
                           "WHERE chat_id = %s ORDER BY id", (chat_id,)).fetchall()
    return [{"id": r[0], "question": r[1], "result": r[2], "feedback": r[3]} for r in rows]


def chats(limit=15):
    """Последние диалоги: id, первый вопрос, время последнего сообщения, число вопросов."""
    with _con() as con:
        rows = con.execute(
            "SELECT chat_id, (array_agg(question ORDER BY id))[1], max(created_at), count(*) "
            "FROM chat_messages GROUP BY chat_id ORDER BY max(created_at) DESC LIMIT %s", (limit,)).fetchall()
    return [{"chat_id": r[0], "title": r[1], "updated": r[2], "count": r[3]} for r in rows]


def set_feedback(message_id, value):
    with _con() as con:
        con.execute("UPDATE chat_messages SET feedback = %s WHERE id = %s", (value, message_id))
