"""Поиск по нормативным документам (RAG): pgvector + полнотекстовый поиск PostgreSQL.

Индекс строится скриптом data/norms_index.py (таблица norm_chunks).
Режимы: vector — косинусная близость эмбеддингов bge-m3; fts — полнотекстовый поиск (русская морфология);
hybrid — оба списка, объединённые по рангам (Reciprocal Rank Fusion).
"""
import re

import ollama
import psycopg

from data.norms_corpus import BY_CODE
from .config import DB_DSN, EMBED_MODEL, LLM_HOST

CANDIDATES = 30          # сколько кандидатов берёт каждый метод перед объединением
RRF_K = 60               # стандартная константа RRF

_client = None


def _embed(text):
    # Эмбеддинг запроса — на CPU (~0,05 с). На GPU 6 ГБ модель эмбеддингов вытесняет LLM,
    # и каждая смена моделей стоит 4–8 с.
    global _client
    _client = _client or ollama.Client(host=LLM_HOST)
    return _client.embed(model=EMBED_MODEL, input=[text], options={"num_gpu": 0}).embeddings[0]


def _vector(con, query, strategy, n):
    q = str(_embed(query))
    return con.execute("SELECT id FROM norm_chunks WHERE strategy = %s "
                       "ORDER BY embedding <=> %s::vector LIMIT %s", (strategy, q, n)).fetchall()


def _fts(con, query, strategy, n):
    # Любое из слов запроса (OR), ранжирование по плотности совпадений: вопросы пишут своими словами,
    # и требование «все слова сразу» почти ничего не находит.
    return con.execute(
        "WITH q AS (SELECT NULLIF(replace(plainto_tsquery('russian', %s)::text, '&', '|'), '')::tsquery AS q) "
        "SELECT id FROM norm_chunks, q WHERE strategy = %s AND tsv @@ q.q "
        "ORDER BY ts_rank_cd(tsv, q.q) DESC LIMIT %s", (query, strategy, n)).fetchall()


def label(doc, section, point):
    """«ПТЭ объектов теплоснабжения (приказ Минэнерго № 511), п. 127» или «…, приложение N 1, п. 5»."""
    out = BY_CODE[doc]["short"]
    m = re.match(r"Приложение\s+N\s*([\d()]+)", section or "")
    if m:
        out += f", приложение N {m.group(1)}"
    return out + (f", п. {point}" if point else "")


def search(query, k=4, mode="hybrid", strategy="structural"):
    """Найти k фрагментов норм, наиболее подходящих к запросу."""
    with psycopg.connect(DB_DSN) as con:
        if mode == "vector":
            ids = [r[0] for r in _vector(con, query, strategy, k)]
        elif mode == "fts":
            ids = [r[0] for r in _fts(con, query, strategy, k)]
        else:
            score = {}
            for ranked in (_vector(con, query, strategy, CANDIDATES), _fts(con, query, strategy, CANDIDATES)):
                for rank, (cid,) in enumerate(ranked):
                    score[cid] = score.get(cid, 0) + 1 / (RRF_K + rank + 1)
            ids = sorted(score, key=score.get, reverse=True)[:k]
        if not ids:
            return []
        rows = {r[0]: r[1:] for r in con.execute(
            "SELECT id, doc, section, point, text FROM norm_chunks WHERE id = ANY(%s)", (ids,)).fetchall()}
    return [{"source": label(doc, section, point), "doc": doc, "section": section, "point": point, "text": text}
            for doc, section, point, text in (rows[i] for i in ids)]
