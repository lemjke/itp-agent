"""Индекс нормативных документов для поиска (RAG): нарезка, эмбеддинги, pgvector.

    python -m data.norms_index          # docs/norms/*.txt → таблица norm_chunks (~5–10 мин на GPU)

Две стратегии нарезки, чтобы сравнить их на одних вопросах (eval/eval_retrieval.py):
- structural — по пунктам документа («127. …»): фрагмент = пункт с подпунктами, длинный пункт
  делится по абзацам; к тексту для эмбеддинга добавляются документ и раздел;
- fixed — окна по 1000 знаков с перекрытием 200, без учёта структуры (базовый вариант).
"""
import re
import sys
from pathlib import Path

import ollama
import psycopg

from data.norms_corpus import BY_CODE, OUT as NORMS_DIR
from itp.config import DB_DSN, EMBED_MODEL, LLM_HOST

MAX_CHARS = 1500
FIXED_SIZE, FIXED_OVERLAP = 1000, 200
POINT = re.compile(r"^(\d+(?:\(\d+\))?)\.\s")             # «127. », «15(1). »
ROW = re.compile(r"^(\d+)\.\s.*\|")                       # строка таблицы: «5. Обеспечение … | …»
EMBED_DIM = 1024                                           # bge-m3

SCHEMA = f"""
CREATE EXTENSION IF NOT EXISTS vector;
DROP TABLE IF EXISTS norm_chunks;
CREATE TABLE norm_chunks (
    id        serial PRIMARY KEY,
    strategy  text NOT NULL,          -- structural | fixed
    doc       text NOT NULL,          -- код документа (PTE511, PP354, …)
    section   text,                   -- раздел / приложение
    point     text,                   -- номер пункта или строки таблицы
    text      text NOT NULL,
    embedding vector({EMBED_DIM}) NOT NULL,
    tsv       tsvector GENERATED ALWAYS AS (to_tsvector('russian', text)) STORED
);
CREATE INDEX ON norm_chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX ON norm_chunks USING gin (tsv);
"""


def _lines(code):
    text = (NORMS_DIR / f"{code}.txt").read_text(encoding="utf-8")
    return [ln for ln in text.splitlines() if ln and not ln.startswith("# ")]


def _split_long(lines, limit=MAX_CHARS):
    """Длинный пункт → части по абзацам (абзац длиннее лимита — по предложениям)."""
    parts, cur = [], ""
    for ln in lines:
        pieces = [ln] if len(ln) <= limit else re.split(r"(?<=[.;])\s+", ln)
        for p in pieces:
            if cur and len(cur) + len(p) + 1 > limit:
                parts.append(cur)
                cur = ""
            cur = f"{cur}\n{p}" if cur else p
    if cur:
        parts.append(cur)
    return parts


def structural(code):
    """Фрагменты по пунктам: [{section, point, text}]."""
    chunks, heading, point, body = [], [], None, []
    part, roman = "", ""              # часть документа (приказ, правила, приложение) и раздел внутри неё

    def flush():
        section = f"{part[:120]} / {roman[:120]}" if roman else part[:200]
        if body:
            for i, piece in enumerate(_split_long(body)):
                chunks.append({"section": section, "point": point, "text": piece if i == 0 or not point
                               else f"{point}. (продолжение) {piece}"})

    for ln in _lines(code):
        if ln.startswith("##"):
            heading.append(ln[3:])
            continue
        if heading:                       # подряд идущие строки-заголовки — один заголовок
            title = " ".join(heading)
            anchor = re.search(r"(Приложение\s+N|Утвержден|ПРАВИЛА|САНИТАРНЫЕ ПРАВИЛА|ТРЕБОВАНИЯ|ИЗМЕНЕНИЯ)",
                               title, re.I)
            if re.match(r"^[IVXL]+\.", title):
                roman = title
            elif anchor or not part:          # «Министр … Утверждены приказом …» — начало новой части
                part, roman = title[anchor.start() if anchor else 0:], ""
            heading = []
        if ln.startswith("<") or set(ln) <= {"-"}:          # сноски и разделители
            continue
        m = ROW.match(ln) or POINT.match(ln)
        if m:
            flush()
            point, body = m.group(1), [ln]
        else:
            body.append(ln)
    flush()
    return chunks


def fixed(code):
    text = "\n".join(ln.lstrip("# ") for ln in _lines(code))
    step = FIXED_SIZE - FIXED_OVERLAP
    return [{"section": None, "point": None, "text": text[i:i + FIXED_SIZE]}
            for i in range(0, max(len(text) - FIXED_OVERLAP, 1), step)]


def embed_text(doc, chunk):
    """Что подаётся в модель эмбеддингов: контекст документа и раздела + сам текст."""
    section = re.sub(r"^Утвержден\w* .*? N \d+\s*", "", chunk["section"] or "", flags=re.I)
    head = BY_CODE[doc]["short"] + (f". {section}" if section else "")
    return f"{head}\n{chunk['text']}"


def embed(client, texts, batch=32):
    out = []
    for i in range(0, len(texts), batch):
        out += client.embed(model=EMBED_MODEL, input=texts[i:i + batch]).embeddings
        print(f"\r  эмбеддинги {min(i + batch, len(texts))}/{len(texts)}", end="", flush=True)
    print()
    return out


def main():
    client = ollama.Client(host=LLM_HOST)
    rows = []
    for strategy, chunker in (("structural", structural), ("fixed", fixed)):
        for code in BY_CODE:
            chunks = chunker(code)
            print(f"{strategy:10} {code:11} фрагментов: {len(chunks)}")
            vecs = embed(client, [embed_text(code, c) for c in chunks])
            rows += [(strategy, code, c["section"], c["point"], c["text"], str(v)) for c, v in zip(chunks, vecs)]
    with psycopg.connect(DB_DSN) as con:
        con.execute(SCHEMA)
        with con.cursor() as cur:
            cur.executemany("INSERT INTO norm_chunks (strategy, doc, section, point, text, embedding) "
                            "VALUES (%s, %s, %s, %s, %s, %s::vector)", rows)
    print(f"Записано фрагментов: {len(rows)}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
