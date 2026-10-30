"""Оценка поиска норм: находит ли поиск эталонный пункт среди первых k фрагментов.

    python -m eval.eval_retrieval       # ~1 мин, нужен индекс (python -m data.norms_index)

Фрагмент считается попаданием, если он из того же документа и содержит начало эталонного пункта.
Так одинаково оцениваются обе нарезки: у фиксированных окон номеров пунктов нет.
Метрики: recall@k — доля вопросов, где эталон в первых k; MRR — средняя обратная позиция эталона (до 10).
"""
import sys
from pathlib import Path

from data.norms_index import structural
from eval.norm_questions import QUESTIONS
from itp.rag import search

OUT = Path(__file__).resolve().parent
CONFIGS = [("structural", "vector"), ("structural", "fts"), ("structural", "hybrid"),
           ("fixed", "vector"), ("fixed", "hybrid")]
STRATEGY_RU = {"structural": "по пунктам", "fixed": "окна 1000 знаков"}
MODE_RU = {"vector": "векторный", "fts": "полнотекстовый", "hybrid": "гибридный (RRF)"}
TOP = 10


def _norm(s):
    return " ".join(s.split())


def gold_snippet(q):
    """Начало эталонного пункта (60 знаков) — по нему узнаём пункт в любом фрагменте."""
    for c in structural(q["doc"]):
        if c["point"] == q["point"] and (not q.get("appendix") or f"Приложение N {q['appendix']} " in c["section"]):
            return _norm(c["text"])[:60]
    raise ValueError(f"{q['id']}: эталонный пункт не найден в тексте")


def rank(q, snippet, strategy, mode):
    hits = search(q["q"], k=TOP, mode=mode, strategy=strategy)
    for i, h in enumerate(hits, 1):
        if h["doc"] == q["doc"] and snippet in _norm(h["text"]):
            return i, hits
    return None, hits


def main():
    snippets = {q["id"]: gold_snippet(q) for q in QUESTIONS}
    results = {}
    for strategy, mode in CONFIGS:
        ranks = {}
        for q in QUESTIONS:
            ranks[q["id"]], _ = rank(q, snippets[q["id"]], strategy, mode)
        n = len(QUESTIONS)
        results[(strategy, mode)] = {
            "ranks": ranks,
            **{f"r@{k}": sum(1 for r in ranks.values() if r and r <= k) / n for k in (1, 3, 5)},
            "mrr": sum(1 / r for r in ranks.values() if r) / n,
        }
        r = results[(strategy, mode)]
        print(f"{strategy:10} {mode:7} R@1={r['r@1']:.2f} R@3={r['r@3']:.2f} R@5={r['r@5']:.2f} MRR={r['mrr']:.2f}")

    num = lambda x: f"{x:.2f}".replace(".", ",")  # noqa: E731
    lines = [
        "# Оценка поиска норм",
        "",
        f"{len(QUESTIONS)} вопросов (`eval/norm_questions.py`) с эталонным пунктом, корпус — 4 документа "
        "(`docs/norms/`), эмбеддинги bge-m3, pgvector (HNSW, косинус).",
        "",
        "| Нарезка | Поиск | Recall@1 | Recall@3 | Recall@5 | MRR |",
        "|---|---|---|---|---|---|",
    ] + [f"| {STRATEGY_RU[s]} | {MODE_RU[m]} | {num(r['r@1'])} | {num(r['r@3'])} | {num(r['r@5'])} | "
         f"{num(r['mrr'])} |" for (s, m), r in results.items()] + [
        "",
        "## Позиция эталона по вопросам (— = нет в первых 10)",
        "",
        "| ID | Вопрос | Эталон | " + " | ".join(f"{STRATEGY_RU[s]}, {m}" for s, m in CONFIGS) + " |",
        "|---|---|---|" + "---|" * len(CONFIGS),
    ]
    for q in QUESTIONS:
        gold = f"{q['doc']} п. {q['point']}" + (f" (прил. {q['appendix']})" if q.get("appendix") else "")
        cells = [str(results[c]["ranks"][q["id"]] or "—") for c in CONFIGS]
        lines.append(f"| {q['id']} | {q['q']} | {gold} | " + " | ".join(cells) + " |")
    (OUT / "RETRIEVAL.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return results


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
