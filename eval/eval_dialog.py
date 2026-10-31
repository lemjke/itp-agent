"""Оценка учёта контекста диалога: уточняющие вопросы («А за январь?»).

    python -m eval.eval_dialog            # ~45 мин: 3 способа × 16 диалогов + проверка самостоятельных вопросов
    python -m eval.eval_dialog --report   # только пересобрать eval/DIALOG.md из последнего прогона

Для каждого способа (itp.agent.CONTEXT_MODES) предыдущие вопросы диалога задаются агенту по-настоящему,
их ответы попадают в историю, затем задаётся уточнение и оценивается так же, как обычный сценарий.
Отдельно: не портит ли переписывание самостоятельные вопросы (40 вопросов из eval/scenarios.py с чужой
историей) — переписанный вопрос должен совпасть с исходным.
"""
import json
import sys
import time
from pathlib import Path

import ollama

from eval.dialogs import DIALOGS, HELD_OUT
from eval.eval_agent import evaluate
from eval.matching import load_truth
from eval.scenarios import SCENARIOS
from itp.agent import CONTEXT_MODES, ask, contextualize
from itp.config import LLM_HOST, LLM_MODEL

OUT = Path(__file__).resolve().parent
UNRELATED = [{"question": "Что не так с ИТП-3 за последнюю неделю?",
              "answer": "На ИТП-3 обратка выше графика на 12,9 % с 21 по 27 февраля — разрегулировка системы "
                        "отопления потребителя; 26 февраля показание ГВС не менялось 18 часов."}]


def run_dialogs():
    truth = load_truth()
    rows = {}
    for mode in CONTEXT_MODES:
        rows[mode] = []
        for d in DIALOGS:
            history = []
            if mode != "none":                  # без контекста история не используется — не тратим время
                for turn in d["turns"]:
                    r = ask(turn, history=history, context=mode)
                    history.append({"question": r["question"], "answer": r["report"]["answer"]})
            row = evaluate(d, truth, history=history, context=mode)
            row["turns"] = d["turns"]
            rows[mode].append(row)
            print(f"{mode:8} {d['id']} {'OK ' if row['correct'] else 'ERR'} {row['seconds']:5.1f} с  "
                  f"понял как: {row['understood_as']}  {row['missed'] or ''} {row['extra'] or ''} {row['notes'] or ''}",
                  flush=True)
    return rows


def _same(a, b):
    norm = lambda s: " ".join(s.lower().replace("ё", "е").strip(" ?.!").split())  # noqa: E731
    return norm(a) == norm(b)


def preservation():
    client = ollama.Client(host=LLM_HOST)
    changed = []
    for s in SCENARIOS:
        out = contextualize(s["question"], UNRELATED, client, LLM_MODEL)
        if not _same(out, s["question"]):
            changed.append({"id": s["id"], "question": s["question"], "rewritten": out})
            print(f"  изменён {s['id']}: {s['question']!r} → {out!r}", flush=True)
    return {"total": len(SCENARIOS), "changed": changed}


def write_markdown(rows, pres):
    ru = {"none": "без контекста", "rewrite": "переписывание вопроса", "messages": "история в сообщениях"}
    lines = [
        "# Контекст диалога: уточняющие вопросы",
        "",
        f"{len(DIALOGS)} диалогов (`eval/dialogs.py`): D01–D10 написаны до реализации и использовались для "
        f"доработки переписывания, D11–D16 — отложенные. Модель `{LLM_MODEL}`, ReAct. "
        "Первый прогон (до доработки) — [DIALOG_round1.md](DIALOG_round1.md).",
        "",
        "| Способ | Верно | D01–D10 (на них дорабатывалось переписывание) | **D11–D16 (отложенные)** | "
        "Среднее время уточнения |",
        "|---|---|---|---|---|",
    ] + [f"| {ru[m]} | {sum(r['correct'] for r in rs)} / {len(rs)} | "
         f"{sum(r['correct'] for r in rs if r['id'] not in HELD_OUT)} / {sum(r['id'] not in HELD_OUT for r in rs)} | "
         f"**{sum(r['correct'] for r in rs if r['id'] in HELD_OUT)} / {sum(r['id'] in HELD_OUT for r in rs)}** | "
         f"{sum(r['seconds'] for r in rs) / len(rs):.0f} с |" for m, rs in rows.items()] + [
        "",
        f"Самостоятельные вопросы с чужой историей: переписывание изменило {len(pres['changed'])} из {pres['total']}.",
        "",
        "## По диалогам",
        "",
        "| ID | Диалог | " + " | ".join(ru[m] for m in rows) + " | Как понят (переписывание) |",
        "|---|---|" + "---|" * len(rows) + "---|",
    ]
    for i, d in enumerate(DIALOGS):
        dialog = " → ".join(d["turns"] + [f"**{d['question']}**"])
        marks = ["✅" if rows[m][i]["correct"] else "❌" for m in rows]
        lines.append(f"| {d['id']} | {dialog} | " + " | ".join(marks) + f" | {rows['rewrite'][i]['understood_as']} |")
    if pres["changed"]:
        lines += ["", "## Самостоятельные вопросы, которые переписывание изменило", "",
                  "| ID | Вопрос | Стал |", "|---|---|---|"]
        lines += [f"| {c['id']} | {c['question']} | {c['rewritten']} |" for c in pres["changed"]]
    (OUT / "DIALOG.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def report_only():
    data = json.loads(sorted((OUT / "results").glob("dialog_*.json"))[-1].read_text(encoding="utf-8"))
    write_markdown(data["rows"], data["preservation"])
    print((OUT / "DIALOG.md").read_text(encoding="utf-8"))


def main():
    rows = run_dialogs()
    print("Проверка самостоятельных вопросов:", flush=True)
    pres = preservation()
    (OUT / "results").mkdir(exist_ok=True)
    path = OUT / "results" / f"dialog_{time.strftime('%Y%m%d_%H%M')}.json"
    path.write_text(json.dumps({"model": LLM_MODEL, "rows": rows, "preservation": pres}, ensure_ascii=False,
                               indent=2, default=str), encoding="utf-8")
    write_markdown(rows, pres)
    print((OUT / "DIALOG.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    report_only() if "--report" in sys.argv else main()
