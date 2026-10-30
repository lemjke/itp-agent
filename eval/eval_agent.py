"""Оценка агента на сценариях. Долго: ~20–60 с на сценарий.

    python -m eval.eval_agent                 # все сценарии
    python -m eval.eval_agent S01 H03         # выборочно

Результаты: eval/results/agent_<модель>_<время>.json и сводка eval/RESULTS.md.
"""
import json
import sys
import time
from pathlib import Path

import pandas as pd
import psycopg

from eval.eval_detectors import run as run_detectors
from eval.matching import TOLERANCE, _sid, load_truth, match_sets, truth_in
from eval.scenarios import SCENARIOS
from itp.agent import _numbers, ask
from itp.config import DB_DSN, LLM_MODEL

OUT = Path(__file__).resolve().parent
GROUPS = {"S": "Диагностика", "N": "Проблем нет", "A": "Нетипичные (на них дорабатывался агент)",
          "H": "Отложенный набор (агент на нём не настраивался)"}
VALUE_TOL = 0.15


def _overlaps(args, start, end):
    try:
        s, e = pd.Timestamp(args["start"]), pd.Timestamp(args["end"])
    except (KeyError, ValueError, TypeError):
        return False
    return s <= pd.Timestamp(end) + pd.Timedelta(hours=23) and e >= pd.Timestamp(start)


def true_value(scn):
    """Эталонное значение для вопросов о числах — прямо из БД, мимо инструментов агента."""
    with psycopg.connect(DB_DSN) as con:
        x = pd.read_sql(f"SELECT {scn['param']} AS v FROM measurements WHERE substation_id = %s "
                        "AND ts BETWEEN %s AND %s",
                        con, params=(scn["substations"][0], scn["start"], f"{scn['end']} 23:00"))["v"]
    return round(float(getattr(x, scn["stat"])()), 1)


def evaluate(scn, truth):
    kind = scn.get("type", "diagnosis")
    t0 = time.time()
    try:
        result = ask(scn["question"])
        error = None
    except Exception as exc:          # сбой модели/JSON — тоже результат
        result, error = {"report": {"answer": "", "findings": []}, "trace": [], "dropped_findings": 0}, str(exc)
    report = result["report"]
    findings = report.get("findings", [])
    row = {"id": scn["id"], "group": scn["id"][0], "type": kind, "question": scn["question"], "error": error,
           "answer": report.get("answer", ""), "found": len(findings),
           "dropped": result.get("dropped_findings", 0), "seconds": round(time.time() - t0, 1),
           "expected": 0, "tp": 0, "category_ok": 0, "requirement_ok": 0, "coverage": None,
           "missed": [], "extra": [], "wrong_category": [], "notes": [],
           "trace": [{"tool": s["tool"], "args": s["args"]} for s in result["trace"]], "report": report}

    if kind == "offtopic":
        row["correct"] = not error and not findings
        if findings:
            row["notes"].append("находки на вопрос вне темы")
        return row

    if kind == "value":
        expected_value = true_value(scn)
        got = [float(n) for n in _numbers(row["answer"])]
        ok = any(abs(v - expected_value) <= VALUE_TOL for v in got)
        row["correct"] = not error and ok
        row["notes"].append(f"эталон {expected_value}" + ("" if ok else f", в ответе: {row['answer'][:80]}"))
        return row

    # diagnosis
    expected = [t for sid in scn["substations"] for t in truth_in(truth, sid, scn["start"], scn["end"])]
    if scn.get("kinds"):
        expected = [t for t in expected if t["kind"] in scn["kinds"]]
    checked = set()
    for step in result["trace"]:
        if step["tool"] == "detect_anomalies" and _overlaps(step["args"], scn["start"], scn["end"]):
            sub = step["args"].get("substation", "")
            checked |= set(scn["substations"]) if str(sub).lower() in ("все", "all", "*") else {_sid(sub)}
    # ИТП, названный адресом, учитываем по найденным событиям
    checked |= {_sid(f["substation"]) for f in findings}
    pairs, missed, extra = match_sets(findings, expected)
    if scn.get("allow_extra"):
        extra = [f for f in extra if _sid(f.get("substation")) not in scn["substations"]]
    cat_ok = sum(f.get("category") == t["category"] for f, t in pairs)
    req_ok = sum(str(f.get("requirement_code", "")).upper() == t["requirement_code"] for f, t in pairs)
    note_ok = not scn.get("note") or "только с" in row["answer"]
    row.update({
        "expected": len(expected), "found": len(pairs) + len(extra), "tp": len(pairs),
        "category_ok": cat_ok, "requirement_ok": req_ok,
        "coverage": len(checked & set(scn["substations"])) / len(scn["substations"]),
        "missed": [f"ИТП-{t['substation']} {t['kind']} {t['start']:%d.%m %H:%M}" for t in missed],
        "extra": [f"{f.get('substation')} {f.get('parameter')} {f.get('start')}" for f in extra],
        "wrong_category": [f"{t['kind']}: {f.get('category')}" for f, t in pairs if f.get("category") != t["category"]],
        "correct": (not error and not missed and not extra and cat_ok == len(pairs)
                    and req_ok == len(pairs) and note_ok),
    })
    if not note_ok:
        row["notes"].append("не сообщил о сокращении периода")
    return row


def summarize(rows):
    diag = [r for r in rows if r["type"] == "diagnosis"]
    exp = sum(r["expected"] for r in diag)
    fnd = sum(r["found"] for r in diag)
    tp = sum(r["tp"] for r in diag)
    p = tp / fnd if fnd else 1.0
    rc = tp / exp if exp else 1.0
    by_group = {g: {"correct": sum(r["correct"] for r in rows if r["group"] == g),
                    "total": sum(r["group"] == g for r in rows)} for g in GROUPS}
    return {
        "scenarios": len(rows),
        "scenarios_correct": sum(r["correct"] for r in rows),
        "by_group": by_group,
        "precision": p, "recall": rc, "f1": 2 * p * rc / (p + rc) if p + rc else 0.0,
        "category_accuracy": sum(r["category_ok"] for r in diag) / tp if tp else 0.0,
        "requirement_accuracy": sum(r["requirement_ok"] for r in diag) / tp if tp else 0.0,
        "dropped_by_guard": sum(r["dropped"] for r in rows),
        "tool_coverage": sum(r["coverage"] for r in diag) / len(diag) if diag else 0.0,
        "values_correct": sum(r["correct"] for r in rows if r["type"] == "value"),
        "values": sum(r["type"] == "value" for r in rows),
        "offtopic_correct": sum(r["correct"] for r in rows if r["type"] == "offtopic"),
        "offtopic": sum(r["type"] == "offtopic" for r in rows),
        "mean_seconds": sum(r["seconds"] for r in rows) / len(rows),
        "errors": sum(bool(r["error"]) for r in rows),
    }


def write_markdown(summary, rows, det):
    pct = lambda x: f"{x * 100:.0f} %"  # noqa: E731
    lines = [
        f"# Результаты оценки · модель `{LLM_MODEL}` · {time.strftime('%Y-%m-%d %H:%M')}",
        "",
        f"Сопоставление событий по ИТП, параметру и времени с допуском ±{int(TOLERANCE.total_seconds() // 3600)} ч.",
        "",
        "## Детекторы (без LLM)",
        "",
        "| Эталон | Найдено | Совпало | Precision | Recall | F1 |",
        "|---|---|---|---|---|---|",
        f"| {det['truth']} | {det['found']} | {det['tp']} | {det['precision']:.2f} | {det['recall']:.2f} | {det['f1']:.2f} |",
        "",
        "## Агент",
        "",
        "| Группа сценариев | Верно |",
        "|---|---|",
    ] + [f"| {GROUPS[g]} | {v['correct']} из {v['total']} |" for g, v in summary["by_group"].items() if v["total"]] + [
        f"| **Всего** | **{summary['scenarios_correct']} из {summary['scenarios']}** |",
        "",
        "| Метрика | Значение |",
        "|---|---|",
        f"| Precision / Recall / F1 по событиям (диагностика) | {summary['precision']:.2f} / {summary['recall']:.2f} / {summary['f1']:.2f} |",
        f"| Верная категория (датчик / режим) | {pct(summary['category_accuracy'])} |",
        f"| Верная ссылка на норму | {pct(summary['requirement_accuracy'])} |",
        f"| Верные числа в ответах на вопросы о значениях | {summary['values_correct']} из {summary['values']} |",
        f"| Корректная реакция на вопросы вне темы | {summary['offtopic_correct']} из {summary['offtopic']} |",
        f"| Находок отброшено проверкой (без опоры на детекторы) | {summary['dropped_by_guard']} |",
        f"| Проверены нужные ИТП за нужный период | {pct(summary['tool_coverage'])} |",
        f"| Среднее время ответа | {summary['mean_seconds']:.0f} с |",
        f"| Сбоев | {summary['errors']} |",
        "",
        "## По сценариям",
        "",
        "| ID | Тип | Вопрос | Ожид. | Найд. | Совп. | Категория | Норма | Итог | Время | Замечания |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        notes = "; ".join(
            [f"пропущено: {m}" for m in r["missed"]] + [f"лишнее: {x}" for x in r["extra"]]
            + [f"категория: {w}" for w in r["wrong_category"]] + r["notes"]
            + ([f"отброшено проверкой: {r['dropped']}"] if r["dropped"] else [])
            + ([f"ошибка: {r['error']}"] if r["error"] else []))
        diag = r["type"] == "diagnosis"
        lines.append(
            f"| {r['id']} | {r['type']} | {r['question']} | "
            + (f"{r['expected']} | {r['found']} | {r['tp']} | {r['category_ok']}/{r['tp']} | "
               f"{r['requirement_ok']}/{r['tp']} | " if diag else "— | — | — | — | — | ")
            + f"{'✅' if r['correct'] else '❌'} | {r['seconds']:.0f} с | {notes or '—'} |")
    (OUT / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(ids):
    truth = load_truth()
    scenarios = [s for s in SCENARIOS if not ids or s["id"] in ids]
    rows = []
    for scn in scenarios:
        r = evaluate(scn, truth)
        rows.append(r)
        print(f"{r['id']} {'OK ' if r['correct'] else 'ERR'} {r['type']:9} exp={r['expected']} found={r['found']} "
              f"tp={r['tp']} {r['seconds']} с {r['missed'] or ''} {r['extra'] or ''} "
              f"{r['wrong_category'] or ''} {r['notes'] or ''} {r['error'] or ''}", flush=True)
    summary = summarize(rows)
    det = run_detectors()
    (OUT / "results").mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M")
    path = OUT / "results" / f"agent_{LLM_MODEL.replace(':', '-')}_{stamp}.json"
    path.write_text(json.dumps({"model": LLM_MODEL, "summary": summary, "rows": rows},
                               ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    if not ids:
        write_markdown(summary, rows, det)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main(sys.argv[1:])
