"""Оценка агента на сценариях. Долго: ~20–60 с на сценарий.

    python -m eval.eval_agent                 # все сценарии
    python -m eval.eval_agent S01 H03         # выборочно
    python -m eval.eval_agent --model=qwen2.5:3b --arch=workflow   # другая модель / архитектура

Результаты: eval/results/agent_<модель>_<время>.json и сводка eval/RESULTS.md.
"""
import json
import sys
import time
from pathlib import Path

import pandas as pd
import psycopg

from data.norms_corpus import BY_CODE
from eval.eval_detectors import run as run_detectors
from eval.eval_retrieval import gold_snippet
from eval.matching import TOLERANCE, _sid, load_truth, match_sets, truth_in
from eval.scenarios import SCENARIOS
from itp.agent import _numbers, ask
from itp.config import DB_DSN, LLM_MODEL

OUT = Path(__file__).resolve().parent
GROUPS = {"S": "Диагностика", "N": "Проблем нет", "A": "Нетипичные (на них дорабатывался агент)",
          "H": "Отложенный набор (агент на нём не настраивался)", "R": "Вопросы о нормах (поиск по документам)"}
VALUE_TOL = 0.15
CITE = {"PTE511": ("511", "технической эксплуатации"), "PP354": ("354", "коммунальных услуг"),
        "PP1034": ("1034", "коммерческого учета", "коммерческого учёта"), "SANPIN3684": ("3684", "санпин")}


def evaluate_norm(scn, row, trace):
    """Вопрос о норме: поиск нашёл эталонный пункт, в ответе его числа и ссылка на документ."""
    snippet = gold_snippet(scn)
    found = any(BY_CODE[scn["doc"]]["short"] in r["source"] and snippet in " ".join(r["text"].split())
                for s in trace if s["tool"] == "search_norms" for r in s["result"].get("results", []))
    answer = row["answer"]
    cites = answer + " " + " ".join(row["report"].get("sources", []))
    missing = [n for n in scn["numbers"] if not _numbers(n) & _numbers(answer)]
    wrong = [n for n in scn.get("forbidden", []) if _numbers(n) & _numbers(answer)]
    cited = any(k in cites.lower() for k in CITE[scn["doc"]])
    row["correct"] = not row["error"] and found and not missing and not wrong and cited
    row["notes"] += ([] if found else ["эталонный пункт не найден"]) + \
                    ([f"нет чисел {missing}"] if missing else []) + \
                    ([f"чужие числа {wrong}"] if wrong else []) + ([] if cited else ["нет ссылки на документ"])
    return row


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


def evaluate(scn, truth, model=LLM_MODEL, arch="react", **ask_kw):
    """ask_kw — history/context для диалоговых сценариев (eval/eval_dialog.py)."""
    kind = scn.get("type", "diagnosis")
    t0 = time.time()
    try:
        result = ask(scn["question"], model=model, arch=arch, **ask_kw)
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
           "trace": [{"tool": s["tool"], "args": s["args"]} for s in result["trace"]], "report": report,
           "understood_as": result.get("question", scn["question"])}

    if kind == "offtopic":
        row["correct"] = not error and not findings
        if findings:
            row["notes"].append("находки на вопрос вне темы")
        return row

    if kind == "norm":
        return evaluate_norm(scn, row, result["trace"])

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
        "norms_correct": sum(r["correct"] for r in rows if r["type"] == "norm"),
        "norms": sum(r["type"] == "norm" for r in rows),
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
        f"| Вопросы о нормах: нужный пункт найден, числа и ссылка в ответе | {summary['norms_correct']} из "
        f"{summary['norms']} |",
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


def main(ids, model=LLM_MODEL, arch="react"):
    truth = load_truth()
    scenarios = [s for s in SCENARIOS if not ids or s["id"] in ids]
    rows = []
    for scn in scenarios:
        r = evaluate(scn, truth, model, arch)
        rows.append(r)
        print(f"{r['id']} {'OK ' if r['correct'] else 'ERR'} {r['type']:9} exp={r['expected']} found={r['found']} "
              f"tp={r['tp']} {r['seconds']} с {r['missed'] or ''} {r['extra'] or ''} "
              f"{r['wrong_category'] or ''} {r['notes'] or ''} {r['error'] or ''}", flush=True)
    summary = summarize(rows)
    det = run_detectors()
    (OUT / "results").mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M")
    path = OUT / "results" / f"agent_{arch}_{model.replace(':', '-')}_{stamp}.json"
    path.write_text(json.dumps({"model": model, "arch": arch, "summary": summary, "rows": rows},
                               ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    if not ids and model == LLM_MODEL and arch == "react":
        write_markdown(summary, rows, det)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return path


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    opts = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--"))
    main([a for a in sys.argv[1:] if not a.startswith("--")],
         model=opts.get("model", LLM_MODEL), arch=opts.get("arch", "react"))
