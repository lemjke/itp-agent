"""Оценка агента на сценариях. Долго: ~30–60 с на сценарий.

    python -m eval.eval_agent                 # все сценарии
    python -m eval.eval_agent S01 N02         # выборочно

Результаты: eval/results/agent_<модель>_<время>.json и сводка eval/RESULTS.md.
"""
import json
import sys
import time
from pathlib import Path

import pandas as pd

from eval.eval_detectors import run as run_detectors
from eval.matching import TOLERANCE, _sid, load_truth, match_sets, matches, truth_in
from eval.scenarios import SCENARIOS
from itp.agent import ask
from itp.config import LLM_MODEL

OUT = Path(__file__).resolve().parent


def _overlaps(args, start, end):
    try:
        s, e = pd.Timestamp(args["start"]), pd.Timestamp(args["end"])
    except (KeyError, ValueError, TypeError):
        return False
    return s <= pd.Timestamp(end) and e >= pd.Timestamp(start)


def evaluate(scn, truth):
    expected = [t for sid in scn["substations"] for t in truth_in(truth, sid, scn["start"], scn["end"])]
    t0 = time.time()
    try:
        result = ask(scn["question"])
        error = None
    except Exception as exc:          # сбой модели/JSON — тоже результат
        result, error = {"report": {"findings": []}, "trace": []}, str(exc)
    findings = result["report"].get("findings", [])

    # Какие события реально вернули детекторы — для проверки «заземлённости» ответа
    tool_events = []
    checked = set()
    for step in result["trace"]:
        if step["tool"] == "detect_anomalies" and "events" in step["result"]:
            res = step["result"]
            names = res.get("substations_checked") or [res["substation"]]
            if _overlaps(step["args"], scn["start"], scn["end"]):
                checked |= {_sid(n) for n in names}
            tool_events += [{**ev, "substation": _sid(ev["substation"])} for ev in res["events"]]
    grounded = [f for f in findings
                if any(matches(f, {"substation": ev["substation"], "parameter": ev["parameter"],
                                   "start": pd.Timestamp(ev["start"]), "end": pd.Timestamp(ev["end"])})
                       for ev in tool_events)]

    pairs, missed, extra = match_sets(findings, expected)
    cat_ok = sum(f.get("category") == t["category"] for f, t in pairs)
    req_ok = sum(str(f.get("requirement_code", "")).upper() == t["requirement_code"] for f, t in pairs)
    coverage = len(checked & set(scn["substations"])) / len(scn["substations"])
    correct = (not error and not missed and not extra and cat_ok == len(pairs) and req_ok == len(pairs))
    return {
        "id": scn["id"], "question": scn["question"], "error": error,
        "expected": len(expected), "found": len(findings), "tp": len(pairs),
        "category_ok": cat_ok, "requirement_ok": req_ok,
        "ungrounded": len(findings) - len(grounded), "coverage": coverage,
        "correct": correct, "seconds": round(time.time() - t0, 1),
        "missed": [f"ИТП-{t['substation']} {t['kind']} {t['start']:%d.%m %H:%M}" for t in missed],
        "extra": [f"{f.get('substation')} {f.get('parameter')} {f.get('start')}" for f in extra],
        "wrong_category": [f"{t['kind']}: {f.get('category')}" for f, t in pairs if f.get("category") != t["category"]],
        "trace": [{"tool": s["tool"], "args": s["args"]} for s in result["trace"]],
        "report": result["report"],
    }


def summarize(rows):
    exp = sum(r["expected"] for r in rows)
    fnd = sum(r["found"] for r in rows)
    tp = sum(r["tp"] for r in rows)
    p = tp / fnd if fnd else 1.0
    rc = tp / exp if exp else 1.0
    return {
        "scenarios": len(rows),
        "scenarios_correct": sum(r["correct"] for r in rows),
        "precision": p, "recall": rc, "f1": 2 * p * rc / (p + rc) if p + rc else 0.0,
        "category_accuracy": sum(r["category_ok"] for r in rows) / tp if tp else 0.0,
        "requirement_accuracy": sum(r["requirement_ok"] for r in rows) / tp if tp else 0.0,
        "ungrounded_findings": sum(r["ungrounded"] for r in rows),
        "tool_coverage": sum(r["coverage"] for r in rows) / len(rows),
        "negatives_clean": sum(r["found"] == 0 for r in rows if r["id"].startswith("N")),
        "negatives": sum(r["id"].startswith("N") for r in rows),
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
        "| Метрика | Значение |",
        "|---|---|",
        f"| Сценариев полностью верно | {summary['scenarios_correct']} из {summary['scenarios']} |",
        f"| Precision / Recall / F1 по событиям | {summary['precision']:.2f} / {summary['recall']:.2f} / {summary['f1']:.2f} |",
        f"| Верная категория (датчик / режим) | {pct(summary['category_accuracy'])} |",
        f"| Верная ссылка на норму | {pct(summary['requirement_accuracy'])} |",
        f"| Находок без опоры на инструменты | {summary['ungrounded_findings']} |",
        f"| Проверены нужные ИТП за нужный период | {pct(summary['tool_coverage'])} |",
        f"| Отрицательные сценарии без ложных находок | {summary['negatives_clean']} из {summary['negatives']} |",
        f"| Среднее время ответа | {summary['mean_seconds']:.0f} с |",
        f"| Сбоев | {summary['errors']} |",
        "",
        "## По сценариям",
        "",
        "| ID | Вопрос | Ожид. | Найд. | Совп. | Категория | Норма | Итог | Время | Замечания |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        notes = "; ".join(
            [f"пропущено: {m}" for m in r["missed"]] + [f"лишнее: {x}" for x in r["extra"]]
            + [f"категория: {w}" for w in r["wrong_category"]] + ([f"ошибка: {r['error']}"] if r["error"] else []))
        lines.append(
            f"| {r['id']} | {r['question']} | {r['expected']} | {r['found']} | {r['tp']} | "
            f"{r['category_ok']}/{r['tp']} | {r['requirement_ok']}/{r['tp']} | "
            f"{'✅' if r['correct'] else '❌'} | {r['seconds']:.0f} с | {notes or '—'} |")
    (OUT / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(ids):
    truth = load_truth()
    scenarios = [s for s in SCENARIOS if not ids or s["id"] in ids]
    rows = []
    for scn in scenarios:
        r = evaluate(scn, truth)
        rows.append(r)
        print(f"{r['id']} {'OK ' if r['correct'] else 'ERR'} exp={r['expected']} found={r['found']} "
              f"tp={r['tp']} cat={r['category_ok']} req={r['requirement_ok']} {r['seconds']} с "
              f"{r['missed'] or ''} {r['extra'] or ''} {r['wrong_category'] or ''} {r['error'] or ''}", flush=True)
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
