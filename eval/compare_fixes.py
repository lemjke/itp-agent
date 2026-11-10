"""Эффект правок по итогам повторных прогонов: до и после, против известного разброса.

    python -m eval.compare_fixes          # отчёт eval/FIXES.md из уже сделанных прогонов

До: 5 прогонов 7B Q4 из партии repeats-2026-11-02 (43 сценария) и 3 прогона новых сценариев V01–V08
на том же коде (eval/results/validation/before_*.json). Правки 1: 3 полных прогона партии fixes-2026-11-11
(в них нашлись регрессии S13 и R02), правки 2 — fixes2-2026-11-11 (R06 стал отказом «вне темы»),
правки 3 — fixes3-2026-11-11, итог.
Сценарии V написаны до правок и на старом коде прогнаны раньше, чем код менялся: на них агент
не настраивался, поэтому они показывают, не подогнаны ли правки под старые вопросы.
"""
import json
import statistics
import sys
from pathlib import Path

from eval.scenarios import SCENARIOS

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "eval" / "results"
MODEL = "qwen2.5-7b"
BEFORE = "repeats-2026-11-02"
ROUNDS = ["fixes-2026-11-11", "fixes2-2026-11-11", "fixes3-2026-11-11"]   # раунды правок по порядку
QUESTION = {s["id"]: s["question"] for s in SCENARIOS}


def batch(name):
    out = []
    for f in sorted(RESULTS.glob(f"agent_react_{MODEL}_*.json")):
        data = json.loads(f.read_text(encoding="utf-8"))
        if data.get("batch") == name:
            out.append(data)
    return out


def commits(runs):
    return ", ".join(sorted({r.get("commit", "?") for r in runs}))


def passes(runs, ids=None):
    """Сколько сценариев верно в каждом прогоне и доля прогонов, где верен каждый сценарий."""
    per_run, rate = [], {}
    for r in runs:
        rows = [row for row in r["rows"] if ids is None or row["id"] in ids]
        per_run.append(sum(row["correct"] for row in rows))
        for row in rows:
            rate.setdefault(row["id"], []).append(row["correct"])
    return per_run, {k: sum(v) / len(v) for k, v in rate.items()}


def ms(values):
    sd = statistics.stdev(values) if len(values) > 1 else 0.0
    return f"{statistics.mean(values):.1f} ± {sd:.1f}".replace(".", ",")


def pct(x):
    return f"{x:.0%}".replace("%", " %")


def main():
    before = batch(BEFORE)
    rounds = [(f"Правки {n}", name, batch(name)) for n, name in enumerate(ROUNDS, 1)]
    v_before = [json.loads(f.read_text(encoding="utf-8")) for f in sorted((RESULTS / "validation").glob("before_*.json"))]
    if not before or not all(runs for *_, runs in rounds):
        sys.exit("Нет прогонов до или после правок")
    old_ids = {row["id"] for row in before[0]["rows"]}
    v_ids = {s["id"] for s in SCENARIOS if s["id"].startswith("V")}
    # для каждого столбца: прогоны старых сценариев и прогоны V (до правок V — отдельные файлы)
    cols = [("До", before, v_before)] + [(title, runs, runs) for title, _, runs in rounds]
    head = "| " + " | ".join(c[0] for c in cols) + " |"
    sep = "|---" * (len(cols) + 1) + "|"
    groups = {"S": "Диагностика", "N": "Проблем нет", "A": "Нетипичные", "H": "Отложенные", "R": "Нормы",
              "C": "Сравнение периодов"}
    desc = "; ".join(f"{t.lower()} — {len(r)} (`{n}`, коммит {commits(r)})" for t, n, r in rounds)

    lines = [
        "# Правки по итогам повторных прогонов",
        "",
        f"7B Q4_K_M, ReAct, temperature 0. До — {len(before)} прогонов (партия `{BEFORE}`); {desc}. "
        "Среднее ± стандартное отклонение по прогонам.",
        "",
        "| Набор " + head, sep,
        f"| Старые сценарии (на них искали слабые места), из {len(old_ids)} | "
        + " | ".join(ms(passes(o, old_ids)[0]) for _, o, _ in cols) + " |",
        f"| **Новые V01–V08 (написаны до правок)**, из {len(v_ids)} | "
        + " | ".join(f"**{ms(passes(v, v_ids)[0])}**" for _, _, v in cols) + " |",
        "",
        "## По группам старых сценариев",
        "",
        "| Группа " + head, sep,
    ]
    for g, name in groups.items():
        ids = {i for i in old_ids if i.startswith(g)}
        lines.append(f"| {name}, из {len(ids)} | " + " | ".join(ms(passes(o, ids)[0]) for _, o, _ in cols) + " |")
    rate = {}
    for i in sorted(old_ids | v_ids):
        rate[i] = [passes(v if i in v_ids else o, {i})[1].get(i) for _, o, v in cols]
    lines += ["", "## Сценарии, у которых менялась доля верных прогонов", "", "| ID | Вопрос " + head, "|---" + sep]
    for i, r in rate.items():
        if None not in r and len(set(r)) > 1:
            lines.append(f"| {i} | {QUESTION.get(i, '')} | " + " | ".join(pct(x) for x in r) + " |")
    lines += ["", "## После последних правок верны не во всех прогонах", "", "| ID | Вопрос | Доля верных |",
              "|---|---|---|"]
    lines += [f"| {i} | {QUESTION.get(i, '')} | {pct(r[-1])} |" for i, r in rate.items() if r[-1] is not None and r[-1] < 1]
    out = ROOT / "eval" / "FIXES.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out.read_text(encoding="utf-8"))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
