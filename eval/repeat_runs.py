"""Повторные прогоны оценки агента: среднее, разброс и нестабильные сценарии.

    python -m eval.repeat_runs              # все конфигурации (~7 ч), в фоне
    python -m eval.repeat_runs --report     # только отчёт из уже сделанных прогонов

Даже при temperature 0 Ollama не полностью детерминирована (разбиение запроса на пакеты, кэш
предыдущего запроса), и 7B-модель иногда выбирает другой инструмент. Один прогон не отличает
разницу в 1–3 сценария от шума — поэтому N прогонов на конфигурацию и среднее со стандартным отклонением.
Каждый прогон помечается партией (BATCH) и хешем коммита — отчёт строится только из прогонов одной
версии кода. Порядок — по важности: если медленные конфигурации в конце не успеют, отчёт будет без них.
Результат: eval/REPEATS.md, docs/repeats.png.
"""
import json
import statistics
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from eval import eval_agent  # noqa: E402
from eval.scenarios import SCENARIOS  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "eval" / "results"
BATCH = "repeats-2026-11-02"
CONFIGS = [   # (модель, подпись, прогонов)
    ("qwen2.5:7b", "7B Q4_K_M", 5),
    ("qwen2.5:3b", "3B Q4_K_M", 3),
    ("qwen2.5:3b-instruct-q8_0", "3B Q8_0", 3),
    ("qwen2.5:7b-instruct-q8_0", "7B Q8_0", 3),
    ("qwen2.5:3b-instruct-fp16", "3B FP16", 3),
]
COLOR = {"7B": "#2a78d6", "3B": "#eb6834"}


def commit():
    return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                          text=True).stdout.strip()


def runs_of(model):
    out = []
    for f in sorted(RESULTS.glob(f"agent_react_{model.replace(':', '-')}_*.json")):
        data = json.loads(f.read_text(encoding="utf-8"))
        if data.get("batch") == BATCH:
            out.append(data)
    return out


def run():
    sha = commit()
    for model, label, n in CONFIGS:
        for i in range(len(runs_of(model)), n):          # продолжение после остановки — без повторов
            print(f"=== {label}, прогон {i + 1}/{n}", flush=True)
            path = eval_agent.main([], model=model, arch="react")
            data = json.loads(path.read_text(encoding="utf-8"))
            data.update(batch=BATCH, commit=sha, repeat=i + 1)
            path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def stats(values):
    return statistics.mean(values), (statistics.stdev(values) if len(values) > 1 else 0.0)


def collect():
    rows = []
    for model, label, _ in CONFIGS:
        runs = runs_of(model)
        if not runs:
            continue
        correct = [r["summary"]["scenarios_correct"] for r in runs]
        groups = {}
        for g in "SNAHRC":
            vals = [r["summary"]["by_group"][g]["correct"] for r in runs if g in r["summary"]["by_group"]]
            if vals:
                groups[g] = (*stats(vals), runs[0]["summary"]["by_group"][g]["total"])
        pass_rate = {}
        for r in runs:
            for row in r["rows"]:
                pass_rate.setdefault(row["id"], []).append(row["correct"])
        rows.append({"model": model, "label": label, "n": len(runs), "total": runs[0]["summary"]["scenarios"],
                     "correct": correct, "mean": stats(correct)[0], "sd": stats(correct)[1],
                     "groups": groups, "seconds": statistics.mean(r["summary"]["mean_seconds"] for r in runs),
                     "pass_rate": {k: sum(v) / len(v) for k, v in pass_rate.items()},
                     "commits": sorted({r.get("commit", "?") for r in runs})})
    return rows


def num(x, d=1):
    return f"{x:.{d}f}".replace(".", ",")


def write_markdown(rows):
    group_ru = {"S": "Диагностика", "N": "Проблем нет", "A": "Нетипичные", "H": "Отложенные",
                "R": "Нормы", "C": "Сравнение периодов"}
    lines = [
        "# Повторные прогоны оценки",
        "",
        f"Партия `{BATCH}`, коммит(ы): {', '.join(sorted({c for r in rows for c in r['commits']}))}. "
        f"{rows[0]['total'] if rows else '—'} сценариев, ReAct, temperature 0. Среднее ± стандартное отклонение "
        "по прогонам.",
        "",
        "| Конфигурация | Прогонов | Верно (среднее ± σ) | Минимум–максимум | Время ответа |",
        "|---|---|---|---|---|",
    ] + [f"| {r['label']} | {r['n']} | **{num(r['mean'])} ± {num(r['sd'])}** из {r['total']} | "
         f"{min(r['correct'])}–{max(r['correct'])} | {num(r['seconds'], 0)} с |" for r in rows] + [
        "",
        "## По группам (среднее ± σ)",
        "",
        "| Группа | " + " | ".join(r["label"] for r in rows) + " |",
        "|---|" + "---|" * len(rows),
    ]
    for g, title in group_ru.items():
        cells = [f"{num(r['groups'][g][0])} ± {num(r['groups'][g][1])} / {r['groups'][g][2]}" if g in r["groups"]
                 else "—" for r in rows]
        lines.append(f"| {title} | " + " | ".join(cells) + " |")
    if rows:
        main = rows[0]
        q = {s["id"]: s["question"] for s in SCENARIOS}
        unstable = sorted((k for k, v in main["pass_rate"].items() if 0 < v < 1), key=lambda k: main["pass_rate"][k])
        never = sorted(k for k, v in main["pass_rate"].items() if v == 0)
        lines += ["", f"## {main['label']}: нестабильные сценарии (проходят не во всех {main['n']} прогонах)", "",
                  "| ID | Вопрос | Доля прогонов, где верно |", "|---|---|---|"]
        lines += [f"| {k} | {q.get(k, '')} | {num(main['pass_rate'][k] * 100, 0)} % |" for k in unstable]
        lines += ["", f"## {main['label']}: не проходят ни в одном прогоне", "",
                  "| ID | Вопрос |", "|---|---|"] + [f"| {k} | {q.get(k, '')} |" for k in never]
    lines += ["", "![Повторные прогоны](../docs/repeats.png)", ""]
    (ROOT / "eval" / "REPEATS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot(rows):
    ink, muted, grid, surface = "#1f1f1d", "#898781", "#e1e0d9", "#fcfcfb"
    fig, ax = plt.subplots(figsize=(8.5, 0.62 * len(rows) + 1.3), facecolor=surface)
    ax.set_facecolor(surface)
    for i, r in enumerate(rows[::-1]):
        color = COLOR[r["label"][:2]]
        pct = [c / r["total"] * 100 for c in r["correct"]]
        ax.scatter(pct, [i] * len(pct), s=34, color=color, alpha=0.55, zorder=3, edgecolor="none")
        m, sd = r["mean"] / r["total"] * 100, r["sd"] / r["total"] * 100
        ax.errorbar(m, i, xerr=sd, fmt="|", ms=16, mew=2.4, color=ink, capsize=4, lw=1.4, zorder=4)
        ax.text(102, i, f"{num(m, 0)} ± {num(sd, 0)} %", va="center", fontsize=9, color=ink)
    ax.set_yticks(range(len(rows)), [r["label"] for r in rows[::-1]], fontsize=9.5, color=ink)
    ax.set_xlim(0, 100)
    ax.set_title("Доля верных сценариев: точки — прогоны, черта — среднее ± σ", loc="left", fontsize=10, color=ink)
    ax.grid(axis="x", color=grid, lw=0.7)
    ax.tick_params(colors=muted, labelsize=8.5, length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(grid)
    fig.tight_layout(rect=(0, 0, 0.9, 1))
    fig.savefig(ROOT / "docs" / "repeats.png", dpi=150, facecolor=surface)


def main(report_only):
    if not report_only:
        run()
    rows = collect()
    write_markdown(rows)
    plot(rows)
    print((ROOT / "eval" / "REPEATS.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main("--report" in sys.argv)
