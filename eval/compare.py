"""Сравнение архитектур агента и моделей на одном наборе из 30 сценариев.

    python -m eval.compare              # прогнать все конфигурации (~1,5–2 ч) и собрать отчёт
    python -m eval.compare --report     # только собрать отчёт из уже сохранённых прогонов

Каждый прогон — eval.eval_agent с нужными --model/--arch; результаты в eval/results/agent_<arch>_<model>_*.json.
Отчёт: eval/COMPARISON.md и график docs/comparison.png.
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from eval import eval_agent  # noqa: E402
from itp.agent import ARCHITECTURES  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "eval" / "results"

CONFIGS = [
    ("react", "qwen2.5:7b"), ("plan", "qwen2.5:7b"), ("workflow", "qwen2.5:7b"),
    ("react", "qwen2.5:3b"), ("workflow", "qwen2.5:3b"),
    ("react", "qwen2.5:1.5b"), ("workflow", "qwen2.5:1.5b"),
    ("react", "qwen3:8b"),
]
ARCH_RU = {"react": "ReAct", "plan": "План → выполнение", "workflow": "Фикс. workflow"}
ARCH_COLOR = {"react": "#2a78d6", "plan": "#eb6834", "workflow": "#1baf7a"}   # слоты 1–3 эталонной палитры


def latest(arch, model):
    """Последний полный прогон (выборочные — eval_agent S01 H03 — пропускаются)."""
    for f in sorted(RESULTS.glob(f"agent_{arch}_{model.replace(':', '-')}_*.json"), reverse=True):
        data = json.loads(f.read_text(encoding="utf-8"))
        if data["summary"]["scenarios"] >= 30:
            return data
    return None


def collect():
    rows = []
    for arch, model in CONFIGS:
        data = latest(arch, model)
        if not data:
            continue
        s = data["summary"]
        g = s["by_group"]
        rows.append({
            "arch": arch, "model": model,
            "correct": s["scenarios_correct"], "total": s["scenarios"],
            "sn": (g["S"]["correct"] + g["N"]["correct"], g["S"]["total"] + g["N"]["total"]),
            "a": (g["A"]["correct"], g["A"]["total"]), "h": (g["H"]["correct"], g["H"]["total"]),
            "precision": s["precision"], "recall": s["recall"],
            "category": s["category_accuracy"], "values": (s["values_correct"], s["values"]),
            "offtopic": (s["offtopic_correct"], s["offtopic"]),
            "dropped": s["dropped_by_guard"], "seconds": s["mean_seconds"], "errors": s["errors"],
        })
    return rows


def frac(pair):
    return f"{pair[0]}/{pair[1]}"


def num(x, digits=2):
    return f"{x:.{digits}f}".replace(".", ",")


def write_markdown(rows):
    lines = [
        "# Сравнение архитектур и моделей",
        "",
        "Один и тот же набор из 30 сценариев (`eval/scenarios.py`), одни и те же инструменты, финальная стадия "
        "отчёта и проверки (grounding, отказ без вызова инструментов). Меняется только то, как агент "
        "решает, какие инструменты вызвать, и какая модель работает. temperature 0, один прогон на конфигурацию, "
        "ноутбук с RTX 4050 (6 ГБ).",
        "",
    ] + [f"- **{ARCH_RU[a]}** — {d.split(': ', 1)[1]}" for a, d in ARCHITECTURES.items()] + [
        "",
        "| Архитектура | Модель | Всего верно | S+N | A | H (отложенный) | Precision / Recall | "
        "Категория | Числа | Вне темы | Отброшено проверкой | Время, с |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {ARCH_RU[r['arch']]} | `{r['model']}` | **{r['correct']}/{r['total']}** | {frac(r['sn'])} | "
            f"{frac(r['a'])} | {frac(r['h'])} | {num(r['precision'])} / {num(r['recall'])} | "
            f"{num(r['category'] * 100, 0)} % | {frac(r['values'])} | {frac(r['offtopic'])} | {r['dropped']} | "
            f"{num(r['seconds'], 0)} |")
    lines += ["", "![Сравнение архитектур и моделей](../docs/comparison.png)", ""]
    (ROOT / "eval" / "COMPARISON.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot(rows):
    ink, muted, grid, surface = "#1f1f1d", "#898781", "#e1e0d9", "#fcfcfb"
    labels = [f"{r['model']} · {ARCH_RU[r['arch']]}" for r in rows]
    colors = [ARCH_COLOR[r["arch"]] for r in rows]
    y = range(len(rows))[::-1]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 0.48 * len(rows) + 1.6), facecolor=surface,
                                   gridspec_kw={"width_ratios": [1.4, 1]}, sharey=True)
    acc = [r["correct"] / r["total"] * 100 for r in rows]
    sec = [r["seconds"] for r in rows]
    for ax, vals, title, fmt in ((ax1, acc, "Верных сценариев, % (из 30)", "{:.0f} %"),
                                 (ax2, sec, "Среднее время ответа, с", "{:.0f} с")):
        ax.set_facecolor(surface)
        ax.barh(list(y), vals, color=colors, height=0.62, zorder=2)
        for yi, v in zip(y, vals):
            ax.text(v + max(vals) * 0.015, yi, fmt.format(v), va="center", fontsize=9, color=ink)
        ax.set_title(title, loc="left", fontsize=10.5, color=ink)
        ax.grid(axis="x", color=grid, lw=0.8, zorder=0)
        ax.tick_params(colors=muted, labelsize=9, length=0)
        for side in ("top", "right", "bottom"):
            ax.spines[side].set_visible(False)
        ax.spines["left"].set_color(grid)
        ax.set_xlim(0, max(vals) * 1.15)
    ax1.set_xlim(0, 112)
    ax1.set_yticks(list(y), labels, fontsize=9.5, color=ink)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in ARCH_COLOR.values()]
    fig.legend(handles, ARCH_RU.values(), loc="lower left", ncol=3, frameon=False, fontsize=9,
               labelcolor=ink, bbox_to_anchor=(0.01, 0.0))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(ROOT / "docs" / "comparison.png", dpi=150, facecolor=surface)


def main(report_only):
    if not report_only:
        for arch, model in CONFIGS:
            print(f"=== {arch} · {model}", flush=True)
            eval_agent.main([], model=model, arch=arch)
    rows = collect()
    write_markdown(rows)
    plot(rows)
    print((ROOT / "eval" / "COMPARISON.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main("--report" in sys.argv)
