"""Сравнение квантизаций: качество агента, скорость и память при разной точности весов.

    python -m eval.compare_quant              # прогон всех вариантов (~2–2,5 ч) и отчёт
    python -m eval.compare_quant --report     # только отчёт из сохранённых прогонов

Для каждого варианта: полная оценка агента (40 сценариев, eval.eval_agent, ReAct), скорость генерации
на одинаковом запросе (токенов/с) и размещение модели (сколько в видеопамяти, сколько в ОЗУ).
7B в FP16 (15,2 ГБ) на ноутбуке с 6 ГБ видеопамяти и 16 ГБ ОЗУ не помещается — в сравнении её нет.
Результат: eval/QUANTIZATION.md, docs/quantization.png.
"""
import json
import sys
from pathlib import Path

import matplotlib
import ollama

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from eval import eval_agent  # noqa: E402
from itp.config import LLM_HOST  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "eval" / "results"
CONFIGS = [
    ("qwen2.5:7b", "7B", "Q4_K_M"),
    ("qwen2.5:7b-instruct-q8_0", "7B", "Q8_0"),
    ("qwen2.5:3b", "3B", "Q4_K_M"),
    ("qwen2.5:3b-instruct-q8_0", "3B", "Q8_0"),
    ("qwen2.5:3b-instruct-fp16", "3B", "FP16"),
]
QUANT_COLOR = {"Q4_K_M": "#2a78d6", "Q8_0": "#eb6834", "FP16": "#1baf7a"}
SPEED_PROMPT = ("Перечисли пять типичных неисправностей датчиков температуры в индивидуальном тепловом пункте "
                "и коротко объясни каждую.")


def speed(model):
    """Скорость на одинаковом запросе и размещение модели (после прогрева)."""
    client = ollama.Client(host=LLM_HOST)
    client.generate(model=model, prompt="Привет", options={"num_predict": 5, "num_ctx": 8192})   # загрузка
    r = client.generate(model=model, prompt=SPEED_PROMPT,
                        options={"temperature": 0, "num_predict": 256, "num_ctx": 8192})
    loaded = next(m for m in client.ps().models if m.model == model or m.name == model)
    return {"tokens_per_s": r.eval_count / (r.eval_duration / 1e9),
            "prompt_tokens_per_s": r.prompt_eval_count / (r.prompt_eval_duration / 1e9),
            "size_gb": loaded.size / 1e9, "vram_gb": loaded.size_vram / 1e9}


def run():
    for model, size, quant in CONFIGS:
        print(f"=== {model}", flush=True)
        perf = speed(model)
        print(f"    {perf}", flush=True)
        path = eval_agent.main([], model=model, arch="react")
        data = json.loads(path.read_text(encoding="utf-8"))
        data["perf"] = perf
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        ollama.Client(host=LLM_HOST).generate(model=model, prompt="", keep_alive=0)        # выгрузить


def latest(model):
    files = sorted(RESULTS.glob(f"agent_react_{model.replace(':', '-')}_*.json"))
    for f in reversed(files):                    # только прогоны этого сравнения (с замером скорости)
        data = json.loads(f.read_text(encoding="utf-8"))
        if "perf" in data:
            return data
    return None


def collect():
    rows = []
    for model, size, quant in CONFIGS:
        data = latest(model)
        if not data:
            continue
        s, p = data["summary"], data["perf"]
        g = s["by_group"]
        rows.append({"model": model, "size": size, "quant": quant, "correct": s["scenarios_correct"],
                     "total": s["scenarios"], "held_out": (g["H"]["correct"], g["H"]["total"]),
                     "norms": (s["norms_correct"], s["norms"]), "precision": s["precision"], "recall": s["recall"],
                     "seconds": s["mean_seconds"], **p})
    return rows


def write_markdown(rows):
    num = lambda x, d=1: f"{x:.{d}f}".replace(".", ",")  # noqa: E731
    lines = [
        "# Сравнение квантизаций",
        "",
        "Одна и та же оценка агента (40 сценариев, ReAct, temperature 0), ноутбук: RTX 4050 6 ГБ, 16 ГБ ОЗУ. "
        "Скорость — генерация 256 токенов на одинаковом запросе после прогрева. 7B в FP16 (15,2 ГБ) "
        "на этом ноутбуке не помещается.",
        "",
        "| Модель | Квантизация | Размер | В видеопамяти | Генерация, ток/с | Верно (из 40) | Отложенные H | "
        "Нормы | Precision / Recall | Время ответа |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['size']} | {r['quant']} | {num(r['size_gb'])} ГБ | {num(r['vram_gb'] / r['size_gb'] * 100, 0)} % | "
            f"{num(r['tokens_per_s'], 0)} | **{r['correct']}** | {r['held_out'][0]}/{r['held_out'][1]} | "
            f"{r['norms'][0]}/{r['norms'][1]} | {num(r['precision'], 2)} / {num(r['recall'], 2)} | "
            f"{num(r['seconds'], 0)} с |")
    lines += ["", "![Квантизации](../docs/quantization.png)", ""]
    (ROOT / "eval" / "QUANTIZATION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot(rows):
    ink, muted, grid, surface = "#1f1f1d", "#898781", "#e1e0d9", "#fcfcfb"
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.6), facecolor=surface)
    labels = [f"{r['size']} {r['quant']}" for r in rows]
    colors = [QUANT_COLOR[r["quant"]] for r in rows]
    panels = [([r["correct"] / r["total"] * 100 for r in rows], "Верных сценариев, %", "{:.0f}"),
              ([r["tokens_per_s"] for r in rows], "Генерация, токенов/с", "{:.0f}"),
              ([r["seconds"] for r in rows], "Среднее время ответа, с", "{:.0f}")]
    for ax, (vals, title, fmt) in zip(axes, panels):
        ax.set_facecolor(surface)
        ax.bar(labels, vals, color=colors, width=0.62, zorder=2)
        for i, v in enumerate(vals):
            ax.text(i, v + max(vals) * 0.02, fmt.format(v), ha="center", fontsize=8.5, color=ink)
        ax.set_title(title, loc="left", fontsize=10, color=ink)
        ax.set_ylim(0, max(vals) * 1.15)
        ax.grid(axis="y", color=grid, lw=0.7, zorder=0)
        ax.tick_params(colors=muted, labelsize=8, length=0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(grid)
    fig.tight_layout()
    fig.savefig(ROOT / "docs" / "quantization.png", dpi=150, facecolor=surface)


def main(report_only):
    if not report_only:
        run()
    rows = collect()
    write_markdown(rows)
    plot(rows)
    print((ROOT / "eval" / "QUANTIZATION.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main("--report" in sys.argv)
