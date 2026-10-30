"""Чувствительность детекторов: случайные аномалии разной силы на случайных рядах.

    python -m eval.eval_sensitivity            # ~1–2 мин, LLM не нужна

Для каждого вида аномалии и уровня силы — N_TRIALS проб: новый синтетический ряд
(своя погода и шум), случайное место и длительность, затем все детекторы.
Попадание — событие нужного типа по нужному параметру с пересечением ±6 ч.
Ложные срабатывания считаются на чистых рядах без аномалий.

Результат: eval/SENSITIVITY.md и график docs/sensitivity.png.
"""
import sys
import zlib
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from data.generate import apply_anomaly, outdoor_temperature, simulate_substation  # noqa: E402
from itp.config import DATA_END, DATA_START  # noqa: E402
import itp.tools as tools  # noqa: E402
from itp.tools import detect_events, with_plan  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
N_TRIALS = 20
N_CLEAN = 30
TOL = pd.Timedelta(hours=6)

# вид: (подпись, единица силы, уровни, параметры, длительность ч (min, max) или None, ожидаемый тип события, порог нормы)
KINDS = {
    "shift":       ("Смещение датчика подачи", "°C", [1, 2, 3, 4, 5, 7, 10], ["t1"], (24, 72), "t1_off_schedule", "±3 % от графика ≈ 2,5–3 °C"),
    "drift":       ("Дрейф датчика подачи (к концу)", "°C", [2, 4, 6, 8, 10, 14], ["t1"], (72, 120), "t1_off_schedule", "±3 % от графика"),
    "underheat":   ("Недогрев от источника", "°C", [1, 2, 3, 4, 6, 8, 12], ["t1"], (12, 48), "t1_off_schedule", "±3 % от графика"),
    "high_return": ("Завышенная обратка", "°C", [1, 2, 3, 4, 5, 7, 10], ["t2"], (24, 96), "t2_above_schedule", "+5 % от графика ≈ 2,5–3 °C"),
    "spike":       ("Одиночный выброс", "°C", [2, 3, 4, 6, 8, 12, 20], ["t1", "t2", "t_dhw"], (1, 1), "spike", "k·σ соседних разностей, не менее 2,5 °C"),
    "dhw_low":     ("ГВС ниже нормы", "°C ниже 60", [0.25, 0.5, 1, 2, 3, 5], ["t_dhw"], (3, 8), "dhw_out_of_range", "60 °C"),
    "frozen":      ("«Замёрзший» датчик", "ч", [2, 3, 4, 5, 6, 8, 12, 24], ["t1", "t2", "t_dhw", "flow"], None, "frozen_value", "порог детектора 6 ч"),
    "gap":         ("Пропуск данных", "ч", [1, 2, 4, 8, 24], ["*"], None, "data_gap", "любой пропуск"),
}


def base_series(seed):
    rng = np.random.default_rng(seed)
    idx = pd.date_range(DATA_START, DATA_END, freq="h")
    t_out = outdoor_temperature(idx, rng)
    df = simulate_substation(1, idx, t_out, rng)
    df[["t_outdoor", "t1", "t2", "t_dhw", "p1", "p2", "flow"]] = df[
        ["t_outdoor", "t1", "t2", "t_dhw", "p1", "p2", "flow"]].round(2)
    return df, rng


def run_detectors(df):
    d = with_plan(df.set_index("ts").copy())
    return detect_events(d, 1, d.index[0], d.index[-1])


def hit(events, etype, param, a, b):
    for ev in events:
        if ev["type"] != etype:
            continue
        if param != "*" and param not in ev["parameter"].split(","):
            continue
        if pd.Timestamp(ev["start"]) <= b + TOL and pd.Timestamp(ev["end"]) >= a - TOL:
            return True
    return False


def extra_events(events, a, b):
    """События, не относящиеся к внесённой аномалии (ложные срабатывания в пробе)."""
    margin = pd.Timedelta(hours=24)
    return [ev for ev in events
            if not (pd.Timestamp(ev["start"]) <= b + margin and pd.Timestamp(ev["end"]) >= a - margin)]


def trial(kind, level, seed):
    _, _, _, params, dur, etype, _ = KINDS[kind]
    df, rng = base_series(seed)
    param = params[rng.integers(len(params))]
    hours = int(level) if dur is None else int(rng.integers(dur[0], dur[1] + 1))
    first = df["ts"].iloc[0] + pd.Timedelta(days=5)
    last = df["ts"].iloc[-1] - pd.Timedelta(days=6)
    a = first + pd.Timedelta(hours=int(rng.integers(0, int((last - first) / pd.Timedelta(hours=1)))))
    b = a + pd.Timedelta(hours=hours - 1)
    magnitude = None
    if kind in ("shift", "drift", "underheat", "high_return", "dhw_low"):
        magnitude = level
    elif kind == "spike":
        magnitude = level * (1 if rng.random() < 0.5 else -1)
    apply_anomaly(df, kind, param, a, b, magnitude)
    events = run_detectors(df)
    return hit(events, etype, param, a, b), len(extra_events(events, a, b)), len(df)


def main():
    rows, extra_total, hours_total = [], 0, 0
    for kind, (title, unit, levels, *_rest) in KINDS.items():
        for level in levels:
            hits = 0
            for t in range(N_TRIALS):
                seed = zlib.crc32(f"{kind}-{level}-{t}".encode())   # воспроизводимо между запусками
                h, extra, n = trial(kind, level, seed)
                hits += h
                extra_total += extra
                hours_total += n
            rows.append({"kind": kind, "level": level, "recall": hits / N_TRIALS})
            print(f"{kind:12} {level:>5} {unit:10} recall={hits / N_TRIALS:.2f}", flush=True)

    clean_events = {}
    clean_hours = 0
    for i in range(N_CLEAN):
        df, _ = base_series(10_000 + i)
        for ev in run_detectors(df):
            clean_events[ev["type"]] = clean_events.get(ev["type"], 0) + 1
        clean_hours += len(df)
    fp_clean = sum(clean_events.values()) / clean_hours * 1000
    fp_trials = extra_total / hours_total * 1000
    res = pd.DataFrame(rows)
    sweep = spike_sweep()
    write_markdown(res, fp_clean, clean_events, clean_hours, fp_trials, sweep)
    plot(res)
    print(f"ложные на чистых рядах: {fp_clean:.3f} на 1000 ч {clean_events}; в пробах: {fp_trials:.3f} на 1000 ч")


SWEEP = [(8, 4.0), (6, 3.0), (5, 3.0), (4, 2.5), (3, 2.0)]
SWEEP_LEVELS = [3, 4, 6, 8, 12]


def spike_sweep():
    """Подбор порога выбросов: recall по силе выброса и ложные выбросы на чистых рядах."""
    saved = tools.SPIKE_K, tools.SPIKE_FLOOR
    out = []
    try:
        for k, floor in SWEEP:
            tools.SPIKE_K, tools.SPIKE_FLOOR = k, floor
            rec = {lvl: sum(trial("spike", lvl, zlib.crc32(f"spike-{lvl}-{t}".encode()))[0]
                            for t in range(N_TRIALS)) / N_TRIALS for lvl in SWEEP_LEVELS}
            fp = hours = 0
            for i in range(N_CLEAN):
                df, _ = base_series(10_000 + i)
                fp += sum(ev["type"] == "spike" for ev in run_detectors(df))
                hours += len(df)
            out.append({"k": k, "floor": floor, "recall": rec, "fp": fp, "fp_rate": fp / hours * 1000})
            print(f"spike k={k} floor={floor}: {rec} ложных {fp}", flush=True)
    finally:
        tools.SPIKE_K, tools.SPIKE_FLOOR = saved
    return out


def detection_limit(sub):
    """Наименьший уровень, начиная с которого recall ≥ 0,9 на всех больших уровнях."""
    ok = sub.sort_values("level")
    for lvl in ok["level"]:
        if (ok[ok["level"] >= lvl]["recall"] >= 0.9).all():
            return lvl
    return None


def write_markdown(res, fp_clean, clean_events, clean_hours, fp_trials, sweep):
    lines = [
        "# Чувствительность детекторов",
        "",
        f"{N_TRIALS} случайных проб на каждый уровень силы: новый синтетический ряд (погода и шум), "
        "случайное место и длительность аномалии. Попадание — событие нужного типа по нужному параметру "
        "с пересечением по времени ±6 ч. Запуск: `python -m eval.eval_sensitivity`.",
        "",
        "| Аномалия | Порог | Обнаруживается с (recall ≥ 0,9) | Recall по уровням |",
        "|---|---|---|---|",
    ]
    for kind, (title, unit, levels, _p, _d, _t, threshold) in KINDS.items():
        sub = res[res["kind"] == kind]
        lim = detection_limit(sub)
        per = " · ".join(f"{_fmt(l)}: {r:.2f}".replace(".", ",") for l, r in zip(sub["level"], sub["recall"]))
        lines.append(f"| {title} | {threshold} | {_fmt(lim) + ' ' + unit if lim is not None else 'не достигнут'} "
                     f"| {per} ({unit}) |")
    lines += [
        "",
        f"**Ложные срабатывания.** На {N_CLEAN} чистых рядах ({clean_hours} ч): "
        f"{_fmt(round(fp_clean, 2))} ложных событий на 1000 ч"
        + (f" ({', '.join(f'{k}: {v}' for k, v in sorted(clean_events.items()))})" if clean_events else "")
        + f". Посторонние события в пробах с аномалиями: {_fmt(round(fp_trials, 3))} на 1000 ч.",
        "",
        "![Чувствительность детекторов](../docs/sensitivity.png)",
        "",
        "## Подбор порога выбросов",
        "",
        "Выброс — точка, отличающаяся и от предыдущего, и от следующего часа больше чем на "
        "max(k · σ, нижняя граница), где σ — робастная оценка разброса часовых разностей (MAD · 1,4826). "
        f"Текущий выбор: k = {tools.SPIKE_K:g}, граница {_fmt(tools.SPIKE_FLOOR)} °C.",
        "",
        "| k | Нижняя граница, °C | " + " | ".join(f"Recall при {l} °C" for l in SWEEP_LEVELS)
        + " | Ложных выбросов на чистых рядах |",
        "|---|---|" + "---|" * len(SWEEP_LEVELS) + "---|",
    ] + [
        (f"| {s['k']} | {_fmt(s['floor'])} | " + " | ".join(f"{s['recall'][l]:.2f}" for l in SWEEP_LEVELS)
         + f" | {s['fp']} ({s['fp_rate']:.2f} на 1000 ч) |").replace(".", ",")
        for s in sweep
    ]
    (ROOT / "eval" / "SENSITIVITY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fmt(x):
    return f"{x:g}".replace(".", ",")


def plot(res):
    ink, muted, grid, surface, series = "#1f1f1d", "#898781", "#e1e0d9", "#fcfcfb", "#2a78d6"
    fig, axes = plt.subplots(2, 4, figsize=(13, 6.2), facecolor=surface)
    for ax, (kind, (title, unit, *_rest)) in zip(axes.flat, KINDS.items()):
        sub = res[res["kind"] == kind]
        ax.set_facecolor(surface)
        ax.plot(sub["level"], sub["recall"], color=series, lw=2, marker="o", ms=5,
                markeredgecolor=surface, markeredgewidth=1.5, zorder=3)
        ax.set_ylim(-0.04, 1.06)
        ax.set_yticks([0, 0.5, 1])
        ax.axhline(0.9, color=muted, lw=0.8, ls=(0, (3, 3)), zorder=1)
        ax.set_title(title, fontsize=10.5, color=ink, loc="left")
        ax.set_xlabel(unit, fontsize=9, color=muted)
        ax.tick_params(colors=muted, labelsize=8.5, length=0)
        ax.grid(axis="y", color=grid, lw=0.8)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(grid)
    fig.suptitle("Доля обнаруженных аномалий (recall) в зависимости от силы", x=0.01, ha="left",
                 fontsize=12.5, color=ink)
    fig.text(0.01, 0.925, f"{N_TRIALS} случайных проб на точку · пунктир — 0,9", fontsize=9, color=muted)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(ROOT / "docs" / "sensitivity.png", dpi=150, facecolor=surface)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
