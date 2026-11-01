"""Статистические и ML-детекторы против правил (itp/ml_detect.py).

    python -m eval.eval_ml          # ~10 мин, LLM не нужна; пишет пороги в itp/ml_thresholds.json

1. Калибровка: пороги по 20 чистым рядам (отдельные зёрна) — максимум на них с запасом 10 %.
2. Ложные срабатывания — на других 30 чистых рядах (те же, что в eval_sensitivity).
3. Чувствительность — те же случайные пробы, что в eval_sensitivity (виды, уровни, зёрна).
   Попадание для всех детекторов одинаковое: любое событие, пересекающее аномалию ±6 ч
   (ML не знает типа события; правила здесь тоже засчитываются без учёта типа).
4. Эталон из БД: 11 внесённых аномалий.
Результат: eval/ML.md, docs/ml_detectors.png.
"""
import json
import sys
import zlib
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from data.generate import apply_anomaly  # noqa: E402
from eval.eval_sensitivity import KINDS, N_TRIALS, base_series  # noqa: E402
from eval.matching import load_truth  # noqa: E402
from itp import ml_detect  # noqa: E402
from itp.config import DATA_END, DATA_START  # noqa: E402
from itp.tools import _load, detect_events, with_plan  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CAL_SEEDS = range(20_000, 20_020)
TEST_SEEDS = range(10_000, 10_030)
MARGIN = 1.10
TOL = pd.Timedelta(hours=6)
DETECTORS = ["rules", "statistical", "iforest", "rules+statistical"]
RU = {"rules": "правила", "statistical": "статистический (z-оценки)", "iforest": "Isolation Forest",
      "rules+statistical": "правила + статистический"}
COLOR = {"rules": "#2a78d6", "statistical": "#eb6834", "iforest": "#1baf7a", "rules+statistical": "#1f1f1d"}


def prep(df):
    return with_plan(df.set_index("ts").copy())


def calibrate():
    zmax, ifmax = {}, 0.0
    for seed in CAL_SEEDS:
        d = prep(base_series(seed)[0])
        z, score = ml_detect.iforest_scores(d)
        for p, v in z.abs().max().items():
            zmax[p] = max(zmax.get(p, 0.0), float(v))
        ifmax = max(ifmax, float(score.max()))
    th = {"z": {p: round(v * MARGIN, 2) for p, v in zmax.items()}, "iforest": round(ifmax * 1.02, 4),
          "calibration": f"{len(CAL_SEEDS)} чистых рядов, максимум × {MARGIN} (IF × 1,02)"}
    ml_detect.THRESHOLDS_FILE.write_text(json.dumps(th, ensure_ascii=False, indent=2), encoding="utf-8")
    return th


def all_events(d, th):
    z = ml_detect.zscores(ml_detect.features(d))
    rules = detect_events(d, 1, d.index[0], d.index[-1])
    stat = ml_detect.statistical(d, th["z"], z=z)
    forest = ml_detect.iforest(d, th["iforest"], z=z)
    return {"rules": rules, "statistical": stat, "iforest": forest, "rules+statistical": rules + stat}


def overlaps(ev, a, b, tol=TOL):
    return pd.Timestamp(ev["start"]) <= b + tol and pd.Timestamp(ev["end"]) >= a - tol


def trial(kind, level, seed, th):
    _, _, _, params, dur, _, _ = KINDS[kind]
    df, rng = base_series(seed)                  # та же процедура, что eval_sensitivity.trial
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
    ev = all_events(prep(df), th)
    return {det: any(overlaps(e, a, b) for e in ev[det]) for det in DETECTORS}


def clean_fp(th):
    counts, hours = {det: 0 for det in DETECTORS}, 0
    for seed in TEST_SEEDS:
        d = prep(base_series(seed)[0])
        for det, evs in all_events(d, th).items():
            counts[det] += len(evs)
        hours += len(d)
    return {det: c / hours * 1000 for det, c in counts.items()}, hours


def planted(th):
    truth = load_truth()
    out = {det: {"tp": 0, "extra": 0} for det in DETECTORS}
    for sid in sorted({t["substation"] for t in truth}):
        d = _load(sid, pd.Timestamp(DATA_START), pd.Timestamp(DATA_END)).loc[DATA_START:DATA_END]
        ev = all_events(d, th)
        mine = [t for t in truth if t["substation"] == sid]
        for det in DETECTORS:
            out[det]["tp"] += sum(any(overlaps(e, pd.Timestamp(t["start"]), pd.Timestamp(t["end"])) for e in ev[det])
                                  for t in mine)
            out[det]["extra"] += sum(not any(overlaps(e, pd.Timestamp(t["start"]), pd.Timestamp(t["end"])) for t in mine)
                                     for e in ev[det])
    return out, len(truth)


def main():
    th = calibrate()
    print("пороги |z|:", th["z"], "IF:", th["iforest"], flush=True)
    rows = []
    for kind, (title, unit, levels, *_rest) in KINDS.items():
        if kind == "gap":
            continue
        for level in levels:
            hits = {det: 0 for det in DETECTORS}
            for t in range(N_TRIALS):
                r = trial(kind, level, zlib.crc32(f"{kind}-{level}-{t}".encode()), th)
                for det in DETECTORS:
                    hits[det] += r[det]
            rows.append({"kind": kind, "level": level, **{det: hits[det] / N_TRIALS for det in DETECTORS}})
            print(f"{kind:12} {level:>5} " + " ".join(f"{det}={hits[det] / N_TRIALS:.2f}" for det in DETECTORS), flush=True)
    res = pd.DataFrame(rows)
    fp, hours = clean_fp(th)
    print("ложные на 1000 ч:", fp, flush=True)
    pl, n_truth = planted(th)
    print("эталон:", pl, flush=True)
    write_markdown(res, fp, hours, pl, n_truth, th)
    plot(res)
    return res, fp, pl


def limit(sub, det):
    ok = sub[sub[det] >= 0.9]
    return ok["level"].min() if len(ok) else None


def write_markdown(res, fp, hours, pl, n_truth, th):
    num = lambda x, d=2: f"{x:.{d}f}".replace(".", ",")  # noqa: E731
    lines = [
        "# Статистические и ML-детекторы против правил",
        "",
        f"Пороги — по {len(CAL_SEEDS)} отдельным чистым рядам (`itp/ml_thresholds.json`). Попадание — любое событие, "
        f"пересекающее аномалию ±6 ч, для всех детекторов одинаково. {N_TRIALS} проб на уровень.",
        "",
        "## Надёжно (recall ≥ 0,9) обнаруживается с уровня",
        "",
        "| Аномалия | " + " | ".join(RU[d] for d in DETECTORS) + " |",
        "|---|" + "---|" * len(DETECTORS),
    ]
    for kind, sub in res.groupby("kind", sort=False):
        title, unit = KINDS[kind][0], KINDS[kind][1]
        cells = [f"{num(v, 2).rstrip('0').rstrip(',')} {unit}" if (v := limit(sub, d)) is not None else "—"
                 for d in DETECTORS]
        lines.append(f"| {title} | " + " | ".join(cells) + " |")
    lines += [
        "",
        f"## Ложные срабатывания на {len(TEST_SEEDS)} чистых рядах ({hours} ч)",
        "",
        "| Детектор | Событий на 1000 ч |", "|---|---|",
    ] + [f"| {RU[d]} | {num(fp[d], 2)} |" for d in DETECTORS] + [
        "",
        f"## Эталон из БД ({n_truth} внесённых аномалий)",
        "",
        "| Детектор | Найдено | Лишних событий |", "|---|---|---|",
    ] + [f"| {RU[d]} | {pl[d]['tp']} / {n_truth} | {pl[d]['extra']} |" for d in DETECTORS] + [
        "",
        "## Recall по уровням",
        "",
        "| Аномалия | Сила | " + " | ".join(RU[d] for d in DETECTORS) + " |",
        "|---|---|" + "---|" * len(DETECTORS),
    ] + [f"| {KINDS[r['kind']][0]} | {num(r['level'], 2).rstrip('0').rstrip(',')} {KINDS[r['kind']][1]} | "
         + " | ".join(num(r[d]) for d in DETECTORS) + " |" for _, r in res.iterrows()] + [
        "",
        "![Детекторы](../docs/ml_detectors.png)",
    ]
    (ROOT / "eval" / "ML.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot(res):
    ink, muted, grid, surface = "#1f1f1d", "#898781", "#e1e0d9", "#fcfcfb"
    kinds = [k for k in KINDS if k != "gap"]
    fig, axes = plt.subplots(2, 4, figsize=(13, 6.2), facecolor=surface)
    for ax, kind in zip(axes.flat, kinds):
        sub = res[res["kind"] == kind]
        ax.set_facecolor(surface)
        for det in DETECTORS:
            ax.plot(sub["level"], sub[det], marker="o", ms=3.5, lw=2 if det != "rules+statistical" else 1.2,
                    ls="-" if det != "rules+statistical" else ":", color=COLOR[det], label=RU[det])
        ax.set_title(KINDS[kind][0], fontsize=9.5, color=ink, loc="left")
        ax.set_xlabel(KINDS[kind][1], fontsize=8, color=muted)
        ax.set_ylim(-0.03, 1.05)
        ax.grid(color=grid, lw=0.7)
        ax.tick_params(colors=muted, labelsize=8, length=0)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(grid)
    axes.flat[-1].axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    axes.flat[-1].legend(handles, labels, loc="center", frameon=False, fontsize=9, labelcolor=ink,
                         title="Recall (доля найденных)", title_fontsize=9)
    fig.tight_layout()
    fig.savefig(ROOT / "docs" / "ml_detectors.png", dpi=150, facecolor=surface)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
