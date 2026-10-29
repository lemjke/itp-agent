"""Оценка детекторов без LLM: события по всем ИТП за весь период против эталона.

    python -m eval.eval_detectors
"""
import sys

from eval.matching import TOLERANCE, load_truth, match_sets
from itp.config import DATA_END, DATA_START
from itp.tools import detect_anomalies, list_substations


def run():
    truth = load_truth()
    found = []
    for s in list_substations()["substations"]:
        res = detect_anomalies(s["id"], DATA_START[:10], DATA_END[:10])
        found += [{**ev, "substation": s["id"]} for ev in res["events"]]
    pairs, missed, extra = match_sets(found, truth)
    tp = len(pairs)
    precision = tp / len(found) if found else 1.0
    recall = tp / len(truth) if truth else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"truth": len(truth), "found": len(found), "tp": tp, "precision": precision,
            "recall": recall, "f1": f1, "missed": missed, "extra": extra}


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    r = run()
    print(f"Детекторы (допуск ±{TOLERANCE.total_seconds() / 3600:.0f} ч): эталон {r['truth']}, найдено {r['found']}, совпало {r['tp']}")
    print(f"Precision {r['precision']:.2f} · Recall {r['recall']:.2f} · F1 {r['f1']:.2f}")
    for t in r["missed"]:
        print(f"  пропущено: ИТП-{t['substation']} {t['kind']} {t['start']}")
    for f in r["extra"]:
        print(f"  лишнее: ИТП-{f['substation']} {f['type']} {f['parameter']} {f['start']}")
