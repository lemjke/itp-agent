"""Вопрос агенту из командной строки.

    python cli.py "Что не так с ИТП-3 за последнюю неделю?"
"""
import json
import sys

from itp.agent import ask, render


def show(kind, payload):
    if kind == "tool_call":
        print(f"→ {payload['name']}({json.dumps(payload['args'], ensure_ascii=False)})")
    elif kind == "tool_result":
        r = payload["result"]
        if "error" in r:
            print(f"  ✗ {r['error']}")
        elif "events_found" in r:
            print(f"  найдено событий: {r['events_found']}")
        else:
            print("  ok")
    elif kind == "retry":
        print("↺ прошу модель исправить вызов")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    question = " ".join(sys.argv[1:]) or "Что не так с ИТП-3 за последнюю неделю?"
    print(f"Вопрос: {question}\n")
    result = ask(question, on_event=show)
    print(f"\n{render(result['report'])}\n")
    if result["dropped_findings"]:
        print(f"⚠ отброшено находок без подтверждения детекторами: {result['dropped_findings']}")
    if result["unverified_numbers"]:
        print(f"⚠ числа, которых нет в данных инструментов: {', '.join(result['unverified_numbers'])}")
    print(f"[{result['seconds']} с]")
