"""Агентный цикл: локальная LLM (Ollama) вызывает инструменты, затем выдаёт отчёт по JSON-схеме.

Фаза 1 — ReAct-подобный цикл с вызовом инструментов (данные, детекторы, нормы).
Фаза 2 — структурированный вывод: модель обязана ответить строго по схеме REPORT_SCHEMA.
"""
import json
import time

import ollama
import pandas as pd

from .config import LLM_HOST, LLM_MODEL, NOW
from .tools import TOOLS, ToolError

MAX_STEPS = 8


def _days_back(n):
    """Период «последние n суток» до текущего момента: 'YYYY-MM-DD — YYYY-MM-DD'."""
    end = pd.Timestamp(NOW).normalize()
    return f"{(end - pd.Timedelta(days=n - 1)):%Y-%m-%d} — {end:%Y-%m-%d}"

SYSTEM_PROMPT = f"""Ты — ассистент инженера по эксплуатации индивидуальных тепловых пунктов (ИТП).
Текущий момент: {NOW}. Текущий год — {NOW[:4]}; если в вопросе год не указан, это {NOW[:4]} год.
Относительные периоды отсчитывай от текущего момента включительно: «последняя неделя» — {_days_back(7)},
«последние 3 дня» — {_days_back(3)}, «вчера» — {_days_back(2)[:10]}.

Как работать:
1. Всегда опирайся на инструменты. Ничего не вычисляй сам и не придумывай чисел — бери их из ответов инструментов.
2. Если номер ИТП или период не ясны — вызови list_substations. Если спрашивают обо всех ИТП —
   вызови detect_anomalies с substation="все". Если инструмент вернул error — исправь аргументы и повтори.
3. Для поиска проблем вызывай detect_anomalies. У каждого события уже есть описание (summary) и норма
   (requirement: code, rule, source) — ссылайся на неё. Нормы своими словами не формулируй.
4. Для каждого события определи категорию и вероятную причину по summary:
   - подача отклоняется от графика, а обратка «почти в норме» — category=sensor_fault: смещение показаний
     датчика t1 (реальный режим не менялся); если в summary сказано, что отклонение растёт в сутки, —
     дрейф датчика t1;
   - подача ниже графика и обратка «тоже отклонилась в ту же сторону» — category=regime_violation:
     недогрев теплоносителя со стороны источника теплоснабжения (это не датчик: изменились оба параметра);
   - обратка выше графика при подаче в норме и выросшем расходе — category=regime_violation:
     разрегулировка системы отопления потребителя (перерасход теплоносителя);
   - ГВС ниже 60 °C или выше 75 °C — category=regime_violation: нарушение режима ГВС;
   - пропуск данных, «замёрзшее» значение, одиночный выброс — category=sensor_fault: проблема измерений
     или связи.
5. Отвечай по-русски, коротко и по делу.
"""

TOOL_SCHEMAS = [
    {"type": "function", "function": {
        "name": "list_substations",
        "description": "Список ИТП (номер, адрес, тип здания) и период, за который есть данные.",
        "parameters": {"type": "object", "properties": {}},
    }},
    {"type": "function", "function": {
        "name": "query_sensors",
        "description": "Посуточные средние значения параметров ИТП за период и плановые t1/t2 по температурному графику.",
        "parameters": {"type": "object", "properties": {
            "substation": {"type": "string", "description": "Номер ИТП, например 'ИТП-3'"},
            "start": {"type": "string", "description": "Начало периода, YYYY-MM-DD"},
            "end": {"type": "string", "description": "Конец периода, YYYY-MM-DD"},
            "parameters": {"type": "array", "items": {"type": "string",
                           "enum": ["t_outdoor", "t1", "t2", "t_dhw", "p1", "p2", "flow"]},
                           "description": "Какие параметры вернуть (по умолчанию все)"},
        }, "required": ["substation", "start", "end"]},
    }},
    {"type": "function", "function": {
        "name": "detect_anomalies",
        "description": "Найти события за период: пропуски данных, «замёрзшие» значения, выбросы, отклонения "
                       "t1/t2 от графика, ГВС вне нормы, низкий перепад давлений. Возвращает код нормы и контекст.",
        "parameters": {"type": "object", "properties": {
            "substation": {"type": "string", "description": "Номер ИТП, например 'ИТП-3', или 'все' — все ИТП сразу"},
            "start": {"type": "string", "description": "Начало периода, YYYY-MM-DD"},
            "end": {"type": "string", "description": "Конец периода, YYYY-MM-DD"},
        }, "required": ["substation", "start", "end"]},
    }},
    {"type": "function", "function": {
        "name": "get_requirement",
        "description": "Текст нормативного требования и источник по коду из detect_anomalies.",
        "parameters": {"type": "object", "properties": {
            "code": {"type": "string", "description": "Код требования, например 'T2_SCHEDULE'"},
        }, "required": ["code"]},
    }},
]

REPORT_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string", "description": "Вывод в 1–3 предложениях"},
        "findings": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "substation": {"type": "string"},
                "parameter": {"type": "string", "description": "Код параметра из события",
                              "enum": ["t1", "t2", "t_dhw", "p1", "p2", "flow", "t_outdoor", "p1,p2", "*"]},
                "start": {"type": "string"},
                "end": {"type": "string"},
                "what": {"type": "string", "description": "Что произошло, с числами из инструментов"},
                "category": {"type": "string", "enum": ["sensor_fault", "regime_violation", "unknown"]},
                "hypothesis": {"type": "string", "description": "Вероятная причина и на чём основан вывод"},
                "requirement_code": {"type": "string"},
                "requirement_source": {"type": "string"},
            },
            "required": ["substation", "parameter", "start", "end", "what", "category",
                         "hypothesis", "requirement_code", "requirement_source"],
        }},
        "recommendations": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["summary", "findings", "recommendations"],
}

FINAL_INSTRUCTION = (
    "Сформируй итоговый отчёт строго по JSON-схеме. В findings — каждое событие, которое вернул detect_anomalies, "
    "по одному элементу на событие, с его датами. В what перескажи summary события с числами. "
    "category и hypothesis — по правилам из пункта 4. requirement_code и requirement_source возьми из поля "
    "requirement события. Если событий нет — пустой findings."
)


def _run_tool(name, args):
    fn = TOOLS.get(name)
    if fn is None:
        return {"error": f"Нет инструмента {name}"}
    try:
        return fn(**(args or {}))
    except ToolError as exc:
        return {"error": str(exc)}
    except TypeError as exc:
        return {"error": f"Неверные аргументы: {exc}"}


def ask(question, on_event=None, model=LLM_MODEL):
    """Задать вопрос агенту. on_event(kind, payload) получает ход работы для отображения."""
    emit = on_event or (lambda *_: None)
    client = ollama.Client(host=LLM_HOST)
    options = {"temperature": 0, "num_ctx": 8192}
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": question}]
    trace = []
    t0 = time.time()

    last_error, nudged = None, False
    for _ in range(MAX_STEPS):
        resp = client.chat(model=model, messages=messages, tools=TOOL_SCHEMAS, options=options)
        msg = resp.message
        messages.append(msg)
        if not msg.tool_calls:
            # Модель сдаётся после ошибки инструмента — один раз просим исправить вызов
            if last_error and not nudged:
                nudged = True
                emit("retry", {"error": last_error})
                messages.append({"role": "user", "content":
                                 f"Последний вызов инструмента вернул ошибку: {last_error} "
                                 "Исправь аргументы по этой подсказке и повтори вызов."})
                continue
            break
        last_error = None
        for call in msg.tool_calls:
            name, args = call.function.name, dict(call.function.arguments or {})
            emit("tool_call", {"name": name, "args": args})
            result = _run_tool(name, args)
            if "error" in result:
                last_error = result["error"]
            trace.append({"tool": name, "args": args, "result": result})
            emit("tool_result", {"name": name, "result": result})
            messages.append({"role": "tool", "tool_name": name,
                             "content": json.dumps(result, ensure_ascii=False, default=str)})

    messages.append({"role": "user", "content": FINAL_INSTRUCTION})
    resp = client.chat(model=model, messages=messages, format=REPORT_SCHEMA, options=options)
    report = json.loads(resp.message.content)
    emit("report", report)
    return {"report": report, "trace": trace, "seconds": round(time.time() - t0, 1)}


CATEGORY_RU = {"sensor_fault": "неисправность измерений", "regime_violation": "нарушение режима",
               "unknown": "причина не определена"}


def render(report):
    """Отчёт в читаемый текст."""
    lines = [report["summary"], ""]
    for i, f in enumerate(report["findings"], 1):
        lines += [
            f"{i}. {f['substation']} · {f['parameter']} · {f['start']} — {f['end']}",
            f"   Что: {f['what']}",
            f"   Тип: {CATEGORY_RU.get(f['category'], f['category'])}. {f['hypothesis']}",
            f"   Норма: {f['requirement_code']} — {f['requirement_source']}",
        ]
    if report.get("recommendations"):
        lines += ["", "Рекомендации:"] + [f"  • {r}" for r in report["recommendations"]]
    return "\n".join(lines)
