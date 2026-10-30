"""Агентный цикл: локальная LLM (Ollama) вызывает инструменты, затем выдаёт отчёт по JSON-схеме.

Фаза 1 — ReAct-подобный цикл с вызовом инструментов (данные, детекторы, нормы).
Фаза 2 — структурированный вывод: модель обязана ответить строго по схеме REPORT_SCHEMA.
"""
import json
import re
import time

import ollama
import pandas as pd

from .config import LLM_HOST, LLM_MODEL, NOW
from .grounding import ground
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
Месяцы: «в январе» — {NOW[:4]}-01-01 — {NOW[:4]}-01-31, «в феврале» — {NOW[:4]}-02-01 — {NOW[:4]}-02-28.

Как работать:
1. Всегда опирайся на инструменты. Ничего не вычисляй сам и не придумывай чисел — бери их из ответов инструментов.
2. ИТП можно указать номером ('ИТП-3') или адресом ('дом на Северном') — инструменты понимают оба варианта;
   проверяй только тот ИТП, о котором спрашивают.
   Если спрашивают обо всех ИТП — вызови detect_anomalies с substation="все". Если период не указан —
   бери последнюю неделю. Если инструмент вернул error — исправь аргументы и повтори.
   Если в ответе инструмента есть поле note — обязательно перескажи его пользователю.
3. Вопрос о значениях («какая средняя…», «максимальная…», «сколько…») — вызови query_sensors и ответь
   числами из period_summary. Вопрос о проблемах, жалобах, «что не так» — вызови detect_anomalies.
   Жалобы на жару в помещениях ищи среди отклонений подачи или обратки выше графика, на холод — ниже графика,
   на горячую воду — среди событий ГВС.
4. Для поиска проблем вызывай detect_anomalies. У каждого события уже есть описание (summary) и норма
   (requirement: code, rule, source) — ссылайся на неё. Нормы своими словами не формулируй.
5. Для каждого события определи категорию и вероятную причину по summary:
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
6. Отвечай по-русски, коротко и по делу.
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
        "answer": {"type": "string",
                   "description": "Прямой ответ на вопрос пользователя: 1–4 законченных предложения "
                                  "с числами из инструментов"},
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
    "required": ["answer", "findings", "recommendations"],
}

FINAL_INSTRUCTION = (
    "Сформируй итоговый отчёт строго по JSON-схеме. В answer — прямой ответ именно на заданный вопрос "
    "законченными предложениями (не одним словом или числом) "
    "(если спрашивали значение — напиши само число из period_summary, например «54,6 °C», а не ссылку на поле; "
    "если в ответе инструмента был note — "
    "перескажи его). В findings — события, которые вернул detect_anomalies и которые относятся к вопросу, "
    "по одному элементу на событие, с его датами; если detect_anomalies не вызывался или событий нет — "
    "пустой findings. В what перескажи summary события с числами. category и hypothesis — по правилам "
    "из пункта 5. requirement_code и requirement_source возьми из поля requirement события."
)

OFF_TOPIC_ANSWER = (
    "Я помогаю только с диагностикой тепловых пунктов: могу проверить ИТП за период, найти неисправности "
    "датчиков и нарушения режима, назвать значения параметров. Например: «Что не так с ИТП-3 за последнюю неделю?»"
)

PLACEHOLDER = re.compile(r"\$\{|\{\{|period_summary|events\[")
_NUM = re.compile(r"\d+(?:[.,]\d+)?")


def _numbers(text):
    """Числа из текста в нормализованном виде (запятая → точка, без ведущих нулей)."""
    out = set()
    for n in _NUM.findall(str(text)):
        n = n.replace(",", ".")
        out.add(n.lstrip("0") or "0")
        if "." in n:
            out.add(n.rstrip("0").rstrip(".").lstrip("0") or "0")
    return out


def unverified_numbers(report, question, trace):
    """Числа из ответа, которых нет ни в вопросе, ни в результатах инструментов."""
    known = _numbers(question) | _numbers(json.dumps([s["result"] for s in trace], ensure_ascii=False))
    text = report.get("answer", "") + " " + " ".join(f.get("what", "") for f in report.get("findings", []))
    return sorted(_numbers(text) - known, key=lambda x: float(x))


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


ARCHITECTURES = {
    "react": "ReAct: модель сама выбирает следующий вызов по результатам предыдущих",
    "plan": "План → выполнение: модель сразу составляет весь план вызовов, обвязка его выполняет",
    "workflow": "Фиксированный workflow: модель только разбирает вопрос, вызовы делает код",
}

TOOL_NAMES = ["list_substations", "query_sensors", "detect_anomalies", "get_requirement"]
PARAM_CODES = ["t_outdoor", "t1", "t2", "t_dhw", "p1", "p2", "flow"]

PLAN_SCHEMA = {
    "type": "object",
    "properties": {"steps": {"type": "array", "maxItems": 6, "items": {
        "type": "object",
        "properties": {
            "tool": {"type": "string", "enum": TOOL_NAMES},
            "substation": {"type": "string", "description": "'ИТП-3', адрес или 'все'"},
            "start": {"type": "string", "description": "YYYY-MM-DD"},
            "end": {"type": "string", "description": "YYYY-MM-DD"},
            "parameters": {"type": "array", "items": {"type": "string", "enum": PARAM_CODES}},
        },
        "required": ["tool"],
    }}},
    "required": ["steps"],
}

PLAN_INSTRUCTION = (
    "Составь план вызовов инструментов, который нужен для ответа на вопрос, — сразу весь, "
    "по одному шагу на вызов. Инструменты: list_substations (без аргументов); query_sensors(substation, "
    "start, end, parameters) — значения параметров; detect_anomalies(substation, start, end) — поиск "
    "проблем, substation может быть 'все'. Если вопрос не о тепловых пунктах — пустой план."
)

INTENT_SCHEMA = {
    "type": "object",
    "properties": {
        "intent": {"type": "string", "enum": ["diagnosis", "value", "offtopic"],
                   "description": "diagnosis — найти проблемы; value — назвать значение параметра; "
                                  "offtopic — вопрос не о тепловых пунктах"},
        "substations": {"type": "array", "items": {"type": "string"},
                        "description": "ИТП как в вопросе: 'ИТП-3', адрес или 'все'"},
        "start": {"type": "string", "description": "YYYY-MM-DD"},
        "end": {"type": "string", "description": "YYYY-MM-DD"},
        "parameter": {"type": "string", "enum": PARAM_CODES + ["none"]},
    },
    "required": ["intent", "substations", "start", "end", "parameter"],
}

INTENT_INSTRUCTION = (
    "Разбери вопрос: что спрашивают (intent), про какие ИТП, за какой период и, если спрашивают "
    "значение, — какой параметр (t1 подача, t2 обратка, t_dhw ГВС, p1/p2 давления, flow расход, "
    "t_outdoor наружная температура). Если период не указан — последняя неделя."
)


def _safe_json(text):
    """JSON из ответа модели; обрезанный по лимиту токенов ответ — пустой словарь (ошибка засчитается агенту)."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {}


def _execute(name, args, trace, messages, emit):
    emit("tool_call", {"name": name, "args": args})
    result = _run_tool(name, args)
    trace.append({"tool": name, "args": args, "result": result})
    emit("tool_result", {"name": name, "result": result})
    messages.append({"role": "user", "content": f"Результат {name}({json.dumps(args, ensure_ascii=False)}): "
                                                + json.dumps(result, ensure_ascii=False, default=str)})
    return result


def _collect_react(client, model, messages, options, emit, chat_kw):
    trace = []
    last_error, nudged = None, False
    for _ in range(MAX_STEPS):
        resp = client.chat(model=model, messages=messages, tools=TOOL_SCHEMAS, options=options, **chat_kw)
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
    return trace


def _collect_plan(client, model, messages, options, emit, chat_kw):
    resp = client.chat(model=model, messages=messages + [{"role": "user", "content": PLAN_INSTRUCTION}],
                       format=PLAN_SCHEMA, options={**options, "num_predict": 600}, **chat_kw)
    steps = _safe_json(resp.message.content).get("steps", [])[:6]
    emit("plan", {"steps": steps})
    keys = {"list_substations": [], "query_sensors": ["substation", "start", "end", "parameters"],
            "detect_anomalies": ["substation", "start", "end"]}
    trace = []
    for st in steps:
        name = st.get("tool")
        if name in keys:
            args = {k: st[k] for k in keys[name] if st.get(k) not in (None, "", [])}
            _execute(name, args, trace, messages, emit)
    return trace


def _collect_workflow(client, model, messages, options, emit, chat_kw):
    resp = client.chat(model=model, messages=messages + [{"role": "user", "content": INTENT_INSTRUCTION}],
                       format=INTENT_SCHEMA, options={**options, "num_predict": 300}, **chat_kw)
    intent = _safe_json(resp.message.content)
    emit("intent", intent)
    trace = []
    if intent.get("intent") == "offtopic":
        return trace
    for sub in (intent.get("substations") or ["все"])[:5]:
        args = {"substation": sub, "start": intent.get("start", ""), "end": intent.get("end", "")}
        if intent.get("intent") == "value" and intent.get("parameter") not in (None, "none"):
            if str(sub).strip().lower() in ("все", "all", "*"):
                continue
            _execute("query_sensors", {**args, "parameters": [intent["parameter"]]}, trace, messages, emit)
        else:
            _execute("detect_anomalies", args, trace, messages, emit)
    return trace


COLLECTORS = {"react": _collect_react, "plan": _collect_plan, "workflow": _collect_workflow}


def ask(question, on_event=None, model=LLM_MODEL, arch="react"):
    """Задать вопрос агенту. on_event(kind, payload) получает ход работы для отображения.

    arch — схема работы агента (см. ARCHITECTURES). Финальная стадия и все проверки общие.
    """
    emit = on_event or (lambda *_: None)
    client = ollama.Client(host=LLM_HOST)
    options = {"temperature": 0, "num_ctx": 8192}
    chat_kw = {"think": False} if model.startswith("qwen3") else {}    # без «размышлений» — честное сравнение
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": question}]
    t0 = time.time()

    trace = COLLECTORS[arch](client, model, messages, options, emit, chat_kw)

    if not trace:
        # Модель не обратилась ни к одному инструменту — значит, вопрос вне темы.
        # Отчёт по схеме не строим: без данных ему не на что опереться.
        report = {"answer": OFF_TOPIC_ANSWER, "findings": [], "recommendations": []}
        emit("report", report)
        return {"report": report, "trace": trace, "dropped_findings": 0, "unverified_numbers": [],
                "off_topic": True, "seconds": round(time.time() - t0, 1)}

    messages.append({"role": "user", "content": FINAL_INSTRUCTION})
    final_opts = {**options, "num_predict": 2500}       # страховка от зацикливания структурированного вывода
    resp = client.chat(model=model, messages=messages, format=REPORT_SCHEMA, options=final_opts, **chat_kw)
    raw = _safe_json(resp.message.content)
    if PLACEHOLDER.search(raw.get("answer", "")):
        # Модель вставила шаблон вместо числа — один раз просим переписать ответ
        emit("retry", {"error": "в ответе шаблон вместо значения"})
        messages += [{"role": "assistant", "content": resp.message.content},
                     {"role": "user", "content": "В answer ссылка на поле вместо значения. Перепиши answer, "
                                                 "подставив сами числа из ответа инструмента."}]
        resp = client.chat(model=model, messages=messages, format=REPORT_SCHEMA, options=final_opts, **chat_kw)
        raw = _safe_json(resp.message.content)
    for key, default in (("answer", ""), ("findings", []), ("recommendations", [])):
        raw.setdefault(key, default)
    report, dropped = ground(raw, trace)          # находки без опоры на детекторы отбрасываются
    # Сообщения инструментов о сокращении периода доводим до пользователя в любом случае
    notes = list(dict.fromkeys(s["result"]["note"] for s in trace if s["result"].get("note")))
    if notes and "только с" not in report["answer"]:
        report["answer"] += "\n\nℹ️ " + " ".join(n.replace(" Сообщи это пользователю.", "") for n in notes)
    emit("report", report)
    return {"report": report, "trace": trace, "dropped_findings": dropped,
            "unverified_numbers": unverified_numbers(report, question, trace),
            "off_topic": False, "seconds": round(time.time() - t0, 1)}


CATEGORY_RU = {"sensor_fault": "неисправность измерений", "regime_violation": "нарушение режима",
               "unknown": "причина не определена"}


def render(report):
    """Отчёт в читаемый текст."""
    lines = [report["answer"], ""]
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
