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
from .compare import compare_periods
from .tools import TOOLS as BASE_TOOLS, ToolError, _by_address

TOOLS = {**BASE_TOOLS, "compare_periods": compare_periods}

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
   Поле suspicions в ответе detect_anomalies — статистические подозрения без нарушения норм (например,
   небольшое смещение датчика, которое проверка по нормам не видит). Упомяни их в answer отдельной фразой
   как требующие проверки, в findings не включай.
6. Вопрос о нормах и правилах («какая норма…», «допустимо ли…», «сколько можно…», «что положено…»,
   «что делать по правилам…») — вызови search_norms. В query пиши официальными терминами документов:
   «подающий трубопровод», а не «подача»; «обратный трубопровод», а не «обратка»; «горячее водоснабжение»,
   «тепловой пункт», «узел учёта». Отвечай только по тексту найденных пунктов и называй источник ровно так,
   как в поле source. Если в найденных пунктах ответа нет — так и скажи, не додумывай.
   Вопрос о допустимости значения без ИТП и периода («давление 0,7 МПа в кране — это нормально?», «подача
   на 2 % ниже графика — допустимо?») — тоже вопрос о норме: search_norms, а не detect_anomalies. В ответе
   назови предельное значение из пункта и сравни с ним значение из вопроса.
7. Сравнение двух периодов одного ИТП («сравни январь и февраль», «стало ли лучше, чем в прошлом месяце»)
   — вызови compare_periods. В answer назови главные изменения числами из change: часы вне нормы, события,
   тепло на градусо-сутки. Сырое тепло за период зависит от погоды — сравнивай тепло на градусо-сутки.
8. Отвечай по-русски, коротко и по делу.
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
    {"type": "function", "function": {
        "name": "compare_periods",
        "description": "Сравнить два периода одного ИТП: отклонения от графика, часы вне нормы, события детекторов, "
                       "тепло (в том числе на градусо-сутки, с поправкой на погоду).",
        "parameters": {"type": "object", "properties": {
            "substation": {"type": "string", "description": "Номер ИТП, например 'ИТП-3', или адрес"},
            "start_1": {"type": "string", "description": "Начало первого периода, YYYY-MM-DD"},
            "end_1": {"type": "string", "description": "Конец первого периода, YYYY-MM-DD"},
            "start_2": {"type": "string", "description": "Начало второго периода, YYYY-MM-DD"},
            "end_2": {"type": "string", "description": "Конец второго периода, YYYY-MM-DD"},
        }, "required": ["substation", "start_1", "end_1", "start_2", "end_2"]},
    }},
    {"type": "function", "function": {
        "name": "search_norms",
        "description": "Поиск пунктов нормативных документов по смыслу: правила технической эксплуатации "
                       "тепловых установок (приказ Минэнерго № 511), правила коммунальных услуг (ПП № 354), "
                       "правила коммерческого учёта тепловой энергии (ПП № 1034), СанПиН 2.1.3684-21. "
                       "Возвращает текст пунктов и источник.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "Вопрос пользователя, пересказанный официальными "
                                                       "терминами документов (без примеров из этого описания)"},
        }, "required": ["query"]},
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
        "sources": {"type": "array", "items": {"type": "string"},
                    "description": "Пункты документов из search_norms, на которые опирается answer, — "
                                   "ровно как в поле source; пусто, если search_norms не вызывался"},
    },
    "required": ["answer", "findings", "recommendations", "sources"],
}

FINAL_INSTRUCTION = (
    "Сформируй итоговый отчёт строго по JSON-схеме. В answer — прямой ответ именно на заданный вопрос "
    "законченными предложениями (не одним словом или числом) "
    "(если спрашивали значение — напиши само число из period_summary, например «54,6 °C», а не ссылку на поле; "
    "если в ответе инструмента был note — "
    "перескажи его). В findings — события, которые вернул detect_anomalies и которые относятся к вопросу, "
    "по одному элементу на событие, с его датами; если detect_anomalies не вызывался или событий нет — "
    "пустой findings. В what перескажи summary события с числами. category и hypothesis — по правилам "
    "из пункта 5. requirement_code и requirement_source возьми из поля requirement события. "
    "Если вызывался search_norms — в answer перескажи требование с его числами и назови источник из поля source."
)

OFF_TOPIC_ANSWER = (
    "Я помогаю только с диагностикой тепловых пунктов: могу проверить ИТП за период, найти неисправности "
    "датчиков и нарушения режима, назвать значения параметров, найти норму в правилах. "
    "Например: «Что не так с ИТП-3 за последнюю неделю?»"
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

TOOL_NAMES = ["list_substations", "query_sensors", "detect_anomalies", "get_requirement", "search_norms"]
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
            "query": {"type": "string", "description": "для search_norms: что ищем в нормах"},
        },
        "required": ["tool"],
    }}},
    "required": ["steps"],
}

PLAN_INSTRUCTION = (
    "Составь план вызовов инструментов, который нужен для ответа на вопрос, — сразу весь, "
    "по одному шагу на вызов. Инструменты: list_substations (без аргументов); query_sensors(substation, "
    "start, end, parameters) — значения параметров; detect_anomalies(substation, start, end) — поиск "
    "проблем, substation может быть 'все'; search_norms(query) — поиск пункта нормативных документов. "
    "Если вопрос не о тепловых пунктах — пустой план."
)

INTENT_SCHEMA = {
    "type": "object",
    "properties": {
        "intent": {"type": "string", "enum": ["diagnosis", "value", "norm", "offtopic"],
                   "description": "diagnosis — найти проблемы; value — назвать значение параметра; "
                                  "norm — что требуют нормы и правила; offtopic — вопрос не о тепловых пунктах"},
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
            "detect_anomalies": ["substation", "start", "end"], "search_norms": ["query"]}
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
    if intent.get("intent") == "norm":          # запрос — сам вопрос пользователя, без переформулировки
        _execute("search_norms", {"query": messages[-1]["content"]}, trace, messages, emit)
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

# ---------- контекст диалога ----------
CONTEXT_MODES = {
    "none": "без истории: каждый вопрос с нуля",
    "rewrite": "уточнение переписывается в самостоятельный вопрос, дальше агент работает как обычно",
    "messages": "предыдущие вопросы и ответы передаются модели как часть переписки",
}
HISTORY_TURNS = 3            # сколько последних пар «вопрос → ответ» учитывать
HISTORY_ANSWER_CHARS = 300   # ответ в истории — только начало: модели нужен предмет разговора, а не детали

REWRITE_SCHEMA = {
    "type": "object",
    "properties": {
        "refers_to_history": {"type": "boolean",
                              "description": "true — новый вопрос неполный и ссылается на предыдущие («а за январь?», "
                                             "«а на ИТП-4?», «а максимальная?», «подробнее про ИТП-3», «там»); "
                                             "false — вопрос понятен сам по себе"},
        "question": {"type": "string", "description": "самостоятельный вопрос (если refers_to_history)"},
    },
    "required": ["refers_to_history", "question"],
}
REWRITE_INSTRUCTION = (
    "Тебе дают предыдущие вопросы пользователя об индивидуальных тепловых пунктах (ИТП) и новый вопрос. "
    "Сначала реши, ссылается ли новый вопрос на предыдущие (refers_to_history). Вопрос о нормах и правилах "
    "или вопрос, в котором уже названы ИТП и период, понятен сам по себе. Если ссылается — допиши в него "
    "из предыдущих вопросов то, чего не хватает: номер ИТП или адрес, период, параметр. Сохрани слова и вид "
    "вопроса пользователя: просьбу проверить или узнать, что не так, оставь проверкой, вопрос о значении — "
    "вопросом о значении. Периоды бери дословно («за последнюю неделю», «в январе», «с 20 по 28 февраля»). "
    "Не отвечай на вопрос и ничего не добавляй от себя."
)
# Самостоятельный вопрос — назван ИТП и период, нет отсылок к сказанному: переписывать нечего.
# Проверяется кодом: модель иногда «уточняла» и такие вопросы, подставляя чужие даты.
_PERIOD = re.compile(r"недел|\bдн[еяи]|вчера|сегодня|январ|феврал|март|декабр|зим|\d{1,2}\s*(?:по|-|—)\s*\d{1,2}|"
                     r"\d{4}-\d\d|week|january|february|december|yesterday|today", re.I)
_SUBSTATION = re.compile(r"итп\s*[-№]?\s*\d|тепловы\w* пункт|substation", re.I)
# Отсылка, которой не заменят названные ИТП и период; «там», «она» при них — разговорные, не отсылки
# («чо там с итп 2 10 февраля»)
_REFERENCE = re.compile(r"\b(?:тот же|того же|то же|той же|тех же)\b", re.I)


def self_contained(question):
    has_sub = bool(_SUBSTATION.search(question) or _by_address(question))
    return has_sub and bool(_PERIOD.search(question)) and not _REFERENCE.search(question)


# Примеры для модели — на другие ИТП, периоды и параметры, чем в сценариях eval/dialogs.py
REWRITE_EXAMPLES = [
    ("Проверь ИТП-5 с 1 по 10 февраля", "А с 11 по 20?", True, "Проверь ИТП-5 с 11 по 20 февраля"),
    ("Проверь ИТП-5 с 1 по 10 февраля", "Какое давление в системе ГВС допускается по правилам?", False, ""),
    ("Проверь ИТП-5 с 1 по 10 февраля", "Что было на ИТП-2 вчера?", False, ""),
    ("Проверь ИТП-5 с 1 по 10 февраля", "А давление там какое было?",
     True, "Какое давление было на ИТП-5 с 1 по 10 февраля?"),
    ("Проверь ИТП-5 с 1 по 10 февраля", "А ИТП-2 за то же время?", True, "Проверь ИТП-2 с 1 по 10 февраля"),
    ("Какой перепад давлений положен по правилам?", "А на ИТП-5 вчера он был в норме?",
     True, "Был ли перепад давлений на ИТП-5 вчера в норме?"),
    ("Проверь ИТП-5 с 1 по 10 февраля", "Спасибо, а как дела?", False, ""),
]


def _rewrite_examples():
    msgs = []
    for prev, new, refers, rewritten in REWRITE_EXAMPLES:
        msgs += [{"role": "user", "content": f"Предыдущие вопросы:\n1. {prev}\n\nНовый вопрос: {new}"},
                 {"role": "assistant", "content": json.dumps({"refers_to_history": refers, "question": rewritten},
                                                             ensure_ascii=False)}]
    return msgs


def contextualize(question, history, client, model, chat_kw=None):
    """Уточняющий вопрос → самостоятельный («А за январь?» → «Что не так с ИТП-3 в январе?»).

    Модель видит только предыдущие вопросы, без ответов: из ответов она переносила в вопрос даты,
    проценты и даже сам ответ.
    """
    if self_contained(question):
        return question
    previous = "\n".join(f"{i}. {h['question']}" for i, h in enumerate(history[-HISTORY_TURNS:], 1))
    prompt = f"Предыдущие вопросы:\n{previous}\n\nНовый вопрос: {question}"
    messages = [{"role": "system", "content": REWRITE_INSTRUCTION}, *_rewrite_examples(),
                {"role": "user", "content": prompt}]
    resp = client.chat(model=model, messages=messages, format=REWRITE_SCHEMA,
                       options={"temperature": 0, "num_predict": 150}, **(chat_kw or {}))
    out = _safe_json(resp.message.content)
    rewritten = str(out.get("question", "")).strip()
    # «Не трогать» решает код по флагу, а не модель, переписывая вопрос «без изменений»: 7B-модель
    # на просьбу вернуть вопрос как есть всё равно приклеивала к нему ИТП и период из истории.
    return rewritten if out.get("refers_to_history") is True and rewritten else question


def ask(question, on_event=None, model=LLM_MODEL, arch="react", history=None, context="rewrite"):
    """Задать вопрос агенту. on_event(kind, payload) получает ход работы для отображения.

    arch — схема работы агента (см. ARCHITECTURES). Финальная стадия и все проверки общие.
    history — предыдущие пары [{"question", "answer"}] этого диалога (question — уже самостоятельный вопрос),
    context — как их учитывать (см. CONTEXT_MODES).
    """
    emit = on_event or (lambda *_: None)
    client = ollama.Client(host=LLM_HOST)
    options = {"temperature": 0, "num_ctx": 8192}
    chat_kw = {"think": False} if model.startswith("qwen3") else {}    # без «размышлений» — честное сравнение
    t0 = time.time()
    original, history = question, (history or [])[-HISTORY_TURNS:]
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history and context == "rewrite":
        question = contextualize(question, history, client, model, chat_kw)
        if question != original:
            emit("rewrite", {"question": question})
    elif history and context == "messages":
        for h in history:
            messages += [{"role": "user", "content": h["question"]},
                         {"role": "assistant", "content": h["answer"][:HISTORY_ANSWER_CHARS]}]
    messages.append({"role": "user", "content": question})

    trace = COLLECTORS[arch](client, model, messages, options, emit, chat_kw)

    if not trace:
        # Модель не обратилась ни к одному инструменту — значит, вопрос вне темы.
        # Отчёт по схеме не строим: без данных ему не на что опереться.
        report = {"answer": OFF_TOPIC_ANSWER, "findings": [], "recommendations": []}
        emit("report", report)
        return {"report": report, "trace": trace, "dropped_findings": 0, "unverified_numbers": [],
                "off_topic": True, "asked": original, "question": question, "seconds": round(time.time() - t0, 1)}

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
    for key, default in (("answer", ""), ("findings", []), ("recommendations", []), ("sources", [])):
        raw.setdefault(key, default)
    report, dropped = ground(raw, trace)          # находки без опоры на детекторы отбрасываются
    # Сообщения инструментов о сокращении периода доводим до пользователя в любом случае
    notes = list(dict.fromkeys(s["result"]["note"] for s in trace if s["result"].get("note")))
    if notes and "только с" not in report["answer"]:
        report["answer"] += "\n\nℹ️ " + " ".join(n.replace(" Сообщи это пользователю.", "") for n in notes)
    emit("report", report)
    return {"report": report, "trace": trace, "dropped_findings": dropped,
            "unverified_numbers": unverified_numbers(report, f"{original} {question}", trace),
            "off_topic": False, "asked": original, "question": question, "seconds": round(time.time() - t0, 1)}


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
