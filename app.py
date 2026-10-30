"""Веб-интерфейс ИТП-агента.

    streamlit run app.py
"""
import json

import pandas as pd
import streamlit as st

from itp.agent import CATEGORY_RU, ask
from itp.charts import event_chart, overview_chart
from itp.config import LLM_MODEL, NOW
from itp.tools import PARAM_NAMES, list_substations

st.set_page_config(page_title="ИТП-агент", page_icon="🌡️", layout="wide")

EXAMPLES = [
    "Что не так с ИТП-3 за последнюю неделю?",
    "Что происходило на ИТП-2 с 20 по 28 февраля?",
    "Есть ли проблемы с датчиками на ИТП-4 в феврале?",
    "Проверь ИТП-1 и ИТП-5 за последнюю неделю",
    "Насколько обратка может быть выше графика по правилам?",
]
CATEGORY_COLOR = {"sensor_fault": "blue", "regime_violation": "red", "unknown": "gray"}
TOOL_RU = {
    "list_substations": "Смотрю список ИТП",
    "query_sensors": "Запрашиваю сводку датчиков",
    "detect_anomalies": "Запускаю детекторы",
    "get_requirement": "Беру текст нормы",
    "search_norms": "Ищу в нормативных документах",
}


@st.cache_data
def substations():
    return list_substations()


def fmt_ts(ts):
    try:
        return pd.Timestamp(ts).strftime("%d.%m %H:%M")
    except (ValueError, TypeError):
        return str(ts)


def period_label(result):
    """«период: 7 сут., 22.02 00:00 — 28.02 23:00» — сколько данных реально проверено (обе даты включительно)."""
    period = result.get("period") if isinstance(result, dict) else None
    if not period:
        return ""
    s, e = pd.Timestamp(period["start"]), pd.Timestamp(period["end"])
    hours = int((e - s) / pd.Timedelta(hours=1)) + 1
    length = f"{hours // 24} сут." if hours % 24 == 0 else f"{hours} ч"
    return f"период: {length}, {fmt_ts(s)} — {fmt_ts(e)}"


def param_label(p):
    return {"*": "все параметры", "p1,p2": "давления"}.get(p, PARAM_NAMES.get(p, p))


def render_result(result, key):
    report = result["report"]
    st.markdown(report["answer"])
    if report.get("sources"):
        st.caption("Источник: " + "; ".join(report["sources"]))
    checks = []
    if result.get("dropped_findings"):
        checks.append(f"отброшено находок без подтверждения детекторами: {result['dropped_findings']}")
    if result.get("unverified_numbers"):
        checks.append("в ответе есть числа, которых нет в данных инструментов: "
                      + ", ".join(result["unverified_numbers"]) + " — проверьте их")
    if checks:
        st.warning("Проверка ответа: " + "; ".join(checks), icon="⚠️")

    for i, f in enumerate(report["findings"]):
        with st.container(border=True):
            color = CATEGORY_COLOR.get(f["category"], "gray")
            st.markdown(
                f"**{f['substation']} · {param_label(f['parameter'])}** · "
                f"{fmt_ts(f['start'])} — {fmt_ts(f['end'])}  \n"
                f":{color}-background[{CATEGORY_RU.get(f['category'], f['category'])}]"
            )
            st.markdown(f"{f['what']}")
            st.markdown(f"**Вероятная причина:** {f['hypothesis']}")
            st.caption(f"Норма {f['requirement_code']}: {f['requirement_source']}")
            try:
                st.altair_chart(event_chart(f["substation"], f["parameter"], f["start"], f["end"]),
                                width="stretch", key=f"chart-{key}-{i}")
            except Exception as exc:  # график — вспомогательный, ответ показываем в любом случае
                st.caption(f"График недоступен: {exc}")

    if report.get("recommendations"):
        st.markdown("**Рекомендации**")
        st.markdown("\n".join(f"- {r}" for r in report["recommendations"]))

    with st.expander(f"Ход работы агента · вызовов инструментов: {len(result['trace'])} · {result['seconds']} с"):
        for step in result["trace"]:
            st.markdown(f"`{step['tool']}({json.dumps(step['args'], ensure_ascii=False)})`")
            if period_label(step["result"]):
                st.caption(period_label(step["result"]))
            st.json(step["result"], expanded=False)


# ---------- боковая панель ----------
with st.sidebar:
    st.title("🌡️ ИТП-агент")
    st.caption(f"Диагностика тепловых пунктов на локальной LLM · модель `{LLM_MODEL}`")
    info = substations()
    st.markdown("**Тепловые пункты**")
    for s in info["substations"]:
        st.markdown(f"**{s['name']}** — {s['address']}")
    d = lambda ts: pd.Timestamp(ts).strftime("%d.%m.%Y")  # noqa: E731
    st.caption(f"Данные: {d(info['data_period']['start'])} — {d(info['data_period']['end'])} · "
               f"«сейчас» = {d(NOW)} {pd.Timestamp(NOW):%H:%M}")
    st.divider()
    st.markdown("**Примеры вопросов**")
    for q in EXAMPLES:
        if st.button(q, width="stretch"):
            st.session_state.pending = q
    if st.button("Очистить диалог", type="tertiary"):
        st.session_state.messages = []
    st.divider()
    st.caption("Синтетические данные: 5 ИТП, 11 внесённых аномалий. "
               "Нормы — демонстрационная таблица, требует проверки экспертом.")

tab_chat, tab_data = st.tabs(["Ассистент", "Данные"])
# Поле ввода — вне вкладок, чтобы быть закреплённым внизу страницы
typed = st.chat_input("Например: что не так с ИТП-3 за последнюю неделю?")

# ---------- чат ----------
with tab_chat:
    st.session_state.setdefault("messages", [])
    if not st.session_state.messages:
        st.info("Спросите о состоянии тепловых пунктов — агент сам выберет ИТП и период, запустит "
                "детекторы и объяснит, что случилось и какую норму это нарушает. Можно спросить и о самих "
                "нормах: «сколько часов может не быть горячей воды?» Ответ занимает 15–60 секунд.")

    for n, m in enumerate(st.session_state.messages):
        with st.chat_message(m["role"]):
            if m["role"] == "user":
                st.markdown(m["content"])
            else:
                render_result(m["result"], key=n)

    question = typed or st.session_state.pop("pending", None)
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.status("Агент работает…", expanded=True) as status:
                def on_event(kind, payload):
                    if kind == "tool_call":
                        args = ", ".join(f"{k}={v}" for k, v in payload["args"].items())
                        label = TOOL_RU.get(payload["name"], payload["name"])
                        status.write(f"{label} · `{args}`" if args else label)
                    elif kind == "tool_result" and "events_found" in payload["result"]:
                        status.write(f"→ {period_label(payload['result'])} · найдено событий: "
                                     f"{payload['result']['events_found']}")
                    elif kind == "retry":
                        status.write(f"↺ Инструмент вернул ошибку, прошу модель исправить вызов: {payload['error']}")
                    elif kind == "report":
                        status.write("Формирую отчёт")
                try:
                    result = ask(question, on_event=on_event)
                except Exception as exc:
                    status.update(label="Ошибка", state="error")
                    st.error(f"Агент не смог ответить: {exc}. Запущен ли Ollama?")
                    st.stop()
                status.update(label=f"Готово за {result['seconds']} с", state="complete", expanded=False)
            render_result(result, key=len(st.session_state.messages))
        st.session_state.messages.append({"role": "assistant", "result": result})

# ---------- обзор данных ----------
with tab_data:
    names = [s["name"] for s in info["substations"]]
    c1, c2, c3 = st.columns([1, 2, 3])
    sub = c1.selectbox("ИТП", names, index=2)
    lo, hi = pd.Timestamp(info["data_period"]["start"]).date(), pd.Timestamp(info["data_period"]["end"]).date()
    period = c2.date_input("Период", value=(pd.Timestamp("2026-02-15").date(), hi), min_value=lo, max_value=hi)
    params = c3.multiselect("Параметры", list(PARAM_NAMES), default=["t1", "t2"], format_func=PARAM_NAMES.get)
    if isinstance(period, tuple) and len(period) == 2 and params:
        st.altair_chart(overview_chart(sub, period[0], period[1], params), width="stretch")
        st.caption("Пунктир — плановое значение по температурному графику.")
