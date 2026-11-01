"""Веб-интерфейс ИТП-агента.

    streamlit run app.py
"""
import json

import pandas as pd
import streamlit as st

from itp import history
from itp.agent import CATEGORY_RU, ask
from itp.charts import CORRIDOR, compare_bars, event_chart, overview_chart, period_chart, period_frame, PERIOD_COLORS
from itp.compare import METRICS, compare_periods
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
    "compare_periods": "Сравниваю периоды",
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
    if result.get("asked") and result.get("question") != result["asked"]:
        st.caption(f"Понял вопрос как: «{result['question']}»")
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

    suspicions = [x for s in result["trace"] if s["tool"] == "detect_anomalies"
                  for x in (s.get("result") or {}).get("suspicions", [])]
    if suspicions:
        st.markdown("**Требует проверки** — статистика, без нарушения норм")
        st.markdown("\n".join(f"- {x.get('substation', '')} {fmt_ts(x['start'])} — {fmt_ts(x['end'])}: {x['summary']}"
                              for x in suspicions))

    if report.get("recommendations"):
        st.markdown("**Рекомендации**")
        st.markdown("\n".join(f"- {r}" for r in report["recommendations"]))

    with st.expander(f"Ход работы агента · вызовов инструментов: {len(result['trace'])} · {result['seconds']} с"):
        for step in result["trace"]:
            st.markdown(f"`{step['tool']}({json.dumps(step['args'], ensure_ascii=False)})`")
            if period_label(step["result"]):
                st.caption(period_label(step["result"]))
            st.json(step["result"], expanded=False)


def render_feedback(m):
    """👍/👎 под ответом; оценка сразу пишется в БД."""
    def save():
        value = st.session_state[f"fb-{m['id']}"]
        m["feedback"] = value
        history.set_feedback(m["id"], value)
    st.feedback("thumbs", key=f"fb-{m['id']}", default=m.get("feedback"), on_change=save)


def open_chat(chat_id):
    """Диалог из адреса страницы (?chat=…): переживает обновление страницы."""
    st.session_state.chat_id, st.session_state.messages = chat_id, []
    if not chat_id:
        return
    try:
        for m in history.load(chat_id):
            st.session_state.messages += [{"role": "user", "content": m["question"]},
                                          {"role": "assistant", "result": m["result"], "id": m["id"],
                                           "feedback": m["feedback"]}]
    except Exception as exc:          # без истории чат всё равно работает
        st.warning(f"Не удалось загрузить историю: {exc}")


if "chat_id" not in st.session_state or st.session_state.chat_id != st.query_params.get("chat"):
    open_chat(st.query_params.get("chat"))


def switch_chat(chat_id):
    if chat_id:
        st.query_params["chat"] = chat_id
    else:
        st.query_params.clear()
    open_chat(chat_id)


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
    st.divider()
    st.markdown("**Диалоги**")
    st.button("＋ Новый диалог", on_click=switch_chat, args=(None,), width="stretch")
    try:
        for c in history.chats():
            title = c["title"] if len(c["title"]) <= 42 else c["title"][:40] + "…"
            current = c["chat_id"] == st.session_state.chat_id
            st.button(f"{title} · {c['count']}", key=f"chat-{c['chat_id']}", type="secondary" if current else "tertiary",
                      on_click=switch_chat, args=(c["chat_id"],), width="stretch",
                      help=f"Последний вопрос: {c['updated']:%d.%m %H:%M}")
    except Exception as exc:
        st.caption(f"История недоступна: {exc}")
    st.divider()
    st.caption("Синтетические данные: 5 ИТП, 11 внесённых аномалий. Нормы — действующие документы "
               "(ПТЭ № 511, ПП № 354, ПП № 1034, СанПиН 2.1.3684-21); таблица порогов требует проверки экспертом.")

tab_chat, tab_data, tab_cmp = st.tabs(["Ассистент", "Данные", "Сравнение периодов"])
# Поле ввода — вне вкладок, чтобы быть закреплённым внизу страницы
typed = st.chat_input("Например: что не так с ИТП-3 за последнюю неделю?")

# ---------- чат ----------
with tab_chat:
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
                if m.get("id"):
                    render_feedback(m)

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
                    elif kind == "rewrite":
                        status.write(f"Уточнение с учётом диалога: «{payload['question']}»")
                    elif kind == "retry":
                        status.write(f"↺ Инструмент вернул ошибку, прошу модель исправить вызов: {payload['error']}")
                    elif kind == "report":
                        status.write("Формирую отчёт")
                # Предыдущие пары диалога (вопрос без ответа — после сбоя агента — пропускаем)
                past = st.session_state.messages[:-1]
                history_turns = history.as_turns((q["content"], a["result"]) for q, a in zip(past, past[1:])
                                                 if q["role"] == "user" and a["role"] == "assistant")
                try:
                    result = ask(question, on_event=on_event, history=history_turns)
                except Exception as exc:
                    status.update(label="Ошибка", state="error")
                    st.error(f"Агент не смог ответить: {exc}. Запущен ли Ollama?")
                    st.stop()
                status.update(label=f"Готово за {result['seconds']} с", state="complete", expanded=False)
            render_result(result, key=len(st.session_state.messages))
            answer = {"role": "assistant", "result": result, "id": None, "feedback": None}
            try:
                if not st.session_state.chat_id:          # первый вопрос — диалог появляется в истории
                    st.session_state.chat_id = history.new_chat_id()
                    st.query_params["chat"] = st.session_state.chat_id
                answer["id"] = history.save(st.session_state.chat_id, question, result, LLM_MODEL)
                render_feedback(answer)
            except Exception as exc:
                st.warning(f"Ответ не сохранён в историю: {exc}")
        st.session_state.messages.append(answer)

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

# ---------- сравнение периодов ----------
with tab_cmp:
    names = [s["name"] for s in info["substations"]]
    lo, hi = pd.Timestamp(info["data_period"]["start"]).date(), pd.Timestamp(info["data_period"]["end"]).date()
    c1, c2, c3, c4 = st.columns([1, 1.4, 2, 2])
    sub = c1.selectbox("ИТП", names, index=2, key="cmp_sub")
    param = c2.selectbox("Параметр", ["t1", "t2", "t_dhw", "flow"], index=1, format_func=PARAM_NAMES.get,
                         key="cmp_param")
    pa = c3.date_input("Первый период", value=(lo, pd.Timestamp("2026-01-31").date()), min_value=lo, max_value=hi,
                       key="cmp_a")
    pb = c4.date_input("Второй период", value=(pd.Timestamp("2026-02-01").date(), hi), min_value=lo, max_value=hi,
                       key="cmp_b")
    if all(isinstance(p, tuple) and len(p) == 2 for p in (pa, pb)):
        label = lambda p: f"{p[0]:%d.%m} — {p[1]:%d.%m}"  # noqa: E731
        frames = [period_frame(sub, p[0], p[1], param) for p in (pa, pb)]
        cols = [c for c in ("value", "lo", "hi") if c in frames[0]]
        vals = pd.concat([f[cols] for f in frames]).stack()
        pad = (vals.max() - vals.min()) * 0.05
        domain = [float(vals.min() - pad), float(vals.max() + pad)]      # одна шкала — графики сравнимы
        g1, g2 = st.columns(2)
        for col, frame, p, color in zip((g1, g2), frames, (pa, pb), PERIOD_COLORS):
            col.altair_chart(period_chart(frame, label(p), color, domain), width="stretch")
        if param in CORRIDOR:
            st.caption(f"Зелёная полоса — коридор нормы ({CORRIDOR[param]}), пунктир — температурный график.")

        cmp = compare_periods(sub, *(f"{d:%Y-%m-%d}" for d in (*pa, *pb)))
        a_lbl, b_lbl = label(pa), label(pb)
        bars = [{"metric": METRICS[k][0].replace("Часов: ", ""), "period": lbl, "value": cmp[pk][k]}
                for k in ("hours_t1_out", "hours_t2_out", "hours_dhw_out", "hours_missing")
                for pk, lbl in (("period_1", a_lbl), ("period_2", b_lbl))]
        types = sorted(set(cmp["period_1"]["events_by_type"]) | set(cmp["period_2"]["events_by_type"]))
        ev_bars = [{"metric": t, "period": lbl, "value": cmp[pk]["events_by_type"].get(t, 0)}
                   for t in types for pk, lbl in (("period_1", a_lbl), ("period_2", b_lbl))]
        b1, b2 = st.columns(2)
        b1.markdown("**Часов вне нормы**")
        b1.altair_chart(compare_bars(bars), width="stretch")
        b2.markdown("**События детекторов по типам**")
        if ev_bars:
            b2.altair_chart(compare_bars(ev_bars), width="stretch")
        else:
            b2.caption("В обоих периодах событий нет.")

        def fmt(v, d):
            if v is None:
                return "—"
            v = 0.0 if abs(v) < 0.5 * 10 ** -d else v          # без «−0,0»
            return f"{v:,.{d}f}".replace(",", " ").replace(".", ",")
        rows = []
        for k, (title, unit, digits) in METRICS.items():
            ch = cmp["change"].get(k, {})
            diff = ch.get("diff")
            rows.append({"Показатель": title + (f", {unit}" if unit else ""), a_lbl: fmt(cmp["period_1"][k], digits),
                         b_lbl: fmt(cmp["period_2"][k], digits),
                         "Изменение": ("" if diff is None else ("+" if diff > 0 else "") + fmt(diff, digits))
                                      + (f" ({ch['pct']:+d} %)" if "pct" in ch else "")})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.caption("Тепло за период зависит от погоды; сравнивайте тепло на градусо-сутки "
                   f"(градусо-сутки: {fmt(cmp['period_1']['degree_days'], 0)} → {fmt(cmp['period_2']['degree_days'], 0)}).")
