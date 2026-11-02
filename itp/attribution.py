"""Сверка «показатель — число»: число в ответе должно относиться к тому показателю, о котором фраза.

Проверка непроверенных чисел (agent.unverified_numbers) ловит только числа, которых нет в данных.
Но 7B-модель переставляет настоящие числа между показателями: «тепло на градусо-сутки −23 %», хотя −23 %
— изменение сырого тепла, или «максимальная обратка 62,5 °C» — максимум посуточных средних вместо часового.
Здесь ответ режется на фразы; в каждой по словам определяется показатель, а числа фразы сверяются с числами
этого показателя из ответа инструмента. Ошибка — число принадлежит другому показателю, но не этому.
Фразы, где показатель не распознан или их несколько, не проверяются: ложная тревога хуже пропуска.
"""
import re

_NUM = re.compile(r"(?<![\w.,])-?\d+(?:[.,]\d+)?")

# compare_periods: ключ change → признаки фразы (все группы должны встретиться) и исключения
COMPARE = {
    "heat_per_degree_day": (["градусо|эффективн|поправк\\w* на погод"], []),
    "heat_gcal": (["тепл"], ["градусо", "эффективн", "поправк\\w* на погод"]),
    "hours_t1_out": (["подач", "час"], []),
    "hours_t2_out": (["обратк", "час"], []),
    "hours_dhw_out": (["гвс|горяч", "час"], []),
    "hours_missing": (["без данных|пропуск"], []),
    "t_outdoor_mean": (["наружн"], []),
    "t1_dev_pct": (["подач", "отклон|график"], ["час"]),
    "t2_dev_pct": (["обратк", "отклон|график"], ["час"]),
    "t_dhw_mean": (["гвс|горяч"], ["час"]),
    "events": (["событ"], []),
}
# query_sensors: параметр и статистика
PARAMS = {"t1": "подач", "t2": "обратк", "t_dhw": "гвс|горяч", "flow": "расход", "t_outdoor": "наружн",
          "p1": "давлен\\w* в подающ", "p2": "давлен\\w* в обратн"}
STATS = {"mean": "средн", "max": "максим|наибольш|пиков|самая высок", "min": "миним|наименьш|самая низк"}


def _norm(n):
    n = n.replace(",", ".").lstrip("-")
    return n.rstrip("0").rstrip(".") if "." in n else n


# Числа, которые не значения показателей: номера ИТП, коды параметров, годы, даты
_NOT_VALUES = re.compile(r"итп[\s\-№]*\d+|\bt[12]\b|\bp[12]\b|\b20\d\d\b|\b\d{1,2}\.\d{2}\b|"
                         r"\b\d{1,2}\s+(?:январ|феврал|март|декабр)\w*|\bс\s+\d+\s+по\s+\d+|\d+\s+(?:дн|сут|недел)\w*",
                         re.I)


def _nums(text, percent=None):
    """Числа фразы. percent=True — только проценты, False — только не проценты, None — все."""
    text = _NOT_VALUES.sub(" ", str(text))
    out = set()
    for m in _NUM.finditer(text):
        is_pct = text[m.end():m.end() + 2].lstrip().startswith("%")
        if percent is None or percent == is_pct:
            out.add(_norm(m.group()))
    return out


def _clauses(answer):
    # десятичная запятая («0,232») — не граница фразы: режем по «, » с пробелом, «.», «;», « а », « и »
    # и переносам строк (ответы-списки)
    return [c for c in re.split(r"(?<!\d)\.(?!\d)|;|,\s+|\s+[аи]\s+|\n", answer) if c.strip()]


def _last_before(patterns, clause, pos):
    """Какой из шаблонов встретился последним до позиции pos (или None)."""
    best, best_at = None, -1
    for key, pat in patterns.items():
        for m in re.finditer(pat, clause[:pos].lower()):
            if m.start() > best_at:
                best, best_at = key, m.start()
    return best


def _sensor_issues(answer, values):
    """Значения датчиков: число относится к ближайшим перед ним параметру и статистике («средняя подача 98,3,
    обратка 54,3»), без них — не проверяется («Подача: 94,1 °C» — неизвестно, какая статистика)."""
    issues = []
    for clause in _clauses(answer):
        text = _NOT_VALUES.sub(lambda m: " " * len(m.group()), clause)     # позиции сохраняются
        for m in _NUM.finditer(text):
            if text[m.end():m.end() + 2].lstrip().startswith("%"):
                continue
            p, st = _last_before(PARAMS, text, m.start()), _last_before(STATS, text, m.start())
            if not p or not st or (p, st) not in values:
                continue
            n = _norm(m.group())
            if n in values[(p, st)]:
                continue
            owners = [k for k, v in values.items() if k != (p, st) and n in v]
            if owners:
                issues.append({"clause": clause.strip(), "number": n, "said": (p, st), "actually": owners})
    return issues


def _has(clause, patterns, exclude=()):
    low = clause.lower()
    return all(re.search(p, low) for p in patterns) and not any(re.search(p, low) for p in exclude)


def _compare_values(result):
    out = {}
    for key, ch in (result.get("change") or {}).items():
        vals = {ch.get("from"), ch.get("to"), ch.get("diff"), ch.get("pct")} - {None}
        out[key] = {_norm(str(v)) for v in vals} | {_norm(f"{v:.1f}") for v in vals if isinstance(v, float)}
    return out


def _sensor_values(result):
    out = {}
    for p, s in (result.get("period_summary") or {}).items():
        for stat in STATS:
            if stat in s:
                out[(p, stat)] = {_norm(str(s[stat]))}
        # посуточные средние: «максимум» из них вместо часового максимума — типичная подмена
        daily = {_norm(str(d[p])) for d in result.get("daily", []) if d.get(p) is not None}
        if daily:
            out[(p, "посуточное среднее")] = daily
    return out


def misattributed(answer, trace):
    """[{фраза, число, сказано_про, на_самом_деле}] — числа, приписанные не тому показателю."""
    issues = []
    for step in trace:
        res, tool = step.get("result") or {}, step.get("tool")
        if tool == "query_sensors":
            issues += _sensor_issues(answer, _sensor_values(res))
            continue
        if tool != "compare_periods":
            continue
        values = _compare_values(res)
        matchers = {k: v for k, v in COMPARE.items() if k in values}
        for clause in _clauses(answer):
            said = [k for k, (pat, exc) in matchers.items() if _has(clause, pat, exc)]
            if len(said) != 1:
                continue
            said = said[0]
            for n in _nums(clause) - values[said]:
                owners = [k for k, v in values.items() if k != said and n in v]
                if owners:
                    issues.append({"clause": clause.strip(), "number": n, "said": said, "actually": owners})
    return issues
