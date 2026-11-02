"""Сверка «показатель — число» (itp/attribution.py): случаи из реальных ответов модели."""
from itp.attribution import misattributed

COMPARE = [{"tool": "compare_periods", "result": {"change": {
    "heat_gcal": {"from": 263.7, "to": 203.7, "diff": -60.0, "pct": -23},
    "heat_per_degree_day": {"from": 0.232, "to": 0.223, "diff": -0.009, "pct": -4},
    "hours_t2_out": {"from": 0, "to": 175, "diff": 175},
    "t_outdoor_mean": {"from": -16.7, "to": -12.6, "diff": 4.1},
}}}]
SENSORS = [{"tool": "query_sensors", "result": {
    "period_summary": {"t1": {"mean": 98.3, "min": 80.1, "max": 112.0}, "t2": {"mean": 53.8, "min": 44.0, "max": 65.8}},
    "daily": [{"t1": 99.0, "t2": 62.5}, {"t1": 97.5, "t2": 50.1}]}}]


def test_raw_heat_percent_called_per_degree_day():
    issues = misattributed("Тепло на градусо-сутки снизилось на 23%.", COMPARE)
    assert [(i["number"], i["said"], i["actually"]) for i in issues] == [("23", "heat_per_degree_day", ["heat_gcal"])]


def test_correct_comparison_answer_passes():
    answer = ("Обратка выше графика более 5 %: 0 → 175 ч. Тепло на градусо-сутки 0,232 → 0,223, −4 %; "
              "сырое тепло −23 % — из-за погоды (наружная −16,7 → −12,6 °C). Эффективность использования тепла "
              "снизилась на 4%. На ИТП-3 в 2026 году.")
    assert misattributed(answer, COMPARE) == []


def test_daily_mean_called_maximum():
    issues = misattributed("Максимальная температура обратки на ИТП-3 в феврале составила 62,5 °C.", SENSORS)
    assert [(i["number"], i["said"]) for i in issues] == [("62.5", ("t2", "max"))]


def test_enumeration_binds_number_to_nearest_parameter():
    answer = "Средняя температура подачи (t1) 98,3 °C и температура обратки (t2) в среднем 53,8 °C, максимум 65,8 °C."
    assert misattributed(answer, SENSORS) == []


def test_unknown_statistic_and_percent_not_checked():
    # «Подача: 99 °C» — непонятно, какая статистика; проценты при значениях датчиков — отклонения, не значения
    assert misattributed("- Подача: 99 °C\n- Средняя подача на 1,5% выше графика", SENSORS) == []
