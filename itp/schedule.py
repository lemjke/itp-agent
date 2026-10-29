"""Упрощённый температурный график теплосети (качественное регулирование)."""
import numpy as np

from .config import T_IN_DESIGN, T_OUT_DESIGN, T1_DESIGN, T2_DESIGN, T1_BREAK


def load_ratio(t_out):
    """Относительная тепловая нагрузка отопления: 0 при +20 °C, 1 при расчётной температуре."""
    phi = (T_IN_DESIGN - np.asarray(t_out, dtype=float)) / (T_IN_DESIGN - T_OUT_DESIGN)
    return np.clip(phi, 0.0, 1.0)


def graph(t_out):
    """Плановые температуры подачи и обратки (t1, t2) по наружной температуре."""
    k = load_ratio(t_out) ** 0.8
    t1 = T_IN_DESIGN + (T1_DESIGN - T_IN_DESIGN) * k
    t2 = T_IN_DESIGN + (T2_DESIGN - T_IN_DESIGN) * k
    t1 = np.maximum(t1, T1_BREAK)   # нижний излом под нужды ГВС
    t2 = np.maximum(t2, 40.0)
    return t1, t2
