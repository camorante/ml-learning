import numpy as np
from scipy import stats
mes = np.arange(1, 11); y = np.array([30, 31, 31, 32, 33, 33, 34, 35, 35, 36.])
futuro, y_fut = np.array([11, 12, 13]), np.array([37, 37, 38.])

def vecinos_en_el_tiempo(m_tr, y_tr, m):
    """Predice la media de los 2 meses de entrenamiento más cercanos (empate: gana el anterior)."""
    orden = np.lexsort((m_tr, np.abs(m_tr - m)))
    return y_tr[orden[:2]].mean()

def evalua(test):
    tr = ~np.isin(mes, test)
    p = np.array([vecinos_en_el_tiempo(mes[tr], y[tr], m) for m in test])
    return p, np.abs(p - y[np.isin(mes, test)]).mean()

print("azar   ", *evalua(np.array([3, 6, 9])))      # [31.5 33.5 35.5] 0.5
print("tiempo ", *evalua(np.array([8, 9, 10])))     # [33.5 33.5 33.5] 1.833
p = np.array([vecinos_en_el_tiempo(mes, y, m) for m in futuro]); e = np.abs(p - y_fut)
print("produc.", p, e.mean())                       # [35.5 35.5 35.5] 1.833
se = e.std(ddof=1) / np.sqrt(3)
print(round(se, 3), round(1.96 * se, 3), round(stats.t.ppf(0.975, 2) * se, 3))
