import numpy as np, pandas as pd
from sim import viajes
from outl import z_clasico, z_robusto, vallas_iqr, huber_irls, ols, isolation_forest, Q75

d = viajes(); x, y = d.carga_t.values, d.consumo.values
lo, hi = vallas_iqr(y)
b_h, w = huber_irls(x, y); r = y - (b_h[0] + b_h[1] * x); s = np.median(np.abs(r)) / Q75
ifo = isolation_forest(np.c_[x, y], 200, 256, 0)
metodos = {
    "z clásico > 3 (consumo)": np.abs(z_clasico(y)) > 3,
    "z robusto > 3.5 (consumo)": np.abs(z_robusto(y)) > 3.5,
    "IQR 1.5 (consumo)": (y < lo) | (y > hi),
    "residuo robusto > 3.5": np.abs(r / s) > 3.5,
    "Isolation Forest (4 %)": ifo >= np.quantile(ifo, 0.96),
}
filas = []
for k, m in metodos.items():
    filas.append(dict(metodo=k, marcados=int(m.sum()), errores=f"{int((m & d.es_error).sum())}/12",
                      raros=f"{int((m & d.es_raro).sum())}/31", normales=int((m & (d.tipo == 'normal')).sum())))
print(pd.DataFrame(filas).to_string(index=False))
norm = d.tipo == "normal"
print("pendiente real 0.9 | MCO todos %.3f | MCO sin errores %.3f | Huber %.3f" % (ols(x, y)[1], ols(x[~d.es_error], y[~d.es_error])[1], b_h[1]))
print("consumo medio: todos %.2f | sin errores %.2f | mediana %.2f" % (y.mean(), y[~d.es_error].mean(), np.median(y)))
for t, g in d[d.es_error | d.es_raro].groupby("tipo"):
    i = g.index; print(t, "z rob consumo", np.round(np.abs(z_robusto(y))[i].mean(), 1), "residuo rob", np.round(np.abs(r / s)[i].mean(), 1))
