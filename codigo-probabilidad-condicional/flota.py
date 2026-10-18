import numpy as np, pandas as pd
from sim import viajes
from cond import chi2

d = viajes()
print(d.groupby("lluvia").alerta.mean().round(4).to_dict(), round(d[d.alerta].lluvia.mean(), 4))
print(d.groupby(["lluvia", "novato"]).alerta.agg(["mean", "size"]).round(4))
# alerta y exceso: dependientes en la flota, independientes dentro de cada tipo de conductor
for nombre, g in [("todos", d), ("novatos", d[d.novato]), ("veteranos", d[~d.novato])]:
    T = pd.crosstab(g.alerta, g.exceso).values; e, gl, p = chi2(T)
    print(f"{nombre:9s} P(exceso)={g.exceso.mean():.4f}  P(exceso|alerta)={g[g.alerta].exceso.mean():.4f}  chi2={e:8.1f}  p={p:.2g}")
# cuánto se mueve una condicional estimada con pocos casos: 500 viajes, 1 000 veces
r = np.random.default_rng(1); est, nB = [], []
for _ in range(1000):
    m = d.iloc[r.integers(0, len(d), 500)]; b = m[m.lluvia & m.novato]
    est.append(b.alerta.mean()); nB.append(len(b))
print("P(alerta | lluvia, novato) con 500 viajes: media", round(np.mean(est), 3), " desv.", round(np.std(est), 3),
      " casos de media", np.mean(nB), " teórica", round(np.sqrt(0.39 * 0.61 / 30), 3))
