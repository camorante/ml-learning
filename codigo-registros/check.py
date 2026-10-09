import numpy as np, pandas as pd, itertools
from scipy import stats
from sklearn.metrics.pairwise import haversine_distances
from tele import haversine, quitar_duplicados, theil_sen, tramos_congelados, marcar_saltos, estimar_deriva
from sim import registros
rng = np.random.default_rng(0)
# haversine frente a scikit-learn
P = np.c_[rng.uniform(-80, 80, 500), rng.uniform(-180, 180, 500)]; Q = np.c_[rng.uniform(-80, 80, 500), rng.uniform(-180, 180, 500)]
ref = np.array([haversine_distances(np.radians([p]), np.radians([q]))[0, 0] for p, q in zip(P, Q)]) * 6371.0088
print("haversine vs sklearn, dif máx (km):", float(np.abs(haversine(P[:, 0], P[:, 1], Q[:, 0], Q[:, 1]) - ref).max()))
# Theil-Sen frente a SciPy
e = 0
for t in range(200):
    n = int(rng.integers(5, 60)); x = rng.uniform(0, 10, n); y = 2 * x + rng.standard_t(2, n)
    e = max(e, abs(theil_sen(x, y)[0] - stats.theilslopes(y, x)[0]))
print("Theil-Sen vs scipy, dif máx:", e)
# duplicados frente a pandas
m, r = registros()
a = quitar_duplicados(m); b = m.sort_values("t_servidor").drop_duplicates(subset=["t_equipo", "lat", "lon", "vel", "odo"])
print("duplicados: mismos índices que pandas:", a.index.equals(b.index), "| reenvíos encontrados", len(m) - len(a), "de", int(m.reenvio.sum()),
      "| ninguno marcado de más:", int((~a.reenvio).sum()) == len(a))
# tramos congelados frente a itertools.groupby
t = np.arange(400) * 60.; v = np.repeat(np.arange(80) % 7, rng.integers(1, 40, 80))[:400]; odo = np.cumsum(rng.uniform(0, 2, 400))
mine = tramos_congelados(t, v, odo, 20, 5)
ref = []; i = 0
for k, g in itertools.groupby(range(len(v)), key=lambda j: v[j]):
    g = list(g); d = (t[g[-1]] - t[g[0]]) / 60
    if d >= 20 and odo[g[-1]] - odo[g[0]] >= 5: ref.append((t[g[0]], t[g[-1]]))
print("tramos congelados = groupby:", [(a_, b_) for a_, b_, *_ in mine] == ref, len(ref))
# saltos y deriva en el día simulado
c = marcar_saltos(a.sort_values("t_equipo"))
print("saltos: detectados", int(c.salto_detectado.sum()), "verdaderos", int(c.salto.sum()), "aciertos", int((c.salto & c.salto_detectado).sum()))
print("deriva estimada (s/h):", round(estimar_deriva(a.t_equipo, a.t_servidor)[0], 2), "(real 40)")
