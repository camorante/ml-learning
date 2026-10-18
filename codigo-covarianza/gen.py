"""Genera ../embed.json (datos de la versión sencilla) e imprime las cifras que se citan en el texto,
calculadas sobre los mismos datos redondeados que ve la página."""
import json
import numpy as np
from scipy import stats
from sim import juguete, velocidad, atipico, confusion, flota, CARGA8, CONSUMO8, NOMBRES

R = lambda a, b: float(np.corrcoef(a, b)[0, 1])
rd = lambda a, d: [round(float(v), d) for v in a]
out = {}

J = juguete()
dec = dict(carga=(1, 1), ralenti=(1, 0), exper=(1, 2), dia=(0, 1))
out["toy"] = {}
for k, (a, b) in J.items():
    a, b = np.round(a, dec[k][0]), np.round(b, dec[k][1])
    out["toy"][k] = [rd(a, dec[k][0]), rd(b, dec[k][1])]
    c = np.cov(a, b)[0, 1]
    agree = np.mean((a - a.mean()) * (b - b.mean()) > 0)
    print(f"juguete {k:8s} r = {R(a, b):+.3f}  cov = {c:9.2f}  medias {a.mean():.2f} {b.mean():.2f}  cuadrantes de acuerdo {agree:.1%}")

v, c = velocidad(); v, c = np.round(v, 1), np.round(c, 1)
out["vel"] = [rd(v, 1), rd(c, 1)]
for lo, hi in [(10, 90), (10, 50), (50, 90), (30, 70)]:
    m = (v >= lo) & (v <= hi)
    print(f"velocidad {lo}-{hi}: n = {m.sum():3d}  r = {R(v[m], c[m]):+.3f}  spearman = {stats.spearmanr(v[m], c[m])[0]:+.3f}")

(a, b), (x, y) = atipico()
a, b, x, y = [np.round(t, 1) for t in (a, b, x, y)]
out["at"] = [[rd(a, 1), rd(b, 1)], [rd(x, 1), rd(y, 1)]]
OUT0 = [(72, 52), (27, 21)]          # posición inicial del atípico en cada escenario
for (p, q), (ox, oy), name in [((a, b), OUT0[0], "sin relación"), ((x, y), OUT0[1], "con relación")]:
    A, Bb = np.r_[p, ox], np.r_[q, oy]
    print(f"atípico {name}: sin él r = {R(p, q):+.3f} rho = {stats.spearmanr(p, q)[0]:+.3f};"
          f" con él en {ox},{oy}: r = {R(A, Bb):+.3f} rho = {stats.spearmanr(A, Bb)[0]:+.3f}")
out["at0"] = OUT0

e, ci, co = confusion(); e, co = np.round(e, 1), np.round(co, 1)
out["conf"] = [rd(e, 1), ci.tolist(), rd(co, 1)]
print(f"confusión: r total = {R(e, co):+.3f}, ciudad (n={ci.sum()}) r = {R(e[ci == 1], co[ci == 1]):+.3f},"
      f" autovía (n={(1 - ci).sum()}) r = {R(e[ci == 0], co[ci == 0]):+.3f}")
print(f"   edad media ciudad {e[ci == 1].mean():.1f}, autovía {e[ci == 0].mean():.1f}; consumo medio ciudad {co[ci == 1].mean():.1f}, autovía {co[ci == 0].mean():.1f}")
print(f"   camiones de más de 10 años: {np.mean(ci[e > 10]):.0%} en ciudad; de menos de 5: {np.mean(ci[e < 5]):.0%}")
b1 = np.polyfit(e, co, 1)[0]
print(f"   pendiente total {b1:.3f} L/100 km por año")

X = flota()
DX = [0, 0, 0, 1, 1, 1, 2, 0, 0]
X = np.column_stack([np.round(X[:, j], DX[j]) for j in range(X.shape[1])])
out["mat"] = [rd(X[:, j], DX[j]) for j in range(X.shape[1])]
out["names"] = NOMBRES
np.set_printoptions(precision=2, suppress=True, linewidth=160)
for n in (500, 15):
    C = np.corrcoef(X[:n].T)
    print(f"matriz n = {n}\n{C}")
    iu = np.triu_indices(9, 1)
    noise = [(i, j) for i, j in zip(*iu) if i >= 7 or j >= 7]
    mx = max(noise, key=lambda t: abs(C[t]))
    print(f"   máx |r| con una columna de azar: {C[mx]:+.2f} ({NOMBRES[mx[0]]} / {NOMBRES[mx[1]]})")

# 8 camiones (idea 1 y ejemplo resuelto)
dx, dy = CARGA8 - CARGA8.mean(), CONSUMO8 - CONSUMO8.mean()
print("8 camiones: medias", CARGA8.mean(), CONSUMO8.mean(), " productos", (dx * dy).tolist(), " suma", (dx * dy).sum(),
      " cov", round(np.cov(CARGA8, CONSUMO8)[0, 1], 4), " r", round(R(CARGA8, CONSUMO8), 4))
out["ex8"] = [CARGA8.tolist(), CONSUMO8.tolist()]
json.dump(out, open("../embed.json", "w"), separators=(",", ":"), ensure_ascii=False)
print("embed.json", len(json.dumps(out, separators=(",", ":"))), "bytes")
