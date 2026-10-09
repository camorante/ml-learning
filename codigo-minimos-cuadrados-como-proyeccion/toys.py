"""Datos de los juguetes de la versión sencilla (se guardan en ../embed.json) y comprobaciones."""
import json, os
import numpy as np
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ---------- juguete: 10 viajes, consumo (L/100 km) frente a carga (t) ----------
carga = np.array([2.0, 4.5, 6.0, 8.5, 10.0, 12.5, 15.0, 17.5, 20.0, 23.5])
rng = np.random.default_rng(11)
cons = np.round(25 + 0.5 * carga + rng.normal(0, 1.4, carga.size), 1)
X = np.c_[np.ones_like(carga), carga]
b = np.linalg.lstsq(X, cons, rcond=None)[0]
r = cons - X @ b
print("toy consumo", cons.tolist())
print("toy b", b.round(4), "SSE", round(float(r @ r), 3), "sum r", round(float(r.sum()), 10), "sum r*x", round(float(r @ carga), 10))
print("toy SST", round(float(((cons - cons.mean()) ** 2).sum()), 3), "mean", cons.mean())
# recta de partida del juguete: horizontal en 28
r0 = cons - 28
print("start SSE (28 plano)", round(float(r0 @ r0), 2))

# ---------- idea 2 y ejemplo: tres viajes ----------
x3 = np.array([4.0, 12.0, 20.0]); y3 = np.array([26.0, 33.0, 34.0])
X3 = np.c_[np.ones(3), x3]; b3 = np.linalg.solve(X3.T @ X3, X3.T @ y3)
print("3 viajes b", b3, "res", y3 - X3 @ b3, "media", y3.mean())

# ---------- idea 4: columnas de ruido ----------
rng = np.random.default_rng(5)
ntr, nte, K = 30, 200, 26
def datos(n):
    c = rng.uniform(2, 24, n); p = rng.uniform(-1, 3, n)          # carga (t) y pendiente media de la ruta (%)
    y = 24 + 0.5 * c + 2.0 * p + rng.normal(0, 1.5, n)
    return c, p, y, rng.normal(0, 1, (n, K))
c1, p1, y1, Z1 = datos(ntr); c2, p2, y2, Z2 = datos(nte)
tr, te = [], []
for k in range(K + 1):
    A1 = np.c_[np.ones(ntr), c1, p1, Z1[:, :k]]; A2 = np.c_[np.ones(nte), c2, p2, Z2[:, :k]]
    bb = np.linalg.lstsq(A1, y1, rcond=None)[0]
    tr.append(float(np.sqrt(np.mean((y1 - A1 @ bb) ** 2)))); te.append(float(np.sqrt(np.mean((y2 - A2 @ bb) ** 2))))
print("idea4 train", np.round(tr, 2).tolist())
print("idea4 test ", np.round(te, 2).tolist())
# solo carga (sin pendiente) para comparar
A1 = np.c_[np.ones(ntr), c1]; A2 = np.c_[np.ones(nte), c2]; bb = np.linalg.lstsq(A1, y1, rcond=None)[0]
tr0 = float(np.sqrt(np.mean((y1 - A1 @ bb) ** 2))); te0 = float(np.sqrt(np.mean((y2 - A2 @ bb) ** 2)))
print("solo carga", round(tr0, 2), round(te0, 2))

json.dump({"x": carga.tolist(), "y": cons.tolist(), "b": b.tolist(),
           "x3": x3.tolist(), "y3": y3.tolist(),
           "tr": [round(v, 4) for v in tr], "te": [round(v, 4) for v in te], "tr0": round(tr0, 4), "te0": round(te0, 4)},
          open("../embed.json", "w"))
print("ok")
