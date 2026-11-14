"""Datos de los juguetes de la versión sencilla (embed.json) y cifras que citan los textos."""
import json, os, math
import numpy as np
os.chdir(os.path.dirname(os.path.abspath(__file__)))
from datos import tramos, verdad
from codos import base, ajustar

v, c = tramos()
redes = json.load(open("red_toy.json"))
json.dump(dict(v=v.tolist(), c=c.tolist(), redes=redes), open("../embed.json", "w"), separators=(",", ":"))
print("embed.json:", len(v), "tramos,", len(redes), "redes")
for k, r in redes.items():
    print(f"  {k:11s} error {r['rmse']:.3f}  nuevos {r['rmse_n']:.3f}  números {r['n']}")
print("ruido de los datos:", round(float(np.sqrt(np.mean((verdad(v) - c) ** 2))), 3))

# idea 2: valores por defecto
w1, b1, w2, b2 = -0.8, 56, 0.5, 30
print(f"idea 2: sin activación {w2 * w1:.3f}·v + {w2 * b1 + b2:.1f};  codo con ReLU en {-b1 / w1:.1f} km/h")

# idea 3: posiciones predefinidas
for codos in ([], [60], [40, 73], [25, 50, 80]):
    k = ajustar(v, c, codos); pred = base(v, codos) @ k
    g = np.arange(10, 115.01, 0.5); pg = base(g, codos) @ k
    orden = sorted(range(len(codos)), key=lambda j: codos[j]); s = [k[1] / 10]
    for j in orden: s.append(s[-1] + k[j + 2] / 10)
    print(f"idea 3 {str(codos):12s} error {np.sqrt(np.mean((pred - c) ** 2)):.3f}  coef {np.round(k, 2)}"
          f"  mínimo {g[np.argmin(pg)]:.1f} km/h {pg.min():.2f}  pendientes por 10 km/h {np.round(s, 2)}")

# idea 5: lo que llega a la primera capa
sig = lambda z: 1 / (1 + math.exp(-z))
for z in (0, 1, 2):
    ds, dt = sig(z) * (1 - sig(z)), 1 - math.tanh(z) ** 2
    print(f"idea 5 z={z}: sigmoide {ds:.4f} → 10 capas {ds ** 9:.2e};  tanh {dt:.4f} → 10 capas {dt ** 9:.2e}")

# idea 6: sesgo a partir del cual la neurona z = 0,1·v + b muere en los 60 tramos
print(f"idea 6: muere con sesgo <= {-0.1 * v.max():.2f} (tramo más rápido {v.max()} km/h);"
      f" con −5 encendida en {int(np.sum(0.1 * v - 5 > 0))} tramos")

# datos del widget de la técnica (tembed.json)
vn, cn = tramos(200, seed=7)
json.dump(dict(v=v.tolist(), c=c.tolist(), vn=vn.tolist(), cn=cn.tolist()), open("../tembed.json", "w"), separators=(",", ":"))
