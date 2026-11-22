"""El Transformer diminuto frente a una LSTM del tema 9 con un número de parámetros parecido, en la misma tarea
y con el mismo entrenamiento (100 épocas, Adam con calentamiento), con tres semillas."""
import json
import numpy as np
import torch
from datos import datos, clave
from modelo import entrenar, n_param, aciertos, Lectora

(X, L), (Xv, Lv), (Xt, Lt) = datos()
reg, tal = clave(Xt, Lt)
TRAMOS = [(1, 2), (3, 5), (6, 10), (11, 24)]                       # distancia hasta el último código
TRAMOS_D = [(10, 17), (18, 24), (25, 35)]                          # distancia hasta el depósito


def mide(m):
    with torch.no_grad():
        pred = m(torch.tensor(Xt)).argmax(-1).numpy()
    porTramo = [aciertos(pred, Xt, [q for q in tal if a <= q[2] <= b]) for a, b in TRAMOS]
    porTramoD = [aciertos(pred, Xt, [q for q in reg if a <= q[2] <= b]) for a, b in TRAMOS_D]
    return aciertos(pred, Xt, reg), aciertos(pred, Xt, tal), porTramo, porTramoD


res = {}
for nombre, crear in [("Transformer", None), ("LSTM", lambda: Lectora(16, 20))]:
    filas = []
    for seed in (0, 1, 2):
        m, curva, seg = entrenar(X, L, Xv, Lv, seed=seed, crear=crear)
        ad, ac, pt, pd = mide(m)
        filas.append((ad, ac, pt, curva, seg, pd))
        print(f"{nombre:11s} semilla {seed} · {n_param(m)} parámetros · {seg:3.0f} s\n  depósito {ad:6.1%} (por distancia"
              f" {' '.join(f'{x:.1%}' for x in pd)}) · último código {ac:6.1%} (por distancia {' '.join(f'{x:.1%}' for x in pt)})")
    res[nombre] = filas
print("distancias hasta el depósito:", ", ".join(f"{a}–{b} ({sum(a <= q[2] <= b for q in reg)} viajes)" for a, b in TRAMOS_D))
print("distancias hasta el último código:", ", ".join(f"{a}–{b} ({sum(a <= q[2] <= b for q in tal)} viajes)" for a, b in TRAMOS))
for nombre, filas in res.items():
    c = np.array([f[3] for f in filas]).mean(0)
    print(f"{nombre:11s} pérdida de validación media, épocas 5, 25, 50, 100: {c[4]:.3f} {c[24]:.3f} {c[49]:.3f} {c[99]:.3f}")
json.dump({k: {"dep": [f[0] for f in v], "cod": [f[1] for f in v], "tramos": [f[2] for f in v], "tramos_d": [f[5] for f in v], "seg": [f[4] for f in v],
               "curva": np.array([f[3] for f in v]).mean(0).round(4).tolist()} for k, v in res.items()},
          open("comparar.json", "w", encoding="utf-8"))
