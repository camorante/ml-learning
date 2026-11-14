"""Coste real: gradiente completo por retropropagación frente a diferencias centrales (2 pasadas por parámetro),
en redes 2-m-m-1 con tanh sobre los 30 viajes estandarizados. Tiempos en este ordenador (NumPy, 1 hilo)."""
import os
os.environ["OMP_NUM_THREADS"] = os.environ["OPENBLAS_NUM_THREADS"] = os.environ["MKL_NUM_THREADS"] = "1"
import json
import time
import numpy as np
from datos import carga_consumo
from retro import Densa, MLP, diferencias_centrales

X, y = carga_consumo(); X = (X - X.mean(0)) / X.std(0)
rng = np.random.default_rng(0)
filas = []


def mejor(f, rep):
    t = []
    for _ in range(rep):
        t0 = time.perf_counter(); f(); t.append(time.perf_counter() - t0)
    return min(t)


print(f"{'m':>4} {'parámetros':>10} {'pasada (µs)':>11} {'retroprop. (µs)':>15} {'÷ pasada':>8} {'dif. centrales':>15} {'÷ retroprop.':>12}")
for m in (2, 8, 32, 128, 512, 2048):
    tam = [2, m, m, 1]
    red = MLP([Densa(rng.normal(0, 1 / np.sqrt(a), (b, a)), np.zeros(b), "tanh" if i < 2 else "identidad")
               for i, (a, b) in enumerate(zip(tam, tam[1:]))], "bce")
    P = len(red.vector())
    tp = mejor(lambda: red.perdida_media(X, y), 200 if m < 1000 else 20)
    tb = mejor(lambda: red.gradientes(X, y), 200 if m < 1000 else 20)
    if P <= 20000:
        tf = mejor(lambda: diferencias_centrales(red, X, y), 3); et = ""
    else:
        tf = 2 * P * tp; et = " (estimado)"
    fmt = lambda s: f"{s * 1e3:.1f} ms" if s < 1 else f"{s:.1f} s" if s < 120 else f"{s / 60:.0f} min" if s < 7200 else f"{s / 3600:.1f} h"
    print(f"{m:>4} {P:>10,} {tp * 1e6:>11.1f} {tb * 1e6:>15.1f} {tb / tp:>8.1f} {fmt(tf):>15}{et} {tf / tb:>12,.0f}")
    filas.append({"m": m, "P": P, "tp": tp, "tb": tb, "tf": tf, "est": bool(et)})
json.dump(filas, open("coste.json", "w", encoding="utf-8"))
