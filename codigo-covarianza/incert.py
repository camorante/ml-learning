import math
import numpy as np
from cor import pearson, p_valor_r, ic_fisher, matriz_corr
from sim import CARGA8, CONSUMO8

r = pearson(CARGA8, CONSUMO8); n = 8
t = r * math.sqrt((n - 2) / (1 - r * r))
print(f"8 camiones: r = {r:.4f}  t = {t:.3f}  p = {p_valor_r(r, n):.5f}  IC 95 % Fisher = [{ic_fisher(r, n)[0]:.3f}, {ic_fisher(r, n)[1]:.3f}]")
# r crítico (p = 0,05, dos colas) según n, y anchura del intervalo para r = 0,5
for m in (8, 15, 30, 100, 500):
    lo, hi = 0.0, 0.999
    for _ in range(60):
        mid = (lo + hi) / 2; lo, hi = (mid, hi) if p_valor_r(mid, m) > 0.05 else (lo, mid)
    a, b = ic_fisher(0.5, m)
    print(f"n = {m:3d}: |r| > {hi:.3f} para p < 0,05;  r = 0,5 → IC 95 % [{a:.2f}, {b:.2f}]")
# muchas parejas de columnas sin ninguna relación: 9 columnas (36 parejas), 15 camiones
rng = np.random.default_rng(3); mx = []; alguna = 0
for _ in range(20_000):
    R = matriz_corr(rng.normal(size=(15, 9))); v = np.abs(R[np.triu_indices(9, 1)])
    mx.append(v.max()); alguna += np.any(v > 0.514)
mx = np.array(mx)
print(f"9 columnas de ruido, 15 filas: |r| máximo mediano {np.median(mx):.2f}; en el {np.mean(mx > 0.5):.0%} de las tablas alguna |r| > 0,5;"
      f" alguna 'significativa' (p < 0,05) en el {alguna / 20_000:.0%}; esperado sin dependencia 1 − 0,95^36 = {1 - 0.95**36:.0%}")
