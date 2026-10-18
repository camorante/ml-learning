import numpy as np
from sim import poblacion

d, ruta, camion = poblacion()                 # las 12 000 paradas del año (minutos)
N = len(d); mu = d.mean(); sigma = d.std()    # parámetros: dividen entre N
print(f"N = {N}  μ = {mu:.3f}  σ = {sigma:.3f}")

rng = np.random.default_rng(1)
R = 100_000                                   # muestras repetidas
for n in [4, 16, 64]:
    idx = rng.integers(0, N, size=(R, n))     # con reemplazo: datos i.i.d.
    xbar = d[idx].mean(axis=1)
    print(f"n = {n:2d}  E[x̄] ≈ {xbar.mean():.3f}  sd(x̄) ≈ {xbar.std():.3f}"
          f"  σ/√n = {sigma/np.sqrt(n):.3f}")

# sin reemplazo y n grande frente a N: hace falta la corrección por población finita
n = 6000
xb = np.array([d[rng.permutation(N)[:n]].mean() for _ in range(4000)])
fpc = np.sqrt((N - n) / (N - 1))
print(f"n = {n} sin reemplazo: sd(x̄) ≈ {xb.std():.4f}  σ/√n = {sigma/np.sqrt(n):.4f}"
      f"  ·fpc = {sigma/np.sqrt(n)*fpc:.4f}")

# una sola muestra: s/√n estima el error estándar (y también baila)
x = d[rng.integers(0, N, 16)]
print(f"una muestra de 16: x̄ = {x.mean():.2f}  s = {x.std(ddof=1):.2f}  s/√n = {x.std(ddof=1)/4:.2f}")
