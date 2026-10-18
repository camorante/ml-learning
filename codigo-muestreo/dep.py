import numpy as np
from sim import poblacion
from muestreo import autocorr, n_efectivo, n_efectivo_ar1

# 1) AR(1) con varianza 1: Var(x̄) exacta = 1ᵀΣ1 / n², con Σ_ij = φ^|i−j|
n, phi = 30, 0.6
S = phi ** np.abs(np.subtract.outer(np.arange(n), np.arange(n)))
exacta = S.sum() / n**2
rng = np.random.default_rng(3)
R = 50_000
e = np.empty((R, n)); e[:, 0] = rng.normal(size=R)
for t in range(1, n):
    e[:, t] = phi * e[:, t - 1] + np.sqrt(1 - phi**2) * rng.normal(size=R)
print(f"AR(1) φ={phi}, n={n}:  Var(x̄) exacta {exacta:.5f}  simulada {e.mean(1).var():.4f}"
      f"  ingenua 1/n {1/n:.5f}")
print(f"  n_ef = {n_efectivo_ar1(n, phi):.2f}   aproximación n(1−φ)/(1+φ) = {n*(1-phi)/(1+phi):.2f}")

# 2) la flota: paradas consecutivas de un mismo camión
d, ruta, camion = poblacion()
ac = np.array([[autocorr(d[camion == c], k) for k in range(1, 30)] for c in range(60)]).mean(0)
print("autocorrelación media dentro de cada camión, retardos 1-5:", ac[:5].round(3))
print(f"n_ef de 30 paradas seguidas (solo autocorrelación): {n_efectivo(30, ac):.1f}")
# dentro del camión (quitando su media) frente a la fórmula
res = d - np.array([d[camion == c].mean() for c in camion])
sw = res.std()
M = []
for _ in range(20_000):
    c = rng.integers(60); s0 = rng.integers(0, 171)
    M.append(res[c*200 + s0: c*200 + s0 + 30].mean())
print(f"dentro de cada camión: sd(x̄_30) = {np.std(M):.2f}   σ_dentro/√30 = {sw/np.sqrt(30):.2f}"
      f"   con n_ef: {sw/np.sqrt(n_efectivo(30, ac)):.2f}")
