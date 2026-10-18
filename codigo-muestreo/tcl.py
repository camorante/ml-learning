import numpy as np
from scipy import stats
from sim import poblacion
from muestreo import berry_esseen

d = poblacion()[0]
rng = np.random.default_rng(2)
casos = {                                       # (generador de una matriz R×n, μ, σ, E|X−μ|³/σ³, asimetría γ)
    "paradas": lambda R, n: d[rng.integers(0, len(d), (R, n))],
    "Poisson(0.8)": lambda R, n: rng.poisson(0.8, (R, n)),
}
mom = {}
m, s = d.mean(), d.std()
mom["paradas"] = (m, s, np.mean(np.abs(d - m) ** 3) / s**3, stats.skew(d))
k = np.arange(60); p = stats.poisson.pmf(k, 0.8)
mom["Poisson(0.8)"] = (0.8, np.sqrt(0.8), np.sum(p * np.abs(k - 0.8) ** 3) / 0.8**1.5, 1 / np.sqrt(0.8))

R = 20_000
for name, gen in casos.items():
    mu, sg, rho3, g = mom[name]
    print(f"{name}: γ = {g:.2f}   E|X−μ|³/σ³ = {rho3:.2f}")
    for n in [1, 5, 30, 100, 400]:
        xb = gen(R, n).mean(axis=1)
        z = (xb - mu) / (sg / np.sqrt(n))
        ks = stats.kstest(z, "norm").statistic
        print(f"  n = {n:3d}  asimetría de x̄ = {stats.skew(xb):5.2f} (γ/√n = {g/np.sqrt(n):4.2f})"
              f"  KS = {ks:.3f}  Berry–Esseen ≤ {berry_esseen(rho3, n):.3f}")

# colas pesadas: Pareto con α = 1.5 (varianza infinita): la dispersión de x̄ NO baja como 1/√n
a, xm = 1.5, 5.0
for n in [30, 300, 3000]:
    xb = (xm * (1 - rng.random((4000, n))) ** (-1 / a)).mean(axis=1)
    q = np.percentile(xb, [5, 50, 95])
    print(f"Pareto α=1.5  n = {n:4d}  percentiles 5/50/95 de x̄: {q.round(2)}  (μ = 15)")
