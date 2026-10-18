"""Cobertura real (fracción de intervalos del 95 % que atrapan la media verdadera) de cuatro métodos."""
import math
from statistics import NormalDist
import numpy as np
from scipy import stats

N01 = NormalDist(); R, B = 4000, 999
def intervalos(x, rng):
    n = len(x); m = x.mean(); s = x.std(ddof=1); se = s / math.sqrt(n)
    tb = x[rng.integers(0, n, (B, n))].mean(1)
    z0 = N01.inv_cdf(min(max((np.sum(tb < m) + np.sum(tb <= m)) / (2 * B), 1e-6), 1 - 1e-6))
    jk = (n * m - x) / (n - 1); u = jk.mean() - jk; a = np.sum(u**3) / (6 * np.sum(u**2) ** 1.5)
    za = -1.959964; q = [N01.cdf(z0 + (z0 + zz) / (1 - a * (z0 + zz))) for zz in (za, -za)]
    return {"1,96 con s": (m - 1.96 * se, m + 1.96 * se),
            "t": (m - stats.t.ppf(0.975, n - 1) * se, m + stats.t.ppf(0.975, n - 1) * se),
            "bootstrap percentil": tuple(np.quantile(tb, [0.025, 0.975])),
            "bootstrap BCa": tuple(np.quantile(tb, q))}

datos = {"normal (consumo, L/100 km)": (lambda r, n: r.normal(32, 4.5, n), 32.0),
         "lognormal σ = 1 (retrasos, min)": (lambda r, n: 7.3 * r.lognormal(0, 1, n), 7.3 * math.exp(0.5))}
for nombre, (gen, mu) in datos.items():
    print(nombre)
    for n in (4, 10, 30, 100):
        rng = np.random.default_rng(n); hits = {}
        for _ in range(R):
            for k, (lo, hi) in intervalos(gen(rng, n), rng).items():
                hits[k] = hits.get(k, 0) + (lo <= mu <= hi)
        print(f"  n = {n:3d}  " + "  ".join(f"{k} {v / R:.1%}" for k, v in hits.items()))
