import numpy as np
from scipy import stats
from sim import ALERTAS_A as x
from mv import ll_poisson, se_poisson, errores_estandar, intervalo_perfil, ll_binom, se_binom

n, lh = len(x), x.mean()
print(f"I(λ̂) = n/λ̂ = {n / lh:.4f}   ee = 1/√I = {se_poisson(lh, n):.4f}   "
      f"hessiano numérico: {errores_estandar(lambda t: -ll_poisson(t[0], x), [lh])[0]:.4f}")
sim = np.random.default_rng(1).poisson(2.5, (200_000, n)).mean(axis=1)
print(f"desviación de λ̂ en 200 000 simulaciones con λ = 2,5: {sim.std():.4f}")
w = (lh - 1.96 * se_poisson(lh, n), lh + 1.96 * se_poisson(lh, n))
p = intervalo_perfil(lambda l: ll_poisson(l, x), lh, 1e-6, 20)
print(f"Wald  [{w[0]:.3f}, {w[1]:.3f}]   perfil [{p[0]:.3f}, {p[1]:.3f}]   asimetría del perfil: "
      f"{lh - p[0]:.3f} abajo, {p[1] - lh:.3f} arriba")

# cobertura exacta: la suma S de n días es Poisson(nλ), así que basta recorrer S
def cobertura(lam, n, metodo):
    S = np.arange(0, int(n * lam + 12 * np.sqrt(n * lam) + 20)); ok = 0.0
    for s in S:
        lhat = s / n
        if metodo == "wald":
            h = 1.96 * np.sqrt(lhat / n); lo, hi = lhat - h, lhat + h
        else:
            ll = lambda l: s * np.log(l) - n * l
            lo = 0.0 if s == 0 else intervalo_perfil(ll, lhat, 1e-9, lhat)[0]
            hi = intervalo_perfil(ll, max(lhat, 1e-9), 1e-9, lhat + 20 / n + 10)[1] if s > 0 else 1.920729 / n
        ok += stats.poisson.pmf(s, n * lam) * (lo <= lam <= hi)
    return ok
for lam, n_ in ((2.5, 14), (0.3, 14), (0.3, 100)):
    print(f"cobertura con λ = {lam}, n = {n_:3d}: Wald {cobertura(lam, n_, 'wald'):.3f}   perfil {cobertura(lam, n_, 'perfil'):.3f}")

k, m = 3, 40
print(f"binomial 3/40: ee = {se_binom(k / m, m):.4f}   perfil 95 % = "
      + "[{:.4f}, {:.4f}]".format(*intervalo_perfil(lambda q: ll_binom(q, k, m), k / m, 1e-9, 1 - 1e-9)))
