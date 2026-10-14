import numpy as np
from scipy import stats
from dist import (norm_pdf, norm_cdf, norm_ppf, norm_rvs, binom_pmf, binom_cdf, binom_rvs, pois_pmf, pois_cdf,
                  pois_rvs, expon_cdf, lognorm_pdf, lognorm_cdf, ajuste_momentos, ajuste_lognormal_mv)

rng = np.random.default_rng(0)
err = lambda a, b: float(np.max(np.abs(np.asarray(a) - np.asarray(b))))
rel = lambda a, b: float(np.max(np.abs(np.asarray(a) - np.asarray(b)) / np.maximum(np.abs(b), 1e-300)))
x = np.linspace(-8, 8, 801)
print(f"{'norm_pdf vs stats.norm.pdf':58s}{err(norm_pdf(x, 1.5, 2.3), stats.norm.pdf(x, 1.5, 2.3)):.1e}")
print(f"{'norm_cdf vs stats.norm.cdf (relativo, colas incluidas)':58s}{rel(norm_cdf(x), stats.norm.cdf(x)):.1e}")
q = np.r_[1e-10, 1e-6, np.linspace(0.001, 0.999, 999), 1 - 1e-6]
print(f"{'norm_ppf vs stats.norm.ppf (q de 1e-10 a 1 − 1e-6)':58s}{err(norm_ppf(q), stats.norm.ppf(q)):.1e}")
eb, ec, ep, eq = 0.0, 0.0, 0.0, 0.0
for n in (1, 5, 20, 100, 1000):
    for p in (0.001, 0.1, 0.5, 0.93):
        k = np.arange(n + 1)
        eb = max(eb, err(binom_pmf(k, n, p), stats.binom.pmf(k, n, p)))
        ec = max(ec, err(binom_cdf(k, n, p), stats.binom.cdf(k, n, p)))
for lam in (0.05, 2, 9.5, 60, 300):
    k = np.arange(0, int(lam + 12 * np.sqrt(lam) + 20))
    ep = max(ep, err(pois_pmf(k, lam), stats.poisson.pmf(k, lam)))
    eq = max(eq, err(pois_cdf(k, lam), stats.poisson.cdf(k, lam)))
print(f"{'binom_pmf / binom_cdf (n de 1 a 1000, 4 valores de p)':58s}{eb:.1e} / {ec:.1e}")
print(f"{'pois_pmf / pois_cdf (λ de 0.05 a 300)':58s}{ep:.1e} / {eq:.1e}")
xp = np.linspace(0.01, 200, 500)
print(f"{'expon_cdf vs stats.expon.cdf':58s}{err(expon_cdf(xp, 0.2), stats.expon.cdf(xp, scale=5)):.1e}")
print(f"{'lognorm_pdf / lognorm_cdf vs stats.lognorm':58s}{err(lognorm_pdf(xp, np.log(12), 0.8), stats.lognorm.pdf(xp, 0.8, scale=12)):.1e} / "
      f"{err(lognorm_cdf(xp, np.log(12), 0.8), stats.lognorm.cdf(xp, 0.8, scale=12)):.1e}")
# muestreo: 200 000 muestras, media y varianza frente a las teóricas, y prueba KS / χ²
N = 200_000
z = norm_rvs(N, 32, 2.5, rng)
b = binom_rvs(N, 20, 0.1, rng)
po = pois_rvs(N, 2.0, rng)
print(f"{'norm_rvs: media, desv. típica (teórico 32, 2.5)':58s}{z.mean():.3f}, {z.std(ddof=1):.3f}   KS p = {stats.kstest(z, stats.norm(32, 2.5).cdf).pvalue:.2f}")
print(f"{'binom_rvs: media, varianza (teórico 2, 1.8)':58s}{b.mean():.3f}, {b.var(ddof=1):.3f}")
print(f"{'pois_rvs: media, varianza (teórico 2, 2)':58s}{po.mean():.3f}, {po.var(ddof=1):.3f}")
obs = np.bincount(po, minlength=12); obs = np.append(obs[:9], obs[9:].sum())
pe = stats.poisson.pmf(np.arange(9), 2.0); pe = np.append(pe, 1 - pe.sum())
print(f"{'pois_rvs: χ² frente a la pmf (9 grados de libertad)':58s}p = {stats.chisquare(obs, N * pe).pvalue:.2f}")
# ajuste
ln = stats.lognorm.rvs(0.8, scale=12, size=5000, random_state=1)
mv = ajuste_lognormal_mv(ln); s_sc, _, sc_sc = stats.lognorm.fit(ln, floc=0)
print(f"{'lognormal por MV vs stats.lognorm.fit(floc=0)':58s}{abs(mv['s'] - s_sc):.1e}, {abs(np.exp(mv['mu']) - sc_sc):.1e}")
nm = ajuste_momentos(z, "normal"); mu_sc, sd_sc = stats.norm.fit(z)
print(f"{'normal por momentos vs stats.norm.fit':58s}μ {abs(nm['mu'] - mu_sc):.1e}; σ con n − 1 / σ con n = {nm['sigma'] / sd_sc:.7f}")
