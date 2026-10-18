"""Compara mv.py con SciPy y scikit-learn en 200 problemas aleatorios: error máximo."""
import numpy as np
from scipy import stats, special, optimize
from sklearn.linear_model import LogisticRegression
import mv

rng = np.random.default_rng(2026); E = {}
def anota(k, a, b, rel=False):
    a, b = np.asarray(a, float), np.asarray(b, float)
    e = np.max(np.abs(a - b) / (np.maximum(np.abs(b), 1e-300) if rel else 1)); E[k] = max(E.get(k, 0.0), float(e))

for _ in range(200):
    n = int(rng.integers(5, 300)); lam = rng.uniform(0.05, 30); x = rng.poisson(lam, n)
    anota("ll_poisson vs poisson.logpmf", mv.ll_poisson(lam, x), stats.poisson.logpmf(x, lam).sum(), True)
    N = int(rng.integers(1, 500)); k = int(rng.integers(0, N + 1)); p = rng.uniform(0.001, 0.999)
    anota("ll_binom vs binom.logpmf", mv.ll_binom(p, k, N), stats.binom.logpmf(k, N, p), True)
    z = rng.normal(rng.uniform(-50, 50), rng.uniform(0.1, 20), n); mu, s2 = mv.mle_normal(z)
    anota("ll_normal vs norm.logpdf", mv.ll_normal(mu, s2, z), stats.norm.logpdf(z, mu, np.sqrt(s2)).sum(), True)
    anota("mle_normal vs norm.fit", [mu, np.sqrt(s2)], stats.norm.fit(z), True)
    m, b = mv.mle_laplace(z)
    anota("ll_laplace vs laplace.logpdf", mv.ll_laplace(m, b, z), stats.laplace.logpdf(z, m, b).sum(), True)
    anota("mle_laplace vs laplace.fit", [m, b], stats.laplace.fit(z), True)
    t = rng.exponential(rng.uniform(0.5, 50), n); tasa = mv.mle_expon(t)
    anota("ll_expon vs expon.logpdf", mv.ll_expon(tasa, t), stats.expon.logpdf(t, scale=1 / tasa).sum(), True)
    anota("mle_expon vs expon.fit(floc=0)", 1 / tasa, stats.expon.fit(t, floc=0)[1], True)
    q = rng.lognormal(rng.uniform(0, 4), rng.uniform(0.2, 1.5), n); lm, ls = mv.mle_lognormal(q)
    sh, _, sc = stats.lognorm.fit(q, floc=0)
    anota("ll_lognormal vs lognorm.logpdf", mv.ll_lognormal(lm, ls, q), stats.lognorm.logpdf(q, ls, scale=np.exp(lm)).sum(), True)
    anota("mle_lognormal vs lognorm.fit(floc=0)", [ls, np.exp(lm)], [sh, sc], True)
    r, mn = rng.uniform(0.3, 20), rng.uniform(0.2, 15); c = rng.negative_binomial(r, r / (r + mn), n)
    anota("ll_negbin vs nbinom.logpmf", mv.ll_negbin(r, mn, c), stats.nbinom.logpmf(c, r, r / (r + mn)).sum(), True)
    v = rng.normal(0, 300, 20)
    anota("logsumexp vs special.logsumexp", mv.logsumexp(v), special.logsumexp(v), True)
    anota("log_sigmoid vs special.log_expit", mv.log_sigmoid(v), special.log_expit(v), True)
    d = rng.uniform(0, 30)
    anota("razon_verosimilitud p vs chi2.sf", mv.razon_verosimilitud(0, d / 2)[1], stats.chi2.sf(d, 1), True)
    if x.sum() > 0:
        lh = x.mean(); f = lambda l: mv.ll_poisson(l, x) - mv.ll_poisson(lh, x) + stats.chi2.ppf(0.95, 1) / 2
        ref = (optimize.brentq(f, 1e-9, lh), optimize.brentq(f, lh, lh * 10 + 10))
        anota("intervalo_perfil vs brentq + chi2.ppf", mv.intervalo_perfil(lambda l: mv.ll_poisson(l, x), lh, 1e-9, lh * 10 + 10), ref, True)
for _ in range(50):
    n, d = int(rng.integers(40, 400)), int(rng.integers(1, 5)); X = rng.normal(0, 1, (n, d)); w = rng.normal(0, 1, d + 1)
    y = (rng.random(n) < special.expit(w[0] + X @ w[1:])).astype(int)
    sk = LogisticRegression(C=np.inf, solver="newton-cholesky", tol=1e-12, max_iter=1000).fit(X, y)
    anota("logistica vs LogisticRegression(C=inf)", mv.logistica(X, y), np.r_[sk.intercept_, sk.coef_[0]], True)
    lam = rng.uniform(0.1, 10); sk = LogisticRegression(C=1 / lam, solver="newton-cholesky", tol=1e-12, max_iter=1000).fit(X, y)
    anota("logistica l2 vs LogisticRegression(C=1/λ)", mv.logistica(X, y, l2=lam), np.r_[sk.intercept_, sk.coef_[0]], True)
for k, e in E.items():
    print(f"{k:44s} {e:.1e}")
