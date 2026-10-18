"""Compara cor.py con numpy, pandas y scipy en 300 problemas aleatorios (con empates)."""
import numpy as np, pandas as pd
from scipy import stats
from cor import *

rng = np.random.default_rng(0); err = {}
def up(k, v): err[k] = max(err.get(k, 0.0), float(v))
for t in range(300):
    n = int(rng.integers(5, 200)); rho = rng.uniform(-0.95, 0.95)
    x = rng.normal(size=n); y = rho * x + np.sqrt(1 - rho**2) * rng.normal(size=n)
    if t % 3 == 0: x, y = np.round(x, 1), np.round(y, 1)          # empates
    X = np.column_stack([x, y, rng.normal(size=n), x**2])
    up("covarianza vs np.cov", abs(covarianza(x, y) - np.cov(x, y)[0, 1]))
    up("matriz_cov vs np.cov", np.abs(matriz_cov(X) - np.cov(X.T)).max())
    up("matriz_corr vs np.corrcoef / DataFrame.corr", max(np.abs(matriz_corr(X) - np.corrcoef(X.T)).max(), np.abs(matriz_corr(X) - pd.DataFrame(X).corr().values).max()))
    pr = stats.pearsonr(x, y); r = pearson(x, y)
    up("pearson vs scipy.stats.pearsonr", abs(r - pr.statistic))
    up("p_valor_r vs pearsonr().pvalue (relativo)", abs(p_valor_r(r, n) - pr.pvalue) / max(pr.pvalue, 1e-300))
    lo, hi = ic_fisher(r, n); ci = pr.confidence_interval(0.95)
    up("ic_fisher vs pearsonr().confidence_interval", max(abs(lo - ci.low), abs(hi - ci.high)))
    up("rangos vs scipy.stats.rankdata", np.abs(rangos(x) - stats.rankdata(x)).max())
    up("spearman vs spearmanr / corr('spearman')", max(abs(spearman(x, y) - stats.spearmanr(x, y).statistic), abs(spearman(x, y) - pd.Series(x).corr(pd.Series(y), method="spearman"))))
    up("kendall_tau_b vs kendalltau / corr('kendall')", max(abs(kendall_tau_b(x, y) - stats.kendalltau(x, y).statistic), abs(kendall_tau_b(x, y) - pd.Series(x).corr(pd.Series(y), method="kendall"))))
    z = X[:, 2]; a = np.c_[np.ones(n), z]
    rx = x - a @ np.linalg.lstsq(a, x, rcond=None)[0]; ry = y - a @ np.linalg.lstsq(a, y, rcond=None)[0]
    rxy, rxz, ryz = np.corrcoef(x, y)[0, 1], np.corrcoef(x, z)[0, 1], np.corrcoef(y, z)[0, 1]
    up("correlacion_parcial vs fórmula de 3 r", abs(correlacion_parcial(x, y, z) - (rxy - rxz * ryz) / np.sqrt((1 - rxz**2) * (1 - ryz**2))))
    # en línea, por trozos y fusionando
    parts = np.array_split(np.arange(n), 3); tot = CovOnline()
    for p in parts:
        c = CovOnline()
        for i in p: c.add(x[i], y[i])
        tot.merge(c)                                   # fusionar con un resumen vacío también vale
    up("CovOnline (3 trozos fusionados) vs np.cov / corrcoef", max(abs(tot.cov() - np.cov(x, y)[0, 1]), abs(tot.corr() - np.corrcoef(x, y)[0, 1])))
    Xv = X[:, :3]
    A = np.c_[np.ones(n), Xv[:, 1:]]; res = Xv[:, 0] - A @ np.linalg.lstsq(A, Xv[:, 0], rcond=None)[0]
    up("vif vs 1/(1 − R²) por regresión", abs(vif(Xv)[0] - 1 / (np.sum(res**2) / np.sum((Xv[:, 0] - Xv[:, 0].mean())**2))))
w = max(len(k) for k in err)
for k, v in err.items(): print(f"# {k:{w}s}  {v:.1e}")
