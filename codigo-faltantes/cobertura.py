"""Sesgo y cobertura del IC del 95 % de la media según el método, con huecos MAR."""
import numpy as np, json
from miss import imputar_media, imputar_regresion, rubin

def un_ensayo(rng, p, n=200, m=5):
    x = rng.normal(0, 1, n); y = 10 + 3 * x + rng.normal(0, 2, n)
    # MAR: falta más cuando x es alto (y se ve x)
    a = np.log(p / (1 - p)); falta = rng.random(n) < 1 / (1 + np.exp(-(a + 1.5 * x)))
    yo = np.where(falta, np.nan, y); out = {}
    cc = yo[~falta]; out["casos completos"] = (cc.mean(), cc.std(ddof=1) / np.sqrt(len(cc)))
    z = imputar_media(yo); out["media"] = (z.mean(), z.std(ddof=1) / np.sqrt(n))
    z = imputar_regresion(yo, x[:, None]); out["regresión"] = (z.mean(), z.std(ddof=1) / np.sqrt(n))
    z = imputar_regresion(yo, x[:, None], ruido=True, seed=int(rng.integers(1e9))); out["regresión + ruido"] = (z.mean(), z.std(ddof=1) / np.sqrt(n))
    q, u = [], []
    for k in range(m):                          # imputación múltiple "propia": bootstrap de los observados para los coeficientes
        b = rng.choice(np.where(~falta)[0], (~falta).sum()); A = np.c_[np.ones(len(b)), x[b]]
        beta = np.linalg.lstsq(A, y[b], rcond=None)[0]; res = y[b] - A @ beta
        z = yo.copy(); z[falta] = beta[0] + beta[1] * x[falta] + rng.choice(res, falta.sum())
        q.append(z.mean()); u.append(z.var(ddof=1) / n)
    qb, T, W, B = rubin(q, u); out["imputación múltiple (m=5)"] = (qb, np.sqrt(T))
    return out

rng = np.random.default_rng(0); R = {}
for p in [0.1, 0.2, 0.3, 0.4, 0.5]:
    acc = {}
    for _ in range(1000):
        for k, (est, se) in un_ensayo(rng, p).items():
            acc.setdefault(k, []).append((est - 10, abs(est - 10) <= 1.96 * se))
    R[p] = {k: (round(float(np.mean([a for a, _ in v])), 3), round(float(np.mean([c for _, c in v])), 3)) for k, v in acc.items()}
    print(p, R[p])
json.dump(R, open("../cobertura.json", "w"))
