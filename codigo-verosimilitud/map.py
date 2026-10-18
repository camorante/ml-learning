import numpy as np
from scipy import stats, optimize
from sklearn.linear_model import LogisticRegression
from sim import entregas
from mv import map_binom, logistica, ll_binom

# previa Beta(1 + m·p0, 1 + m·(1 − p0)) para la tasa de retrasos de un cliente nuevo (flota: 8 %)
p0 = 0.08
for k, n in ((0, 5), (3, 40), (16, 200)):
    m = 20; a, b = 1 + m * p0, 1 + m * (1 - p0)
    num = optimize.minimize_scalar(lambda p: -(ll_binom(p, k, n) + stats.beta.logpdf(p, a, b)),
                                   bounds=(1e-9, 1 - 1e-9), method="bounded", options={"xatol": 1e-10}).x
    print(f"{k:2d}/{n:3d}: MV {k / n:.4f}  MAP fórmula {map_binom(k, n, m, p0):.4f}  MAP numérico {num:.4f}"
          f"  media a posteriori {(k + a) / (n + a + b):.4f}")

# regresión logística: previa normal N(0, 1/λ) en el peso = penalización l2 (λ/2)·w²
km, y = entregas(); X = (km[:, None] - 150) / 100
for lam in (0.0, 1.0, 10.0):
    w = logistica(X, y, l2=lam)
    C = np.inf if lam == 0 else 1 / lam
    sk = LogisticRegression(C=C, tol=1e-12, max_iter=10_000).fit(X, y)
    print(f"λ = {lam:4.1f}: MAP desde cero {np.round(w, 5)}  sklearn C = {C}: {np.round(np.r_[sk.intercept_, sk.coef_[0]], 5)}")
