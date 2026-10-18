
import numpy as np
from scipy import stats, special
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sim import alertas_conductores, entregas
from mv import ajustar_negbin, log_sigmoid

# 1) binomial negativa: sin fórmula cerrada para r → minimize sobre (log r, log μ)
a, _ = alertas_conductores()
r, mu, ll = ajustar_negbin(a)
res = stats.fit(stats.nbinom, a, bounds={"n": (0.01, 50), "p": (1e-4, 1 - 1e-4), "loc": (0, 0)},
                optimizer=lambda fun, bounds, **kw: minimize(fun, [1.0, 0.5, 0.0], bounds=bounds, method="L-BFGS-B"))
print(f"desde cero: r = {r:.5f}  p = r/(r+μ) = {r / (r + mu):.5f}  ℓ = {ll:.4f}")
print(f"stats.fit : n = {res.params.n:.5f}  p = {res.params.p:.5f}  ℓ = {stats.nbinom.logpmf(a, res.params.n, res.params.p).sum():.4f}")

# 2) logística con minimize y gradiente analítico frente a scikit-learn
km, y = entregas(); X = np.c_[np.ones(40), (km - 150) / 100]
def negll(w):
    z = X @ w; return -np.sum(y * log_sigmoid(z) + (1 - y) * log_sigmoid(-z))
def grad(w):
    return -X.T @ (y - special.expit(X @ w))
w = minimize(negll, np.zeros(2), jac=grad, method="BFGS", options={"gtol": 1e-10}).x
sk = LogisticRegression(C=np.inf, tol=1e-12, max_iter=10_000).fit(X[:, 1:], y)
print("BFGS:", np.round(w, 6), " sklearn:", np.round(np.r_[sk.intercept_, sk.coef_[0]], 6))
