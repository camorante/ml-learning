import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression, QuantileRegressor, LogisticRegression
from sklearn.metrics import log_loss
from sim import consumo_carga, DESCARGA, entregas
from mv import ll_normal, ll_laplace, logistica, ll_logistica

# 1) ruido normal → mínimos cuadrados
c, y = consumo_carga()
negll = lambda t: -ll_normal(0, np.exp(2 * t[2]), y - t[0] - t[1] * c)   # θ = (a, b, log σ)
t = minimize(negll, [30, 0, 0], method="BFGS").x
ols = LinearRegression().fit(c[:, None], y)
print(f"normal : MV a = {t[0]:.4f} b = {t[1]:.4f} σ̂ = {np.exp(t[2]):.4f} | MC a = {ols.intercept_:.4f} b = {ols.coef_[0]:.4f}"
      f" | √(SC/n) = {np.sqrt(np.mean((y - ols.predict(c[:, None]))**2)):.4f}")

# 2) ruido de Laplace → error absoluto (mediana; recta de la mediana)
r = minimize(lambda m: -ll_laplace(m[0], 4.0, DESCARGA), [28.0], method="Nelder-Mead", options={"xatol": 1e-8})
print(f"Laplace: MV posición = {r.x[0]:.4f} | mediana = {np.median(DESCARGA)} | media = {DESCARGA.mean():.4f}")
negll = lambda t: -ll_laplace(0, np.exp(t[2]), y - t[0] - t[1] * c)
t = minimize(negll, [30, 0, 0], method="Nelder-Mead", options={"xatol": 1e-9, "fatol": 1e-12, "maxiter": 20000}).x
qr = QuantileRegressor(quantile=0.5, alpha=0, solver="highs").fit(c[:, None], y)
print(f"         recta MV Laplace a = {t[0]:.4f} b = {t[1]:.4f} | QuantileRegressor(0.5) a = {qr.intercept_:.4f} b = {qr.coef_[0]:.4f}")

# 3) Bernoulli → pérdida logarítmica (entropía cruzada)
km, tarde = entregas(); X = (km[:, None] - 150) / 100
w = logistica(X, tarde)                                                 # Newton-Raphson sobre ℓ
sk = LogisticRegression(C=np.inf, max_iter=10_000, tol=1e-12).fit(X, tarde)
print(f"Bernoulli: MV w = {np.round(w, 5)} | sklearn C=inf: {np.round(np.r_[sk.intercept_, sk.coef_[0]], 5)}")
ll = ll_logistica(w, X, tarde)
print(f"   ℓ = {ll:.5f}  −ℓ/n = {-ll / len(tarde):.5f}  log_loss = {log_loss(tarde, sk.predict_proba(X)[:, 1]):.5f}")
print(f"   punto medio = {150 - w[0] / w[1] * 100:.2f} km, inclinación = {w[1]:.4f} por 100 km")
