"""Velocidad a partir de la distancia acumulada con ruido: pseudoinversa, SVD truncada y ridge."""
import numpy as np
from sklearn.linear_model import Ridge
from svd import svd_jacobi

n = 120; t = np.arange(n)                                       # 2 minutos, una muestra por segundo
v = np.interp(t, [0, 25, 55, 70, 80, 95, 119], [0, 22, 23, 9, 9, 20, 20])   # m/s: arranca, rotonda, sigue
A = np.tril(np.ones((n, n)))                                    # distancia = suma acumulada de la velocidad (Δt = 1 s)
rng = np.random.default_rng(5)
b = A @ v + rng.normal(0, 3.0, n)                               # GPS: ±3 m en la distancia
U, s, Vt = svd_jacobi(A)
c = U.T @ b
print(f"σ₁ = {s[0]:.1f}, σ_n = {s[-1]:.3f}, κ(A) = {s[0] / s[-1]:.0f}")
rmse = lambda x: np.sqrt(np.mean((x - v) ** 2))
x_pinv = Vt.T @ (c / s)
print(f"pseudoinversa (todas las capas): RMSE {rmse(x_pinv):.2f} m/s")
best_k = min(range(1, n + 1), key=lambda k: rmse(Vt[:k].T @ (c[:k] / s[:k])))
print(f"SVD truncada, mejor k = {best_k}: RMSE {rmse(Vt[:best_k].T @ (c[:best_k] / s[:best_k])):.2f} m/s")
lams = 10 ** np.linspace(-3, 3, 241)
xr = lambda lam: Vt.T @ (s / (s**2 + lam) * c)                   # filtros σ²/(σ²+λ) aplicados a c/σ
best_l = min(lams, key=lambda l: rmse(xr(l)))
print(f"ridge, mejor λ = {best_l:.3g}: RMSE {rmse(xr(best_l)):.2f} m/s")
sk = Ridge(alpha=best_l, fit_intercept=False, solver="svd").fit(A, b).coef_
print(f"ridge por filtros vs sklearn Ridge(solver='svd'): {np.abs(sk - xr(best_l)).max():.1e}")
f = s**2 / (s**2 + best_l)
print(f"filtros con ese λ: f₁ = {f[0]:.4f}, f_{best_k} = {f[best_k - 1]:.3f}, f_n = {f[-1]:.4f}; grados de libertad {f.sum():.1f}")
