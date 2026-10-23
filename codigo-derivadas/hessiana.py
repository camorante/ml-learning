import numpy as np
from datos import CARGA, CONSUMO

X = np.c_[np.ones(10), CARGA / 10]; y = CONSUMO          # parámetros: altura, L por cada 10 t
sse = lambda b: float(np.sum((y - X @ b) ** 2))
grad = lambda b: -2 * X.T @ (y - X @ b)                   # ∇ SSE = 2Xᵀ(Xβ − y)
H = 2 * X.T @ X                                           # hessiana: constante (SSE es cuadrática)
lam, V = np.linalg.eigh(H)
b0 = np.array([23.0, 6.0]); smin = sse(np.linalg.lstsq(X, y, rcond=None)[0])
print("gradiente en (23, 6):", grad(b0).round(3), "  SSE:", round(sse(b0), 3))
print("hessiana:", H.round(3).tolist(), " autovalores:", lam.round(3),
      " κ =", round(lam[1] / lam[0], 2))
print("paso máximo estable 2/λmax =", round(2 / lam[1], 4))
for eta in (0.005, 0.03, 0.04):
    r = np.max(np.abs(1 - eta * lam)); b = b0.copy(); n = 0
    while 1e-6 < sse(b) - smin < 1e12 and n < 5000:
        b = b - eta * grad(b); n += 1
    print(f"η = {eta}: factor por paso max|1 − ηλ| = {r:.4f}"
          f"  pasos hasta SSE − mín < 1e-6: {n if r < 1 else 'diverge'}")
bn = b0 - np.linalg.solve(H, grad(b0))                    # Newton: un paso con la curvatura
print("Newton desde (23, 6), un paso:", bn.round(4), " SSE", round(sse(bn), 4),
      " lstsq:", np.linalg.lstsq(X, y, rcond=None)[0].round(4))
Xc = np.c_[np.ones(10), (CARGA - CARGA.mean()) / 10]        # centrar la carga (escalado, D7)
lc = np.linalg.eigvalsh(2 * Xc.T @ Xc)
print("con la carga centrada: autovalores", lc.round(3), " κ =", round(lc[1] / lc[0], 2))
# XGBoost con pérdida logarítmica: gradiente y hessiana por dato respecto a la puntuación z
p, yy, lam_reg = 0.3775, 1, 1.0
g, h = p - yy, p * (1 - p)
print(f"XGBoost, una entrega con p = {p}, tarde: g = {g:.4f}  h = {h:.4f}"
      f"  peso de hoja −g/(h + λ) = {-g / (h + lam_reg):.4f}")
