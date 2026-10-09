import numpy as np
from numpy.linalg import lstsq
from sklearn.linear_model import LinearRegression, Ridge
from mcuadrados import (con_intercepto, normales, householder_qr, mc_qr, mc_svd, ridge,
                        sombrero, r2, vif, ponderados)

rng = np.random.default_rng(0)
err = {k: 0.0 for k in ["normales", "qr", "svd", "QR", "ortog", "sk", "ridge", "H", "Xtr", "r2", "wls"]}
for _ in range(200):
    n, p = rng.integers(8, 60), rng.integers(1, 7)
    X = rng.normal(0, 1, (n, p)) * rng.uniform(0.5, 5, p); y = X @ rng.normal(0, 2, p) + rng.normal(0, 1, n)
    A = con_intercepto(X); ref = lstsq(A, y, rcond=None)[0]
    rel = lambda b: np.max(np.abs(b - ref)) / np.max(np.abs(ref))
    err["normales"] = max(err["normales"], rel(normales(A, y)))
    err["qr"] = max(err["qr"], rel(mc_qr(A, y)))
    err["svd"] = max(err["svd"], rel(mc_svd(A, y)))
    Q, R = householder_qr(A); Qn, Rn = np.linalg.qr(A)
    sg = np.sign(np.diag(R)) * np.sign(np.diag(Rn))           # QR es única salvo el signo de cada columna
    err["QR"] = max(err["QR"], np.max(np.abs(Q * sg - Qn)), np.max(np.abs(Q @ R - A)))
    err["ortog"] = max(err["ortog"], np.max(np.abs(Q.T @ Q - np.eye(A.shape[1]))))
    lr = LinearRegression().fit(X, y); b = mc_qr(A, y)
    err["sk"] = max(err["sk"], np.max(np.abs(np.r_[lr.intercept_, lr.coef_] - b)) / np.max(np.abs(b)))
    lam = rng.uniform(0.01, 50); b0, bb = ridge(X, y, lam); rd = Ridge(alpha=lam, solver="svd").fit(X, y)
    err["ridge"] = max(err["ridge"], np.max(np.abs(np.r_[b0 - rd.intercept_, bb - rd.coef_])))
    H = sombrero(A)
    err["H"] = max(err["H"], np.max(np.abs(H @ H - H)), np.max(np.abs(H - H.T)), abs(np.trace(H) - A.shape[1]))
    r = y - A @ b
    err["Xtr"] = max(err["Xtr"], np.max(np.abs(A.T @ r)) / (np.linalg.norm(A) * np.linalg.norm(y)))
    err["r2"] = max(err["r2"], abs(r2(y, A @ b) - lr.score(X, y)))
    w = rng.uniform(0.1, 3, n)
    lw = LinearRegression().fit(X, y, sample_weight=w)
    err["wls"] = max(err["wls"], np.max(np.abs(ponderados(A, y, w) - np.r_[lw.intercept_, lw.coef_])) / np.max(np.abs(b)))

print(f"normales vs np.linalg.lstsq (error relativo)          {err['normales']:.1e}")
print(f"mc_qr (Householder) vs lstsq                          {err['qr']:.1e}")
print(f"mc_svd (pseudoinversa) vs lstsq                       {err['svd']:.1e}")
print(f"householder_qr vs np.linalg.qr (salvo signo), QR - X  {err['QR']:.1e}")
print(f"|QᵀQ - I|                                             {err['ortog']:.1e}")
print(f"coeficientes vs sklearn LinearRegression              {err['sk']:.1e}")
print(f"ridge vs sklearn Ridge (200 valores de λ)             {err['ridge']:.1e}")
print(f"H² = H, H = Hᵀ, traza(H) = p                          {err['H']:.1e}")
print(f"Xᵀr = 0 (relativo a ‖X‖‖y‖)                           {err['Xtr']:.1e}")
print(f"r2 vs LinearRegression.score                          {err['r2']:.1e}")
print(f"ponderados vs LinearRegression(sample_weight)         {err['wls']:.1e}")

# VIF frente a statsmodels si está instalado; si no, frente a la diagonal de la inversa de la correlación
X = rng.normal(0, 1, (300, 4)); X[:, 3] = X[:, 0] + 0.1 * rng.normal(0, 1, 300)
C = np.corrcoef(X, rowvar=False)
print(f"vif vs diag(corr⁻¹)                                   {np.max(np.abs(vif(X) - np.diag(np.linalg.inv(C)))):.1e}")
# matriz de rango deficiente: la SVD da la solución de norma mínima, como lstsq
X = rng.normal(0, 1, (20, 3)); X = np.c_[X, X[:, 0] + X[:, 1]]; y = rng.normal(0, 1, 20)
print(f"rango deficiente: mc_svd vs lstsq                     {np.max(np.abs(mc_svd(X, y) - lstsq(X, y, rcond=None)[0])):.1e}")
