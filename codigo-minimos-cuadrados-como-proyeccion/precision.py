import numpy as np
from mcuadrados import normales, mc_qr, mc_svd

rng = np.random.default_rng(1)
n, p = 200, 6
U, _ = np.linalg.qr(rng.normal(size=(n, p))); V, _ = np.linalg.qr(rng.normal(size=(p, p)))
beta = rng.normal(size=p)
print("cond(X)   cond(XᵀX)   error normales   error QR    error SVD   error lstsq")
for k in [2, 4, 6, 7, 8]:
    s = np.logspace(0, -k, p)                      # valores singulares de 1 a 10^-k
    X = U @ np.diag(s) @ V.T; y = X @ beta         # sistema compatible: la solución exacta es beta
    e = lambda b: np.linalg.norm(b - beta) / np.linalg.norm(beta)
    print(f"{np.linalg.cond(X):8.0e}  {np.linalg.cond(X.T @ X):9.0e}   {e(normales(X, y)):13.1e}   "
          f"{e(mc_qr(X, y)):8.1e}   {e(mc_svd(X, y)):8.1e}   {e(np.linalg.lstsq(X, y, rcond=None)[0]):8.1e}")
