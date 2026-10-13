import numpy as np
from svd import svd_jacobi

rng = np.random.default_rng(2)
U, _ = np.linalg.qr(rng.normal(size=(50, 5))); V, _ = np.linalg.qr(rng.normal(size=(5, 5)))
print("σ_min     vía AᵀA (eigvalsh)   svd_jacobi    np.linalg.svd   (error relativo en σ_min)")
for k in [2, 4, 6, 8, 10]:
    s = np.logspace(0, -k, 5); A = U @ np.diag(s) @ V.T
    ev = np.linalg.eigvalsh(A.T @ A)                              # autovalores de AᵀA, de menor a mayor
    via = np.sqrt(np.abs(ev[0]))
    e = lambda x: abs(x - s[-1]) / s[-1]
    print(f"1e-{k:<2d}     {e(via):17.1e}   {e(svd_jacobi(A)[1][-1]):10.1e}   {e(np.linalg.svd(A, compute_uv=False)[-1]):13.1e}")
