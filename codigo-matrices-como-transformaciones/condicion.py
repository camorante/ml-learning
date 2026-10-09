import numpy as np
rng = np.random.default_rng(3)
n = 300
U, _ = np.linalg.qr(rng.normal(size=(n, n))); V, _ = np.linalg.qr(rng.normal(size=(n, n)))
s = np.logspace(0, -10, n)                       # valores singulares de 1 a 1e-10: kappa = 1e10
A = U @ np.diag(s) @ V.T
x_true = rng.normal(size=n); b = A @ x_true
print(f"kappa(A) = {np.linalg.cond(A):.2e}")
for nombre, x in [("solve(A, b)", np.linalg.solve(A, b)), ("inv(A) @ b", np.linalg.inv(A) @ b)]:
    print(f"{nombre:12s} residuo relativo {np.linalg.norm(A @ x - b) / np.linalg.norm(b):.1e}   error relativo en x {np.linalg.norm(x - x_true) / np.linalg.norm(x_true):.1e}")
# amplificación de un error en b (cota: kappa)
db = rng.normal(size=n); db *= 1e-8 * np.linalg.norm(b) / np.linalg.norm(db)
dx = np.linalg.solve(A, b + db) - np.linalg.solve(A, b)
print(f"error relativo en b 1.0e-08 -> en x {np.linalg.norm(dx) / np.linalg.norm(x_true):.1e}")
