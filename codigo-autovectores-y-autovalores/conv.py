import numpy as np
from autovectores import potencia, potencia_inversa

rng = np.random.default_rng(2)
Q, _ = np.linalg.qr(rng.normal(size=(50, 50)))             # autovectores: una base ortonormal al azar de R⁵⁰
print("|λ₂/λ₁|  iteraciones hasta error de dirección < 1e-10   predicción log(1e-10)/log|λ₂/λ₁|")
for r in [0.1, 0.5, 0.9, 0.99]:
    S = Q @ np.diag(np.r_[10.0, 10 * r, rng.uniform(0, 9 * r, 48)]) @ Q.T
    x = rng.normal(size=50); x /= np.linalg.norm(x); v1 = Q[:, 0]; it = 0
    while min(np.linalg.norm(x - v1), np.linalg.norm(x + v1)) > 1e-10:
        x = S @ x; x /= np.linalg.norm(x); it += 1
    print(f"{r:6.2f}   {it:8d}                                       {np.log(1e-10) / np.log(r):8.0f}")
# λ₁ = 10, λ₂ = 9.9: la potencia sufre; la inversa desplazada converge al autovalor más cercano a μ
S = Q @ np.diag(np.r_[10.0, 9.9, rng.uniform(0, 8, 48)]) @ Q.T
l, v, k = potencia(S, rng.normal(size=50), tol=1e-14, max_iter=100_000)
print(f"potencia:            λ = {l:.12f}  en {k} iteraciones")
for mu in [9.0, 9.97, 10.01]:
    l, v, k = potencia_inversa(S, mu, x0=rng.normal(size=50), tol=1e-14)
    print(f"inversa, μ = {mu:5.2f}:  λ = {l:.12f}  en {k:3d} iteraciones   (razón |λ₁−μ|/|λ₂−μ| = {abs(10 - mu) / abs(9.9 - mu):.3f})")
