import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.utils.extmath import randomized_svd
from svd import svd_2x2, svd_jacobi, potencia_deflacion, svd_aleatorizada, rango, pinv, rango_k, normas

rng = np.random.default_rng(0)
err = {k: 0.0 for k in ["sv", "rec", "ort", "pinv", "rk", "ey2", "eyF", "nor", "2x2", "rango"]}
for t in range(200):
    m, n = rng.integers(1, 13, 2)
    A = rng.normal(size=(m, n))
    if t % 4 == 1 and min(m, n) > 1:                                 # rango deficiente
        r = rng.integers(1, min(m, n)); A = rng.normal(size=(m, r)) @ rng.normal(size=(r, n))
    if t % 4 == 2:                                                   # σ repetidos
        Q1, _ = np.linalg.qr(rng.normal(size=(m, m))); Q2, _ = np.linalg.qr(rng.normal(size=(n, n)))
        d = np.zeros((m, n)); p = min(m, n); d[range(p), range(p)] = np.repeat([3.0, 1.0], p)[:p]
        A = Q1 @ d @ Q2.T
    U, s, Vt = svd_jacobi(A); s0 = np.linalg.svd(A, compute_uv=False); sc = s0[0] if s0[0] else 1
    err["sv"] = max(err["sv"], np.abs(s - s0).max() / sc)
    err["rec"] = max(err["rec"], np.abs((U * s) @ Vt - A).max() / sc)
    err["ort"] = max(err["ort"], np.abs(U.T @ U - np.eye(U.shape[1])).max(), np.abs(Vt @ Vt.T - np.eye(Vt.shape[0])).max())
    err["pinv"] = max(err["pinv"], np.abs(pinv(A) - np.linalg.pinv(A)).max() * sc)
    err["rango"] = max(err["rango"], abs(rango(A) - np.linalg.matrix_rank(A)))
    for k in range(1, min(m, n)):
        if s0[k - 1] - s0[k] < 1e-8 * sc:                             # con σₖ = σₖ₊₁ la aproximación no es única
            continue
        Uo, so, Vo = np.linalg.svd(A, full_matrices=False); Ak0 = (Uo[:, :k] * so[:k]) @ Vo[:k]
        Ak = rango_k(A, k)
        err["rk"] = max(err["rk"], np.abs(Ak - Ak0).max() / sc)
        err["ey2"] = max(err["ey2"], abs(np.linalg.norm(A - Ak, 2) - s0[k]) / sc)
        err["eyF"] = max(err["eyF"], abs(np.linalg.norm(A - Ak, "fro") - np.sqrt(np.sum(s0[k:] ** 2))) / sc)
    n2, nf, c = normas(A)
    err["nor"] = max(err["nor"], abs(n2 - np.linalg.norm(A, 2)) / sc, abs(nf - np.linalg.norm(A, "fro")) / sc)
for t in range(200):
    A = rng.normal(size=(2, 2)) * rng.choice([0.01, 1, 100])
    U, s, Vt = svd_2x2(A); s0 = np.linalg.svd(A, compute_uv=False)
    err["2x2"] = max(err["2x2"], np.abs(s - s0).max() / s0[0], np.abs((U * s) @ Vt - A).max() / s0[0], np.abs(U.T @ U - np.eye(2)).max())
print(f"svd_jacobi: valores singulares vs np.linalg.svd (relativo)     {err['sv']:.1e}")
print(f"svd_jacobi: |U Σ Vᵀ − A| (relativo)                            {err['rec']:.1e}")
print(f"svd_jacobi: |UᵀU − I|, |VᵀV − I|                               {err['ort']:.1e}")
print(f"pinv vs np.linalg.pinv                                         {err['pinv']:.1e}")
print(f"rango vs np.linalg.matrix_rank (diferencias)                   {err['rango']:.0f}")
print(f"rango_k vs aproximación con np.linalg.svd                      {err['rk']:.1e}")
print(f"Eckart–Young: ‖A − A_k‖₂ − σ_k+1                               {err['ey2']:.1e}")
print(f"Eckart–Young: ‖A − A_k‖_F − √(Σ σ² restantes)                  {err['eyF']:.1e}")
print(f"normas (2 y Frobenius) vs np.linalg.norm                       {err['nor']:.1e}")
print(f"svd_2x2 (200 matrices, escalas 0.01 a 100)                     {err['2x2']:.1e}")
# matrices grandes de rango bajo + ruido: capas sueltas frente a scikit-learn
M = rng.normal(size=(400, 6)) @ np.diag([50, 30, 20, 12, 8, 5]) @ rng.normal(size=(6, 120)) / 10 + 0.05 * rng.normal(size=(400, 120))
ts = TruncatedSVD(n_components=6, algorithm="arpack").fit(M)
U, s, Vt = potencia_deflacion(M, 6)
cos = np.abs(np.sum(Vt * ts.components_, axis=1))
print(f"potencia_deflacion vs TruncatedSVD(arpack): σ (relativo)       {np.abs(s - ts.singular_values_).max() / s[0]:.1e}")
print(f"  |cos| entre componentes (1 = misma dirección, salvo signo)   mín {cos.min():.12f}")
Ur, sr, Vr = svd_aleatorizada(M, 6); Us, ss, Vs = randomized_svd(M, 6, random_state=0)
print(f"svd_aleatorizada vs randomized_svd de sklearn: σ (relativo)    {np.abs(sr - ss).max() / ss[0]:.1e}")
print(f"  |cos| mínimo entre componentes                               {np.abs(np.sum(Vr * Vs, axis=1)).min():.12f}")
