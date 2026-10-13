import numpy as np
from sklearn.decomposition import PCA
from autovectores import (eig2, potencia, potencia_inversa, deflacion, jacobi, qr_simetrica,
                          estacionaria, pca)

rng = np.random.default_rng(0)
def simetrica_separada(n):
    """Simétrica aleatoria con autovalores separados (|λ| distintos), para que la potencia converja."""
    Q, _ = np.linalg.qr(rng.normal(size=(n, n)))
    lam = rng.permutation(np.r_[rng.uniform(1, 10, n // 2 + n % 2), -rng.uniform(0.5, 9.5, n // 2)])
    lam = np.sign(lam) * (np.abs(lam) + 0.15 * np.arange(n)[np.argsort(np.argsort(np.abs(lam)))])
    return Q @ np.diag(lam) @ Q.T
def mismo_vector(u, v):
    """Distancia entre direcciones: los autovectores solo están definidos salvo el signo."""
    return min(np.linalg.norm(u - v), np.linalg.norm(u + v))

e = dict.fromkeys(["eig2", "pot", "potv", "inv", "def", "jac", "jacv", "qr", "qrv", "rec", "est", "pca", "pcav", "evr"], 0.0)
for _ in range(300):
    A = rng.normal(size=(2, 2))
    e["eig2"] = max(e["eig2"], np.max(np.abs(np.sort_complex(eig2(A)) - np.sort_complex(np.linalg.eigvals(A).astype(complex)))))
for _ in range(200):
    n = int(rng.integers(2, 9)); S = simetrica_separada(n)
    w, V = np.linalg.eigh(S); i = np.argmax(np.abs(w))
    lam, v, _ = potencia(S, rng.normal(size=n), tol=1e-15, max_iter=200_000)
    e["pot"] = max(e["pot"], abs(lam - w[i]) / abs(w[i])); e["potv"] = max(e["potv"], mismo_vector(v, V[:, i]))
    j = rng.integers(n); mu = w[j] + 0.1 * rng.uniform(-1, 1) * np.min(np.abs(np.delete(w, j) - w[j]))
    li, vi, _ = potencia_inversa(S, mu)
    e["inv"] = max(e["inv"], abs(li - w[j]) / np.max(np.abs(w)))
    ld, Vd = deflacion(S)
    e["def"] = max(e["def"], np.max(np.abs(np.sort(ld) - w)) / np.max(np.abs(w)))
    lj, Vj = jacobi(S); lq, Vq, _ = qr_simetrica(S); ws, Vs = w[::-1], V[:, ::-1]
    e["jac"] = max(e["jac"], np.max(np.abs(lj - ws)) / np.max(np.abs(w)))
    e["jacv"] = max(e["jacv"], max(mismo_vector(Vj[:, k], Vs[:, k]) for k in range(n)))
    e["qr"] = max(e["qr"], np.max(np.abs(lq - ws)) / np.max(np.abs(w)))
    e["qrv"] = max(e["qrv"], max(mismo_vector(Vq[:, k], Vs[:, k]) for k in range(n)))
    e["rec"] = max(e["rec"], np.max(np.abs(Vj @ np.diag(lj) @ Vj.T - S)) / np.max(np.abs(S)),
                   np.max(np.abs(Vj.T @ Vj - np.eye(n))))
for _ in range(200):
    n = int(rng.integers(2, 8)); P = rng.uniform(0, 1, (n, n)) ** 2 + 0.01; P /= P.sum(1, keepdims=True)
    w, VL = np.linalg.eig(P.T); k = np.argmin(np.abs(w - 1)); ref = np.real(VL[:, k]); ref /= ref.sum()
    e["est"] = max(e["est"], np.max(np.abs(estacionaria(P) - ref)), np.max(np.abs(estacionaria(P, "lineal") - ref)))
for _ in range(100):
    n, p = int(rng.integers(20, 200)), int(rng.integers(2, 7))
    X = rng.normal(size=(n, p)) @ rng.normal(size=(p, p)) + rng.normal(size=p) * 5
    comp, var, mu = pca(X); sk = PCA().fit(X)
    e["pca"] = max(e["pca"], np.max(np.abs(var - sk.explained_variance_)) / var[0])
    e["pcav"] = max(e["pcav"], max(mismo_vector(comp[k], sk.components_[k]) for k in range(p)))
    e["evr"] = max(e["evr"], np.max(np.abs(var / var.sum() - sk.explained_variance_ratio_)))

print(f"eig2 (traza y determinante) vs np.linalg.eigvals, 300 matrices 2×2   {e['eig2']:.1e}")
print(f"potencia: λ dominante vs eigh (relativo)                             {e['pot']:.1e}")
print(f"potencia: autovector vs eigh (salvo signo)                           {e['potv']:.1e}")
print(f"potencia_inversa desplazada: λ más cercano a μ                        {e['inv']:.1e}")
print(f"deflacion (todos los λ) vs eigh                                      {e['def']:.1e}")
print(f"jacobi: λ vs eigh | autovectores (salvo signo)                       {e['jac']:.1e} | {e['jacv']:.1e}")
print(f"qr_simetrica: λ vs eigh | autovectores (salvo signo)                 {e['qr']:.1e} | {e['qrv']:.1e}")
print(f"V diag(λ) Vᵀ − S y VᵀV − I (Jacobi)                                  {e['rec']:.1e}")
print(f"estacionaria (potencia y lineal) vs autovector de Pᵀ con λ = 1        {e['est']:.1e}")
print(f"pca: varianzas vs sklearn explained_variance_ (relativo)             {e['pca']:.1e}")
print(f"pca: componentes vs sklearn components_ (salvo signo)                {e['pcav']:.1e}")
print(f"pca: proporción explicada vs explained_variance_ratio_               {e['evr']:.1e}")
