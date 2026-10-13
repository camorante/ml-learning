"""SVD desde cero: fórmula cerrada 2×2, Jacobi de un lado, potencia con deflación,
versión aleatorizada, pseudoinversa y aproximación de rango bajo."""
import numpy as np


def svd_2x2(A):
    """A = U diag(s) Vᵀ para A de 2×2, con el ángulo de AᵀA en forma cerrada."""
    A = np.asarray(A, float)
    G = A.T @ A
    teta = 0.5 * np.arctan2(2 * G[0, 1], G[0, 0] - G[1, 1])      # dirección de v₁
    V = np.array([[np.cos(teta), -np.sin(teta)], [np.sin(teta), np.cos(teta)]])
    W = A @ V                                                     # columnas ortogonales: σ₁u₁, σ₂u₂
    s = np.linalg.norm(W, axis=0)
    U = np.zeros((2, 2))
    U[:, 0] = W[:, 0] / s[0] if s[0] > 0 else [1.0, 0.0]
    U[:, 1] = [-U[1, 0], U[0, 0]]                                 # completa a base ortonormal
    if s[1] > 0 and U[:, 1] @ W[:, 1] < 0:                        # el signo va a U (reflejo), no a σ₂
        U[:, 1] *= -1
    return U, s, V.T


def _completar(U, r):
    """Completa las columnas r.. de U a una base ortonormal (Gram–Schmidt con la base canónica)."""
    m = U.shape[0]
    for k in range(r, U.shape[1]):
        for e in np.eye(m):
            w = e - U[:, :k] @ (U[:, :k].T @ e)
            w = w - U[:, :k] @ (U[:, :k].T @ w)                    # segunda pasada: reortogonaliza
            if np.linalg.norm(w) > 1e-8:
                U[:, k] = w / np.linalg.norm(w)
                break
    return U


def svd_jacobi(A, tol=1e-15, max_barridos=60):
    """SVD reducida por Jacobi de un lado (Hestenes): gira pares de columnas hasta que son ortogonales.
    Devuelve U (m×p), s (p), Vt (p×n) con p = min(m, n) y s en orden decreciente."""
    A = np.asarray(A, float)
    m, n = A.shape
    if m < n:                                                     # trabaja con la forma alta
        U, s, Vt = svd_jacobi(A.T, tol, max_barridos)
        return Vt.T, s, U.T
    W = A.copy(); V = np.eye(n)
    for _ in range(max_barridos):
        quieto = True
        for p in range(n - 1):
            for q in range(p + 1, n):
                a, b = W[:, p] @ W[:, p], W[:, q] @ W[:, q]
                c = W[:, p] @ W[:, q]
                if abs(c) <= tol * np.sqrt(a * b) or c == 0:
                    continue
                quieto = False
                z = (b - a) / (2 * c)                                 # rotación que anula el producto punto
                t = np.sign(z) / (abs(z) + np.sqrt(1 + z * z)) if z != 0 else 1.0
                cs = 1 / np.sqrt(1 + t * t); sn = cs * t
                Wp, Wq = W[:, p].copy(), W[:, q].copy()
                W[:, p], W[:, q] = cs * Wp - sn * Wq, sn * Wp + cs * Wq
                Vp, Vq = V[:, p].copy(), V[:, q].copy()
                V[:, p], V[:, q] = cs * Vp - sn * Vq, sn * Vp + cs * Vq
        if quieto:
            break
    s = np.linalg.norm(W, axis=0)
    o = np.argsort(-s, kind="stable"); s, W, V = s[o], W[:, o], V[:, o]
    r = int(np.sum(s > max(m, n) * np.finfo(float).eps * (s[0] if len(s) else 0)))
    U = np.zeros((m, n)); U[:, :r] = W[:, :r] / s[:r]
    s[r:] = 0.0
    return _completar(U, r), s, V.T


def potencia_deflacion(A, k, iters=500, tol=1e-13, seed=0):
    """Las k primeras capas por el método de la potencia sobre AᵀA, restando cada capa encontrada."""
    rng = np.random.default_rng(seed)
    R = np.array(A, float); m, n = R.shape
    U, s, Vt = np.zeros((m, k)), np.zeros(k), np.zeros((k, n))
    for j in range(k):
        v = rng.normal(size=n); v /= np.linalg.norm(v); sig = 0.0
        for _ in range(iters):
            w = R.T @ (R @ v)                                     # un paso de potencia con AᵀA, sin formarla
            nw = np.linalg.norm(w)
            if nw == 0:
                break
            v_nuevo = w / nw
            if np.linalg.norm(v_nuevo - v) < tol:
                v = v_nuevo; break
            v = v_nuevo
        u = R @ v; sig = np.linalg.norm(u)
        u = u / sig if sig > 0 else u
        U[:, j], s[j], Vt[j] = u, sig, v
        R -= sig * np.outer(u, v)                                 # deflación: quita la capa
    return U, s, Vt


def svd_aleatorizada(A, k, sobremuestreo=10, potencias=2, seed=0):
    """Halko, Martinsson y Tropp (2011): proyecta sobre k + p direcciones al azar, ortonormaliza y
    hace la SVD exacta de una matriz pequeña."""
    rng = np.random.default_rng(seed)
    A = np.asarray(A, float)
    Q, _ = np.linalg.qr(A @ rng.normal(size=(A.shape[1], k + sobremuestreo)))
    for _ in range(potencias):                                    # iteraciones de subespacio: separan mejor los σ
        Q, _ = np.linalg.qr(A.T @ Q)
        Q, _ = np.linalg.qr(A @ Q)
    Ub, s, Vt = svd_jacobi(Q.T @ A)                               # (k + p) × n: pequeña
    return (Q @ Ub)[:, :k], s[:k], Vt[:k]


def rango(A, rtol=None):
    _, s, _ = svd_jacobi(A)
    rtol = max(np.shape(A)) * np.finfo(float).eps if rtol is None else rtol
    return int(np.sum(s > rtol * s[0])) if s[0] > 0 else 0


def pinv(A, rtol=None):
    """Pseudoinversa A⁺ = V Σ⁺ Uᵀ: invierte los σ grandes y pone a 0 los demás."""
    U, s, Vt = svd_jacobi(A)
    rtol = max(np.shape(A)) * np.finfo(float).eps if rtol is None else rtol
    keep = s > rtol * s[0] if s[0] > 0 else np.zeros_like(s, bool)
    return (Vt[keep].T / s[keep]) @ U[:, keep].T


def rango_k(A, k):
    """Mejor aproximación de rango k (Eckart–Young): suma de las k primeras capas σᵢ uᵢ vᵢᵀ."""
    U, s, Vt = svd_jacobi(A)
    return (U[:, :k] * s[:k]) @ Vt[:k]


def normas(A):
    """Norma 2 (σ₁), Frobenius (raíz de la suma de σ²) y número de condición σ₁/σ_min."""
    _, s, _ = svd_jacobi(A)
    return s[0], np.sqrt(np.sum(s**2)), (s[0] / s[-1] if s[-1] > 0 else np.inf)
