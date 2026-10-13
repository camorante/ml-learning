"""Autovalores y autovectores desde cero: 2×2 cerrada, potencia, inversa desplazada, deflación, Jacobi, QR, Markov y PCA."""
import numpy as np

def eig2(A):
    """Autovalores de una 2×2 por traza y determinante: λ = t/2 ± √(t²/4 − d). Complejos si el discriminante es < 0."""
    t = A[0, 0] + A[1, 1]; d = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
    disc = t * t / 4 - d
    r = np.sqrt(complex(disc)) if disc < 0 else np.sqrt(disc)
    return np.array([t / 2 + r, t / 2 - r])

def potencia(A, x0=None, tol=1e-12, max_iter=10_000):
    """Método de la potencia: x ← Ax/‖Ax‖. Devuelve (λ dominante, autovector unitario, iteraciones).
    Converge si |λ₁| > |λ₂|, con error en dirección ~ |λ₂/λ₁|^k."""
    A = np.asarray(A, float); n = A.shape[0]
    x = np.ones(n) / np.sqrt(n) if x0 is None else np.asarray(x0, float) / np.linalg.norm(x0)
    lam = 0.0
    for k in range(1, max_iter + 1):
        y = A @ x
        lam_nuevo = x @ y                       # cociente de Rayleigh (x unitario)
        x_nuevo = y / np.linalg.norm(y)
        if x_nuevo @ x < 0:                     # λ negativo: el vector cambia de sentido cada paso
            x_nuevo = -x_nuevo
        if abs(lam_nuevo - lam) <= tol * abs(lam_nuevo) and np.linalg.norm(x_nuevo - x) < np.sqrt(tol):
            return lam_nuevo, x_nuevo, k
        x, lam = x_nuevo, lam_nuevo
    return lam, x, max_iter

def potencia_inversa(A, mu, x0=None, tol=1e-12, max_iter=500):
    """Iteración inversa desplazada: potencia sobre (A − μI)⁻¹. Encuentra el autovalor más cercano a μ.
    Se factoriza una vez (aquí con solve en cada paso para no depender de scipy)."""
    A = np.asarray(A, float); n = A.shape[0]; B = A - mu * np.eye(n)
    x = np.ones(n) / np.sqrt(n) if x0 is None else np.asarray(x0, float) / np.linalg.norm(x0)
    lam = mu
    for k in range(1, max_iter + 1):
        y = np.linalg.solve(B, x); x_nuevo = y / np.linalg.norm(y)
        lam_nuevo = x_nuevo @ A @ x_nuevo
        if abs(lam_nuevo - lam) <= tol * max(1.0, abs(lam_nuevo)):
            return lam_nuevo, x_nuevo, k
        x, lam = x_nuevo, lam_nuevo
    return lam, x, max_iter

def deflacion(S, k=None, tol=1e-13):
    """Autopares de una simétrica: potencia + deflación de Hotelling S ← S − λ v vᵀ.
    Ordena por |λ| (lo que encuentra la potencia). Didáctico: el error se acumula de un par al siguiente."""
    S = np.array(S, float); n = S.shape[0]; k = n if k is None else k
    vals, vecs = [], []
    rng = np.random.default_rng(0)
    for _ in range(k):
        lam, v, _ = potencia(S, rng.normal(size=n), tol=tol)
        vals.append(lam); vecs.append(v)
        S = S - lam * np.outer(v, v)
    return np.array(vals), np.array(vecs).T

def jacobi(S, tol=1e-14, max_barridos=50):
    """Jacobi cíclico para simétricas: rotaciones de Givens que anulan cada a_pq. Devuelve (λ, V) con S = V diag(λ) Vᵀ."""
    A = np.array(S, float); n = A.shape[0]; V = np.eye(n)
    for _ in range(max_barridos):
        off = np.sqrt(np.sum(np.tril(A, -1) ** 2))
        if off < tol * np.linalg.norm(A):
            break
        for p in range(n - 1):
            for q in range(p + 1, n):
                if abs(A[p, q]) < 1e-300:
                    continue
                theta = (A[q, q] - A[p, p]) / (2 * A[p, q])
                t = np.sign(theta) / (abs(theta) + np.sqrt(theta * theta + 1)) if theta != 0 else 1.0
                c = 1 / np.sqrt(t * t + 1); s = t * c
                G = np.eye(n); G[p, p] = G[q, q] = c; G[p, q] = s; G[q, p] = -s
                A = G.T @ A @ G; V = V @ G          # O(n³) por rotación: claro, no eficiente
    lam = np.diag(A).copy(); idx = np.argsort(lam)[::-1]
    return lam[idx], V[:, idx]

def qr_simetrica(S, tol=1e-13, max_iter=10_000):
    """Iteración QR con desplazamiento de Wilkinson y deflación, para simétricas pequeñas.
    A ← RQ + μI conserva los autovalores (es Qᵀ(A−μI)Q + μI) y empuja la última fila a (0, …, 0, λ)."""
    A = np.array(S, float); n = A.shape[0]; V = np.eye(n); m = n; it = 0
    while m > 1 and it < max_iter:
        it += 1
        a, b, c = A[m - 2, m - 2], A[m - 2, m - 1], A[m - 1, m - 1]
        d = (a - c) / 2; sg = 1.0 if d >= 0 else -1.0
        mu = c - sg * b * b / (abs(d) + np.hypot(d, b)) if b != 0 else c   # autovalor de la 2×2 final más cercano a c
        Q, R = np.linalg.qr(A[:m, :m] - mu * np.eye(m))
        A[:m, :m] = R @ Q + mu * np.eye(m)
        A[:m, m:] = Q.T @ A[:m, m:]; A[m:, :m] = A[:m, m:].T
        V[:, :m] = V[:, :m] @ Q
        if abs(A[m - 1, m - 2]) < tol * (abs(A[m - 1, m - 1]) + abs(A[m - 2, m - 2])):
            m -= 1
    lam = np.diag(A).copy(); idx = np.argsort(lam)[::-1]
    return lam[idx], V[:, idx], it

def estacionaria(P, metodo="potencia", tol=1e-14, max_iter=100_000):
    """Distribución estacionaria de una cadena de Markov (P por filas: P[i, j] = Pr(i → j)).
    π es autovector por la izquierda con λ = 1: πP = π, Σπ = 1."""
    P = np.asarray(P, float); n = P.shape[0]
    if metodo == "lineal":                       # (Pᵀ − I)π = 0 con una ecuación cambiada por Σπ = 1
        M = P.T - np.eye(n); M[-1, :] = 1.0
        b = np.zeros(n); b[-1] = 1.0
        return np.linalg.solve(M, b)
    pi = np.ones(n) / n
    for _ in range(max_iter):
        nuevo = pi @ P
        if np.abs(nuevo - pi).sum() < tol:
            return nuevo / nuevo.sum()
        pi = nuevo
    return pi / pi.sum()

def tiempo_mezcla(P, eps=0.01):
    """Pasos hasta que, empiece donde empiece, la distribución está a distancia de variación total < eps de π."""
    P = np.asarray(P, float); pi = estacionaria(P, "lineal"); Pk = np.eye(len(P))
    for k in range(1, 100_000):
        Pk = Pk @ P
        if 0.5 * np.abs(Pk - pi).sum(axis=1).max() < eps:
            return k
    return None

def pca(X, k=None):
    """PCA por autovectores de la covarianza (simétrica): Jacobi. Devuelve (componentes por filas, varianzas, media).
    Signo fijado para que la mayor componente de cada vector sea positiva (los autovectores no tienen signo propio)."""
    X = np.asarray(X, float); mu = X.mean(0); Xc = X - mu
    C = Xc.T @ Xc / (len(X) - 1)
    lam, V = jacobi(C)
    k = len(lam) if k is None else k
    V = V[:, :k] * np.sign(V[np.argmax(np.abs(V[:, :k]), axis=0), np.arange(k)])
    return V.T, lam[:k], mu
