"""Detección de atípicos y regresión robusta desde cero."""
import numpy as np

Q75 = 0.6744897501960817            # cuantil 0.75 de la normal estándar

def z_clasico(x):
    x = np.asarray(x, float); return (x - x.mean()) / x.std(ddof=1)

def mad(x):
    x = np.asarray(x, float); return np.median(np.abs(x - np.median(x)))

def z_robusto(x):
    """(x - mediana) / (MAD / 0.6745): en datos normales vale lo mismo que el z clásico."""
    x = np.asarray(x, float); return (x - np.median(x)) / (mad(x) / Q75)

def vallas_iqr(x, k=1.5):
    q1, q3 = np.percentile(x, [25, 75]); r = q3 - q1
    return q1 - k * r, q3 + k * r

def ols(X, y):
    A = np.column_stack([np.ones(len(X)), X]); return np.linalg.lstsq(A, y, rcond=None)[0]

def huber_irls(X, y, t=1.345, tol=1e-10, max_iter=200):
    """Regresión de Huber por mínimos cuadrados reponderados.
    Escala: mediana de |residuos| / 0.6745 (los residuos ya están centrados en 0), como statsmodels."""
    A = np.column_stack([np.ones(len(X)), np.asarray(X, float)]); y = np.asarray(y, float)
    b = np.linalg.lstsq(A, y, rcond=None)[0]
    for _ in range(max_iter):
        r = y - A @ b; s = np.median(np.abs(r)) / Q75
        u = np.abs(r / s); w = np.where(u <= t, 1.0, t / np.maximum(u, 1e-300))
        sw = np.sqrt(w); b_new = np.linalg.lstsq(A * sw[:, None], y * sw, rcond=None)[0]
        if np.max(np.abs(b_new - b)) < tol: b = b_new; break
        b = b_new
    return b, w

def cook(X, y):
    """Distancia de Cook de cada fila en una regresión lineal por mínimos cuadrados."""
    A = np.column_stack([np.ones(len(X)), np.asarray(X, float)]); y = np.asarray(y, float); n, p = A.shape
    H = A @ np.linalg.pinv(A.T @ A) @ A.T; h = np.diag(H)
    e = y - H @ y; s2 = e @ e / (n - p)
    return e**2 / (p * s2) * h / (1 - h) ** 2, h

def mahalanobis(X, centro, cov):
    D = np.asarray(X, float) - centro; return np.sqrt(np.einsum("ij,jk,ik->i", D, np.linalg.inv(cov), D))

# ---------- Isolation Forest ----------
def c_n(n):
    """Longitud media de un camino sin éxito en un árbol binario de búsqueda con n puntos."""
    n = np.asarray(n, float); m = np.maximum(n, 3)
    return np.where(n > 2, 2 * (np.log(m - 1) + np.euler_gamma) - 2 * (m - 1) / m, np.where(n == 2, 1.0, 0.0))

def _arbol(X, rng, prof, lim):
    n = len(X)
    if prof >= lim or n <= 1 or np.all(X.max(0) == X.min(0)):
        return ("hoja", n)
    j = rng.choice(np.where(X.max(0) > X.min(0))[0]); lo, hi = X[:, j].min(), X[:, j].max()
    c = rng.uniform(lo, hi); m = X[:, j] < c
    return ("nodo", j, c, _arbol(X[m], rng, prof + 1, lim), _arbol(X[~m], rng, prof + 1, lim))

def _camino(x, nodo, prof=0):
    if nodo[0] == "hoja": return prof + float(c_n(nodo[1]))
    _, j, c, iz, de = nodo
    return _camino(x, iz if x[j] < c else de, prof + 1)

def isolation_forest(X, n_arboles=200, muestra=256, seed=0):
    """Puntuación s = 2^(-E[h(x)] / c(psi)): cerca de 1, fácil de aislar (atípico); cerca de 0.5 o menos, normal."""
    X = np.asarray(X, float); rng = np.random.default_rng(seed); psi = min(muestra, len(X))
    lim = int(np.ceil(np.log2(psi))); h = np.zeros(len(X))
    for _ in range(n_arboles):
        t = _arbol(X[rng.choice(len(X), psi, replace=False)], rng, 0, lim)
        h += np.array([_camino(x, t) for x in X])
    return 2 ** (-(h / n_arboles) / float(c_n(psi)))
