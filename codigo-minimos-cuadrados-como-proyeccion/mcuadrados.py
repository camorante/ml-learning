"""Mínimos cuadrados desde cero: ecuaciones normales, QR (Householder), SVD y ridge."""
import numpy as np

def con_intercepto(X):
    """Añade una columna de unos delante (el término independiente)."""
    X = np.asarray(X, float)
    return np.c_[np.ones(len(X)), X]

def normales(X, y):
    """Resuelve XᵀX β = Xᵀy. Rápido, pero eleva al cuadrado el número de condición."""
    X = np.asarray(X, float); y = np.asarray(y, float)
    return np.linalg.solve(X.T @ X, X.T @ y)

def householder_qr(X):
    """QR reducida con reflexiones de Householder: X = Q R, Q (n×p) ortonormal, R (p×p) triangular superior."""
    A = np.array(X, float); n, p = A.shape
    Q = np.eye(n)
    for k in range(p):
        x = A[k:, k]
        v = x.copy(); s = 1.0 if x[0] >= 0 else -1.0
        v[0] += s * np.linalg.norm(x)                  # v = x + signo(x₁)‖x‖e₁ (evita cancelación)
        nv = np.linalg.norm(v)
        if nv == 0:
            continue
        v /= nv
        A[k:, :] -= 2.0 * np.outer(v, v @ A[k:, :])    # H = I − 2vvᵀ aplicada a la submatriz
        Q[:, k:] -= 2.0 * np.outer(Q[:, k:] @ v, v)    # acumula Q = H₁ H₂ … H_p
    return Q[:, :p], np.triu(A[:p, :])

def resolver_triangular(R, b):
    """Sustitución hacia atrás para R β = b."""
    p = len(b); beta = np.zeros(p)
    for i in range(p - 1, -1, -1):
        beta[i] = (b[i] - R[i, i + 1:] @ beta[i + 1:]) / R[i, i]
    return beta

def mc_qr(X, y):
    """β = R⁻¹ Qᵀ y: no forma XᵀX, así que el error depende de cond(X), no de cond(X)²."""
    Q, R = householder_qr(X)
    return resolver_triangular(R, Q.T @ np.asarray(y, float))

def mc_svd(X, y, rcond=None):
    """Pseudoinversa: β = V Σ⁺ Uᵀ y. Descarta valores singulares < rcond·σ_max (solución de norma mínima)."""
    X = np.asarray(X, float); U, s, Vt = np.linalg.svd(X, full_matrices=False)
    rcond = np.finfo(float).eps * max(X.shape) if rcond is None else rcond
    keep = s > rcond * s[0]
    return Vt[keep].T @ ((U[:, keep].T @ np.asarray(y, float)) / s[keep])

def ridge(X, y, lam, intercepto=True):
    """Ridge: minimiza ‖y − b₀ − Xβ‖² + λ‖β‖² (sin penalizar el intercepto). Devuelve (b₀, β)."""
    X = np.asarray(X, float); y = np.asarray(y, float)
    if intercepto:
        mx, my = X.mean(0), y.mean(); Xc, yc = X - mx, y - my
    else:
        mx, my, Xc, yc = np.zeros(X.shape[1]), 0.0, X, y
    # como mínimos cuadrados aumentados: [Xc; √λ I] β ≈ [yc; 0], resuelto por QR
    A = np.r_[Xc, np.sqrt(lam) * np.eye(X.shape[1])]; b = np.r_[yc, np.zeros(X.shape[1])]
    beta = mc_qr(A, b)
    return my - mx @ beta, beta

def sombrero(X):
    """H = X (XᵀX)⁻¹ Xᵀ = Q Qᵀ. Proyecta y sobre el espacio columna de X."""
    Q, _ = householder_qr(X)
    return Q @ Q.T

def r2(y, yhat):
    y = np.asarray(y, float); r = y - yhat; d = y - y.mean()
    return 1.0 - (r @ r) / (d @ d)

def vif(X):
    """Factor de inflación de la varianza de cada columna: 1 / (1 − R²_j), con R²_j de regresar x_j sobre las demás."""
    X = np.asarray(X, float); out = []
    for j in range(X.shape[1]):
        otras = con_intercepto(np.delete(X, j, axis=1))
        b = mc_qr(otras, X[:, j]); out.append(1.0 / (1.0 - r2(X[:, j], otras @ b)))
    return np.array(out)

def ponderados(X, y, w):
    """Mínimos cuadrados ponderados: minimiza Σ wᵢ (yᵢ − xᵢβ)². Equivale a escalar cada fila por √wᵢ."""
    sw = np.sqrt(np.asarray(w, float))
    return mc_qr(np.asarray(X, float) * sw[:, None], np.asarray(y, float) * sw)
