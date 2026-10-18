"""Máxima verosimilitud desde cero: log-verosimilitudes, estimadores cerrados,
información de Fisher, intervalos por perfil, razón de verosimilitudes y ajuste numérico."""
import math
import numpy as np
from scipy.optimize import minimize          # solo para el ajuste numérico (§6)


# ---------- utilidades numéricas ----------
def logsumexp(a, axis=None):
    """log(sum(exp(a))) sin desbordar: se saca el máximo fuera."""
    a = np.asarray(a, float); m = np.max(a, axis=axis, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    out = m + np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True))
    return np.squeeze(out, axis=axis) if axis is not None else float(out.squeeze())


def xlogy(x, y):
    """x·log(y) con el convenio 0·log(0) = 0 (hace falta con p̂ = 0 o 1)."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    return np.where(x == 0, 0.0, x * np.log(np.where(x == 0, 1.0, y)))


def log_sigmoid(z):
    """log(1 / (1 + e^(−z))) estable para z muy negativos o muy positivos."""
    z = np.asarray(z, float)
    return np.where(z >= 0, -np.log1p(np.exp(-np.abs(z))), z - np.log1p(np.exp(-np.abs(z))))


_lgam = np.vectorize(math.lgamma)


# ---------- log-verosimilitudes ----------
def ll_poisson(lam, x):
    x = np.asarray(x, float)
    return float(np.sum(xlogy(x, lam) - lam - _lgam(x + 1)))


def ll_binom(p, k, n):
    return float(math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
                 + xlogy(k, p) + xlogy(n - k, 1 - p))


def ll_normal(mu, s2, x):
    x = np.asarray(x, float)
    return float(-0.5 * x.size * math.log(2 * math.pi * s2) - np.sum((x - mu) ** 2) / (2 * s2))


def ll_laplace(m, b, x):
    x = np.asarray(x, float)
    return float(-x.size * math.log(2 * b) - np.sum(np.abs(x - m)) / b)


def ll_expon(tasa, x):
    x = np.asarray(x, float)
    return float(x.size * math.log(tasa) - tasa * x.sum())


def ll_lognormal(mu, s, x):
    lx = np.log(np.asarray(x, float))
    return ll_normal(mu, s * s, lx) - float(lx.sum())          # jacobiano: dx = x · d(log x)


def ll_negbin(r, mu, x):
    """Binomial negativa con media mu y 'tamaño' r (varianza mu + mu²/r)."""
    x = np.asarray(x, float)
    return float(np.sum(_lgam(x + r) - _lgam(r) - _lgam(x + 1)
                        + r * math.log(r / (r + mu)) + xlogy(x, mu / (r + mu))))


def ll_bernoulli(p, y):
    y = np.asarray(y, float)
    return float(np.sum(xlogy(y, p) + xlogy(1 - y, 1 - np.asarray(p, float))))


# ---------- estimadores cerrados (derivada de ℓ igual a 0) ----------
def mle_poisson(x):
    return float(np.mean(x))                                  # ℓ' = Σx/λ − n = 0


def mle_binom(k, n):
    return k / n                                              # ℓ' = k/p − (n−k)/(1−p) = 0


def mle_normal(x):
    x = np.asarray(x, float); mu = float(x.mean())
    return mu, float(np.mean((x - mu) ** 2))                  # σ² con n, no con n − 1 (sesgado)


def mle_expon(x):
    return 1 / float(np.mean(x))                              # tasa; escala = media


def mle_lognormal(x):
    mu, s2 = mle_normal(np.log(x)); return mu, math.sqrt(s2)


def mle_laplace(x):
    x = np.asarray(x, float); m = float(np.median(x))
    return m, float(np.mean(np.abs(x - m)))


# ---------- incertidumbre ----------
def se_poisson(lam, n):
    return math.sqrt(lam / n)                                 # I(λ) = n/λ


def se_binom(p, n):
    return math.sqrt(p * (1 - p) / n)                         # I(p) = n/(p(1−p))


def hessiano(f, th, h=1e-4):
    """Hessiano numérico de f en th por diferencias centrales."""
    th = np.asarray(th, float); k = th.size; H = np.empty((k, k))
    for i in range(k):
        for j in range(k):
            e_i = np.eye(k)[i] * h; e_j = np.eye(k)[j] * h
            H[i, j] = (f(th + e_i + e_j) - f(th + e_i - e_j) - f(th - e_i + e_j) + f(th - e_i - e_j)) / (4 * h * h)
    return H


def errores_estandar(negll, th):
    """Raíz de la diagonal de la inversa de la información observada (hessiano de −ℓ)."""
    return np.sqrt(np.diag(np.linalg.inv(hessiano(negll, th))))


def _biseccion(g, a, b, it=200):
    ga = g(a)
    for _ in range(it):
        m = (a + b) / 2; gm = g(m)
        if (gm > 0) == (ga > 0): a, ga = m, gm
        else: b = m
    return (a + b) / 2


def intervalo_perfil(ll, th_hat, lo, hi, nivel=0.95):
    """Todos los θ con ℓ(θ) ≥ ℓ(θ̂) − c/2, con c el cuantil χ²₁ (3,841 para el 95 %)."""
    z = _cuantil_normal((1 + nivel) / 2); c = z * z
    g = lambda t: ll(t) - (ll(th_hat) - c / 2)
    return _biseccion(g, lo, th_hat), _biseccion(g, th_hat, hi)


def _cuantil_normal(q):
    return _biseccion(lambda z: 0.5 * math.erfc(-z / math.sqrt(2)) - q, -10, 10)


def razon_verosimilitud(ll0, ll1):
    """Estadístico 2·(ℓ1 − ℓ0) y su p-valor con 1 grado de libertad (χ²₁ = Z²)."""
    d = max(0.0, 2 * (ll1 - ll0)); return d, math.erfc(math.sqrt(d / 2))


def aic(ll, k):
    return 2 * k - 2 * ll


# ---------- ajuste numérico ----------
def ajustar(negll, th0, metodo="L-BFGS-B"):
    """Minimiza −ℓ. Parámetros positivos: pasarlos en logaritmo dentro de negll."""
    r = minimize(negll, np.asarray(th0, float), method=metodo)
    return r.x, -r.fun, r


def ajustar_negbin(x):
    x = np.asarray(x, float)
    negll = lambda t: -ll_negbin(math.exp(t[0]), math.exp(t[1]), x)
    t, ll, _ = ajustar(negll, [0.0, math.log(x.mean())])
    return math.exp(t[0]), math.exp(t[1]), ll


def logistica(X, y, it=50, l2=0.0):
    """Regresión logística por Newton-Raphson sobre ℓ (con penalización l2/2·|w|² si l2 > 0,
    sin penalizar el intercepto). Devuelve [intercepto, pesos...]."""
    X = np.c_[np.ones(len(X)), np.asarray(X, float)]; y = np.asarray(y, float); w = np.zeros(X.shape[1])
    P = np.eye(X.shape[1]) * l2; P[0, 0] = 0
    for _ in range(it):
        p = 1 / (1 + np.exp(-X @ w))
        g = X.T @ (y - p) - P @ w                             # gradiente de ℓ
        H = X.T @ (X * (p * (1 - p))[:, None]) + P            # información observada
        paso = np.linalg.solve(H, g); w = w + paso
        if np.max(np.abs(paso)) < 1e-12: break
    return w


def ll_logistica(w, X, y):
    z = np.c_[np.ones(len(X)), np.asarray(X, float)] @ w; y = np.asarray(y, float)
    return float(np.sum(y * log_sigmoid(z) + (1 - y) * log_sigmoid(-z)))


def map_binom(k, n, m, p0):
    """Máximo de ℓ + log prior Beta(1 + m·p0, 1 + m·(1 − p0)): como sumar m casos imaginarios."""
    return (k + m * p0) / (n + m)
