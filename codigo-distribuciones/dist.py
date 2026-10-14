"""Distribuciones desde cero: normal, binomial, Poisson, exponencial y lognormal.
pmf/pdf, cdf, cuantiles, muestreo y ajuste por momentos, solo con math y numpy."""
import math
import numpy as np


# ---------- normal ----------
def norm_pdf(x, mu=0.0, sigma=1.0):
    z = (np.asarray(x, float) - mu) / sigma
    return np.exp(-0.5 * z * z) / (sigma * math.sqrt(2 * math.pi))


def norm_cdf(x, mu=0.0, sigma=1.0):
    z = (np.asarray(x, float) - mu) / (sigma * math.sqrt(2))
    return 0.5 * np.vectorize(math.erfc)(-z)                 # erfc evita restar 1 − algo en la cola izquierda


def norm_ppf(q, mu=0.0, sigma=1.0):
    """Cuantil: bisección sobre Φ(z) = q en [−40, 40] (60 pasos) y dos pasos de Newton para pulir."""
    q = np.asarray(q, float)
    lo, hi = np.full(q.shape, -40.0), np.full(q.shape, 40.0)
    for _ in range(60):
        mid = (lo + hi) / 2
        bajo = norm_cdf(mid) < q
        lo, hi = np.where(bajo, mid, lo), np.where(bajo, hi, mid)
    z = (lo + hi) / 2
    for _ in range(2):
        z = z - (norm_cdf(z) - q) / norm_pdf(z)
    return mu + sigma * z


def norm_rvs(n, mu=0.0, sigma=1.0, rng=None):
    """Box–Muller: dos uniformes dan dos normales independientes."""
    rng = rng or np.random.default_rng()
    m = (n + 1) // 2
    u1, u2 = 1 - rng.random(m), rng.random(m)                # 1 − U ∈ (0, 1]: evita log(0)
    r = np.sqrt(-2 * np.log(u1))
    z = np.concatenate([r * np.cos(2 * np.pi * u2), r * np.sin(2 * np.pi * u2)])[:n]
    return mu + sigma * z


# ---------- binomial ----------
def log_comb(n, k):
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def binom_pmf(k, n, p):
    k = np.atleast_1d(np.asarray(k))
    out = np.zeros(k.shape)
    for i, kk in enumerate(k):
        if 0 <= kk <= n:
            if p in (0.0, 1.0):
                out[i] = float(kk == (n if p == 1.0 else 0))
            else:
                out[i] = math.exp(log_comb(n, kk) + kk * math.log(p) + (n - kk) * math.log1p(-p))
    return out


def binom_cdf(k, n, p):
    return np.array([binom_pmf(np.arange(0, int(math.floor(kk)) + 1), n, p).sum() if kk >= 0 else 0.0
                     for kk in np.atleast_1d(k)])


def binom_rvs(size, n, p, rng=None):
    """Suma de n ensayos de Bernoulli (uniforme < p): directo y exacto, O(n) por muestra."""
    rng = rng or np.random.default_rng()
    return (rng.random((size, n)) < p).sum(axis=1)


# ---------- Poisson ----------
def pois_pmf(k, lam):
    k = np.atleast_1d(np.asarray(k, float))
    out = np.where(k >= 0, np.exp(k * math.log(lam) - lam - np.vectorize(math.lgamma)(np.maximum(k, 0) + 1)), 0.0)
    return out


def pois_cdf(k, lam):
    return np.array([pois_pmf(np.arange(0, int(math.floor(kk)) + 1), lam).sum() if kk >= 0 else 0.0
                     for kk in np.atleast_1d(k)])


def pois_rvs(size, lam, rng=None):
    """Inversión secuencial: recorre k = 0, 1, 2… acumulando la pmf hasta pasar una uniforme."""
    rng = rng or np.random.default_rng()
    u = rng.random(size)
    out = np.zeros(size, int)
    p = math.exp(-lam); F = p; k = 0
    pend = np.ones(size, bool)
    while pend.any():
        hit = pend & (u <= F)
        out[hit] = k; pend &= ~hit
        k += 1; p *= lam / k; F += p
        if k > lam + 40 * math.sqrt(lam) + 40:                  # red de seguridad numérica
            out[pend] = k; break
    return out


# ---------- exponencial y lognormal ----------
def expon_cdf(x, tasa):
    return np.where(np.asarray(x) > 0, -np.expm1(-tasa * np.asarray(x, float)), 0.0)


def lognorm_pdf(x, mu, s):
    x = np.asarray(x, float)
    xs = np.where(x > 0, x, 1.0)
    return np.where(x > 0, norm_pdf(np.log(xs), mu, s) / xs, 0.0)


def lognorm_cdf(x, mu, s):
    x = np.asarray(x, float)
    return np.where(x > 0, norm_cdf(np.log(np.where(x > 0, x, 1.0)), mu, s), 0.0)


# ---------- ajuste y comprobación ----------
def ajuste_momentos(datos, familia):
    """Parámetros que hacen coincidir la media (y la varianza) teóricas con las de los datos."""
    x = np.asarray(datos, float)
    m, v = x.mean(), x.var(ddof=1)
    if familia == "normal":
        return {"mu": m, "sigma": math.sqrt(v)}
    if familia == "poisson":
        return {"lam": m}
    if familia == "binomial_p":                                # n conocido: datos = éxitos de n intentos
        raise ValueError("usa ajuste_binomial(datos, n)")
    if familia == "exponencial":
        return {"tasa": 1 / m}
    if familia == "lognormal":                                 # E = e^(μ+s²/2), Var = (e^(s²) − 1) E²
        s2 = math.log(1 + v / m**2)
        return {"mu": math.log(m) - s2 / 2, "s": math.sqrt(s2)}
    raise ValueError(familia)


def ajuste_binomial(datos, n):
    return {"n": n, "p": float(np.mean(datos)) / n}


def ajuste_lognormal_mv(datos):
    """Máxima verosimilitud: media y desviación (con n, no n − 1) de los logaritmos."""
    y = np.log(np.asarray(datos, float))
    return {"mu": y.mean(), "s": y.std()}


def qq_puntos(datos, mu=None, sigma=None):
    """Puntos del gráfico QQ normal: cuantiles teóricos en (i − 0.5)/n frente a los datos ordenados."""
    x = np.sort(np.asarray(datos, float)); n = len(x)
    mu = x.mean() if mu is None else mu
    sigma = x.std(ddof=1) if sigma is None else sigma
    return norm_ppf((np.arange(1, n + 1) - 0.5) / n, mu, sigma), x


def tabla_frecuencias(datos, pmf, kmax):
    """Observado frente a esperado por valor 0..kmax−1 y una última casilla «kmax o más»; estadístico χ²."""
    x = np.asarray(datos); n = len(x)
    obs = np.array([np.sum(x == k) for k in range(kmax)] + [np.sum(x >= kmax)], float)
    p = np.array([pmf(k) for k in range(kmax)], float)
    esp = n * np.append(p, 1 - p.sum())
    return obs, esp, float(np.sum((obs - esp) ** 2 / esp))
