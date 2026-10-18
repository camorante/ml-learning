"""Intervalos de confianza y bootstrap desde cero (solo numpy y la librería estándar)."""
import math
from statistics import NormalDist
import numpy as np

N01 = NormalDist()


def _betacf(a, b, x, it=300, eps=1e-15):
    """Fracción continua de la beta incompleta (Numerical Recipes, §6.4)."""
    qab, qap, qam = a + b, a + 1, a - 1
    c, d = 1.0, 1 - qab * x / qap
    d = 1 / (d if abs(d) > 1e-300 else 1e-300); h = d
    for m in range(1, it + 1):
        m2 = 2 * m
        for aa in (m * (b - m) * x / ((qam + m2) * (a + m2)), -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))):
            d = 1 + aa * d; d = 1 / (d if abs(d) > 1e-300 else 1e-300)
            c = 1 + aa / c; c = c if abs(c) > 1e-300 else 1e-300
            h *= d * c
        if abs(d * c - 1) < eps:
            break
    return h


def beta_inc(a, b, x):
    """Beta incompleta regularizada I_x(a, b)."""
    if x <= 0: return 0.0
    if x >= 1: return 1.0
    lb = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x)
    if x < (a + 1) / (a + b + 2):
        return math.exp(lb) * _betacf(a, b, x) / a
    return 1 - math.exp(lb) * _betacf(b, a, 1 - x) / b


def _biseccion(f, lo, hi, it=200):
    """Raíz de una función creciente en [lo, hi]."""
    for _ in range(it):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if f(mid) < 0 else (lo, mid)
    return (lo + hi) / 2


def t_cdf(t, df):
    """P(T ≤ t) para una t de Student con df grados de libertad."""
    p = 0.5 * beta_inc(df / 2, 0.5, df / (df + t * t))
    return 1 - p if t > 0 else p


def t_ppf(q, df):
    """Cuantil de la t de Student (bisección sobre t_cdf)."""
    return _biseccion(lambda t: t_cdf(t, df) - q, -1e4, 1e4)


def ic_media(x, conf=0.95, sigma=None):
    """Media ± multiplicador · error estándar. Con sigma conocida, z; si no, t con n − 1 g.l."""
    x = np.asarray(x, float); n = len(x); m = x.mean()
    if sigma is not None:
        h = N01.inv_cdf((1 + conf) / 2) * sigma / math.sqrt(n)
    else:
        h = t_ppf((1 + conf) / 2, n - 1) * x.std(ddof=1) / math.sqrt(n)
    return m - h, m + h


def ic_prediccion(x, conf=0.95):
    """Dónde caerá UN trayecto nuevo: media ± t · s · sqrt(1 + 1/n)."""
    x = np.asarray(x, float); n = len(x)
    h = t_ppf((1 + conf) / 2, n - 1) * x.std(ddof=1) * math.sqrt(1 + 1 / n)
    return x.mean() - h, x.mean() + h


def ic_wald(k, n, conf=0.95):
    p = k / n; h = N01.inv_cdf((1 + conf) / 2) * math.sqrt(p * (1 - p) / n)
    return p - h, p + h


def ic_wilson(k, n, conf=0.95):
    """Los p para los que |p̂ − p| ≤ z·sqrt(p(1 − p)/n): una ecuación de segundo grado en p."""
    z = N01.inv_cdf((1 + conf) / 2); p = k / n; z2 = z * z
    c = (p + z2 / (2 * n)) / (1 + z2 / n)
    h = z / (1 + z2 / n) * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n))
    return c - h, c + h


def binom_cdf(k, n, p):
    """P(X ≤ k) con X ~ Binomial(n, p) = I_{1−p}(n − k, k + 1)."""
    if k < 0: return 0.0
    if k >= n: return 1.0
    return beta_inc(n - k, k + 1, 1 - p)


def ic_clopper_pearson(k, n, conf=0.95):
    """«Exacto»: los p para los que lo observado no es demasiado raro en ninguna de las dos colas."""
    a = (1 - conf) / 2
    lo = 0.0 if k == 0 else _biseccion(lambda p: (1 - binom_cdf(k - 1, n, p)) - a, 0, 1)
    hi = 1.0 if k == n else _biseccion(lambda p: a - binom_cdf(k, n, p), 0, 1)
    return lo, hi


def ic_diferencia(x, y, conf=0.95):
    """Diferencia de medias (x − y) con Welch: varianzas distintas y g.l. de Welch-Satterthwaite."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    vx, vy = x.var(ddof=1) / len(x), y.var(ddof=1) / len(y)
    gl = (vx + vy) ** 2 / (vx ** 2 / (len(x) - 1) + vy ** 2 / (len(y) - 1))
    d = x.mean() - y.mean(); h = t_ppf((1 + conf) / 2, gl) * math.sqrt(vx + vy)
    return d - h, d + h


def _cuantil(v, q):
    """Cuantil lineal (el de np.quantile por defecto), escrito a mano."""
    v = np.sort(v); pos = q * (len(v) - 1); i = int(math.floor(pos)); f = pos - i
    return v[i] if i + 1 >= len(v) else v[i] + f * (v[i + 1] - v[i])


def remuestreos(x, B, seed):
    """B remuestreos con reemplazo: los mismos índices que usa scipy.stats.bootstrap con esa semilla."""
    x = np.asarray(x, float)
    return x[np.random.default_rng(seed).integers(0, len(x), (B, len(x)))]


def bootstrap(x, est, B=9999, conf=0.95, metodo="percentil", seed=0):
    """Intervalo bootstrap de est (una función f(muestras, axis)) con el método percentil, básico o BCa."""
    x = np.asarray(x, float); theta = est(x, axis=-1)
    tb = est(remuestreos(x, B, seed), axis=-1); a = (1 - conf) / 2
    if metodo == "percentil":
        return _cuantil(tb, a), _cuantil(tb, 1 - a)
    if metodo == "basico":
        return 2 * theta - _cuantil(tb, 1 - a), 2 * theta - _cuantil(tb, a)
    # BCa: z0 corrige el sesgo (qué fracción de remuestreos queda por debajo) y a la asimetría (navaja)
    z0 = N01.inv_cdf((np.sum(tb < theta) + np.sum(tb <= theta)) / (2 * B))
    jk = np.array([est(np.delete(x, i), axis=-1) for i in range(len(x))])
    u = jk.mean() - jk
    acc = np.sum(u ** 3) / (6 * np.sum(u ** 2) ** 1.5)
    za = N01.inv_cdf(a)
    a1 = N01.cdf(z0 + (z0 + za) / (1 - acc * (z0 + za)))
    a2 = N01.cdf(z0 + (z0 - za) / (1 - acc * (z0 - za)))
    return _cuantil(tb, a1), _cuantil(tb, a2)


def bootstrap_grupos(x, g, est, B=2000, conf=0.95, seed=0):
    """Bootstrap por grupos (clústeres): se remuestrean camiones enteros, con todos sus trayectos."""
    x = np.asarray(x, float); g = np.asarray(g); ids = np.unique(g)
    partes = [x[g == i] for i in ids]; r = np.random.default_rng(seed); tb = np.empty(B)
    for b in range(B):
        tb[b] = est(np.concatenate([partes[j] for j in r.integers(0, len(ids), len(ids))]), axis=-1)
    a = (1 - conf) / 2
    return _cuantil(tb, a), _cuantil(tb, 1 - a)


def bootstrap_bloques(x, est, L, B=2000, conf=0.95, seed=0):
    """Bootstrap de bloques móviles (Künsch): trozos seguidos de longitud L, para series temporales."""
    x = np.asarray(x, float); n = len(x); k = math.ceil(n / L); r = np.random.default_rng(seed)
    ini = r.integers(0, n - L + 1, (B, k))
    idx = (ini[:, :, None] + np.arange(L)).reshape(B, -1)[:, :n]
    tb = est(x[idx], axis=-1); a = (1 - conf) / 2
    return _cuantil(tb, a), _cuantil(tb, 1 - a)
