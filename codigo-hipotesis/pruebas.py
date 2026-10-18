"""Pruebas de hipótesis desde cero (solo numpy y la librería estándar)."""
import math
from itertools import combinations
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


def t_sf(t, gl):
    """P(T > t) para una t de Student con gl grados de libertad."""
    p = 0.5 * beta_inc(gl / 2, 0.5, gl / (gl + t * t))
    return p if t > 0 else 1 - p


def t_ppf(q, gl):
    lo, hi = -1e3, 1e3
    for _ in range(200):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if 1 - t_sf(mid, gl) < q else (lo, mid)
    return (lo + hi) / 2


def _p(stat, sf, alternativa):
    """p-valor a partir de la función de supervivencia de la distribución nula (simétrica)."""
    if alternativa == "mayor": return sf(stat)
    if alternativa == "menor": return 1 - sf(stat)
    return min(1.0, 2 * sf(abs(stat)))


def welch(x, y, alternativa="bilateral", conf=0.95):
    """Test t de Welch para media(x) − media(y): estadístico, grados de libertad, p e intervalo."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    vx, vy = x.var(ddof=1) / len(x), y.var(ddof=1) / len(y)
    ee = math.sqrt(vx + vy); d = x.mean() - y.mean(); t = d / ee
    gl = (vx + vy) ** 2 / (vx**2 / (len(x) - 1) + vy**2 / (len(y) - 1))      # Welch-Satterthwaite
    h = t_ppf((1 + conf) / 2, gl) * ee
    return dict(dif=d, ee=ee, t=t, gl=gl, p=_p(t, lambda s: t_sf(s, gl), alternativa), ic=(d - h, d + h))


def student(x, y, alternativa="bilateral"):
    """Test t con varianza común (supone la misma dispersión en los dos grupos)."""
    x, y = np.asarray(x, float), np.asarray(y, float); n1, n2 = len(x), len(y)
    sp2 = ((n1 - 1) * x.var(ddof=1) + (n2 - 1) * y.var(ddof=1)) / (n1 + n2 - 2)
    t = (x.mean() - y.mean()) / math.sqrt(sp2 * (1 / n1 + 1 / n2)); gl = n1 + n2 - 2
    return dict(t=t, gl=gl, p=_p(t, lambda s: t_sf(s, gl), alternativa))


def emparejada(x, y, alternativa="bilateral"):
    """Test t emparejado: el mismo camión antes (x) y después (y)."""
    d = np.asarray(x, float) - np.asarray(y, float); n = len(d)
    t = d.mean() / (d.std(ddof=1) / math.sqrt(n))
    return dict(t=t, gl=n - 1, p=_p(t, lambda s: t_sf(s, n - 1), alternativa))


def z_proporciones(k1, n1, k2, n2, alternativa="bilateral"):
    """Test z de dos proporciones con la proporción común bajo H0."""
    p1, p2, p = k1 / n1, k2 / n2, (k1 + k2) / (n1 + n2)
    z = (p1 - p2) / math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    return dict(z=z, p=_p(z, lambda s: 1 - N01.cdf(s), alternativa))


def chi2_tabla(tabla):
    """χ² de independencia de Pearson (sin corrección de Yates) para una tabla r × c."""
    O = np.asarray(tabla, float); E = O.sum(1, keepdims=True) * O.sum(0, keepdims=True) / O.sum()
    chi2 = float(((O - E) ** 2 / E).sum()); gl = (O.shape[0] - 1) * (O.shape[1] - 1)
    # supervivencia de la χ²_gl = gamma incompleta regularizada superior (serie / fracción continua)
    a, x = gl / 2, chi2 / 2
    if x < a + 1:
        s, term, k = 1 / a, 1 / a, 0
        while abs(term) > 1e-16 * abs(s):
            k += 1; term *= x / (a + k); s += term
        p = 1 - s * math.exp(-x + a * math.log(x) - math.lgamma(a))
    else:
        b = x + 1 - a; c = 1e300; d = 1 / b; h = d
        for i in range(1, 400):
            an = -i * (i - a); b += 2; d = an * d + b; d = 1 / (d if abs(d) > 1e-300 else 1e-300)
            c = b + an / c; c = c if abs(c) > 1e-300 else 1e-300; h *= d * c
            if abs(d * c - 1) < 1e-16: break
        p = math.exp(-x + a * math.log(x) - math.lgamma(a)) * h
    return dict(chi2=chi2, gl=gl, p=p, esperados=E)


def fisher_2x2(tabla):
    """Test exacto de Fisher bilateral: suma las tablas con márgenes fijos tan o menos probables."""
    (a, b), (c, d) = tabla; r1, c1, n = a + b, a + c, a + b + c + d
    lp = lambda k: (math.lgamma(r1 + 1) - math.lgamma(k + 1) - math.lgamma(r1 - k + 1) + math.lgamma(n - r1 + 1)
                    - math.lgamma(c1 - k + 1) - math.lgamma(n - r1 - c1 + k + 1) - math.lgamma(n + 1) + math.lgamma(c1 + 1) + math.lgamma(n - c1 + 1))
    ks = range(max(0, c1 - (n - r1)), min(r1, c1) + 1); obs = lp(a)
    return dict(p=min(1.0, sum(math.exp(lp(k)) for k in ks if lp(k) <= obs + 1e-7)))


def rangos(v):
    """Rangos medios (los empates reciben la media de sus posiciones)."""
    v = np.asarray(v, float); o = np.argsort(v, kind="mergesort"); r = np.empty(len(v)); s = v[o]; i = 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and s[j + 1] == s[i]: j += 1
        r[o[i:j + 1]] = (i + j) / 2 + 1; i = j + 1
    return r


def mann_whitney(x, y, alternativa="bilateral"):
    """U de Mann-Whitney con aproximación normal, corrección por empates y de continuidad."""
    n1, n2 = len(x), len(y); z_ = np.r_[x, y]; r = rangos(z_)
    U = r[:n1].sum() - n1 * (n1 + 1) / 2; mu = n1 * n2 / 2; N = n1 + n2
    _, t = np.unique(z_, return_counts=True)
    sd = math.sqrt(n1 * n2 / 12 * ((N + 1) - (t**3 - t).sum() / (N * (N - 1))))
    if alternativa == "bilateral":
        z = (abs(U - mu) - 0.5) / sd; p = min(1.0, 2 * (1 - N01.cdf(z)))
    elif alternativa == "mayor":
        p = 1 - N01.cdf((U - mu - 0.5) / sd)
    else:
        p = N01.cdf((U - mu + 0.5) / sd)
    return dict(U=U, p=p, auc=U / (n1 * n2))


def permutacion(x, y, m=None, seed=0, alternativa="bilateral"):
    """Test de permutación de la diferencia de medias. m=None: exacto (todas las particiones);
    si no, m barajadas al azar y p = (b + 1) / (m + 1)."""
    x, y = np.asarray(x, float), np.asarray(y, float); z = np.r_[x, y]; n1, S = len(x), z.sum(); N = len(z)
    obs = x.mean() - y.mean()
    if m is None:
        sumas = np.array([z[list(c)].sum() for c in combinations(range(N), n1)])
    else:
        rng = np.random.default_rng(seed); sumas = np.array([z[rng.permutation(N)[:n1]].sum() for _ in range(m)])
    d = sumas / n1 - (S - sumas) / (N - n1); tol = 1e-9 * max(1.0, abs(obs))
    if alternativa == "mayor": b = np.sum(d >= obs - tol)
    elif alternativa == "menor": b = np.sum(d <= obs + tol)
    else: b = np.sum(np.abs(d) >= abs(obs) - tol)
    p = b / len(d) if m is None else (b + 1) / (m + 1)
    return dict(dif=obs, b=int(b), total=len(d), p=float(p))


def bonferroni(p):
    p = np.asarray(p, float); return np.minimum(1, p * len(p))


def holm(p):
    """p-valores ajustados de Holm (escalón descendente)."""
    p = np.asarray(p, float); m = len(p); o = np.argsort(p)
    adj = np.maximum.accumulate((m - np.arange(m)) * p[o]); out = np.empty(m); out[o] = np.minimum(1, adj)
    return out


def bh(p):
    """p-valores ajustados de Benjamini-Hochberg (controlan la tasa de falsos descubrimientos)."""
    p = np.asarray(p, float); m = len(p); o = np.argsort(p)
    adj = np.minimum.accumulate((m / np.arange(m, 0, -1)) * p[o][::-1])[::-1]; out = np.empty(m); out[o] = np.minimum(1, adj)
    return out


def potencia_normal(delta, sigma, n, alfa=0.05):
    """Potencia aproximada (normal) del test bilateral con n por grupo."""
    ee = sigma * math.sqrt(2 / n); z = N01.inv_cdf(1 - alfa / 2); l = delta / ee
    return (1 - N01.cdf(z - l)) + N01.cdf(-z - l)


def n_necesario(delta, sigma, alfa=0.05, potencia=0.8):
    """n por grupo para detectar delta con la potencia pedida (aproximación normal)."""
    return math.ceil(2 * ((N01.inv_cdf(1 - alfa / 2) + N01.inv_cdf(potencia)) * sigma / delta) ** 2)


def potencia_t(delta, sigma, n1, n2=None, alfa=0.05, puntos=4001):
    """Potencia exacta del test t de Student bilateral: integra la t no central sobre la χ² del denominador."""
    n2 = n2 or n1; gl = n1 + n2 - 2; lam = delta / (sigma * math.sqrt(1 / n1 + 1 / n2)); tc = t_ppf(1 - alfa / 2, gl)
    v = np.linspace(1e-12, gl + 60 * math.sqrt(2 * gl) + 60, puntos)
    dens = np.exp((gl / 2 - 1) * np.log(v) - v / 2 - gl / 2 * math.log(2) - math.lgamma(gl / 2))
    c = tc * np.sqrt(v / gl); Phi = np.vectorize(N01.cdf)
    f = (1 - Phi(c - lam) + Phi(-c - lam)) * dens
    h = v[1] - v[0]; w = np.ones(puntos); w[1:-1:2] = 4; w[2:-1:2] = 2       # Simpson
    return float(h / 3 * np.sum(w * f))
