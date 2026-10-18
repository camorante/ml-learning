"""Covarianza y correlación desde cero (solo numpy y la librería estándar)."""
import math
from statistics import NormalDist
import numpy as np


def covarianza(x, y, ddof=1):
    """Dos pasadas: primero las medias, luego la media de los productos de las desviaciones."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    return float(np.sum((x - x.mean()) * (y - y.mean())) / (len(x) - ddof))


def pearson(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    dx, dy = x - x.mean(), y - y.mean()
    return float(np.sum(dx * dy) / math.sqrt(np.sum(dx * dx) * np.sum(dy * dy)))   # coseno de dx y dy


def rangos(x):
    """Puestos 1..n; los empates reciben la media de sus puestos."""
    x = np.asarray(x, float); o = np.argsort(x, kind="mergesort"); xs = x[o]
    r = np.empty(len(x)); i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and xs[j + 1] == xs[i]:
            j += 1
        r[o[i:j + 1]] = (i + j) / 2 + 1; i = j + 1
    return r


def spearman(x, y):
    return pearson(rangos(x), rangos(y))


def kendall_tau_b(x, y):
    """(concordantes − discordantes) / sqrt((n0 − n1)(n0 − n2)); O(n²), suficiente para miles de filas."""
    x = np.asarray(x, float); y = np.asarray(y, float); n = len(x)
    i, j = np.triu_indices(n, 1)
    s = np.sign(x[i] - x[j]) * np.sign(y[i] - y[j])
    n0 = n * (n - 1) / 2; tx = np.sum(x[i] == x[j]); ty = np.sum(y[i] == y[j])
    return float(np.sum(s) / math.sqrt((n0 - tx) * (n0 - ty)))


def matriz_cov(X, ddof=1):
    """Filas = observaciones, columnas = variables: Xcᵀ Xc / (n − 1)."""
    X = np.asarray(X, float); Xc = X - X.mean(axis=0)
    return Xc.T @ Xc / (len(X) - ddof)


def matriz_corr(X):
    S = matriz_cov(X); d = np.sqrt(np.diag(S))
    R = S / np.outer(d, d); np.fill_diagonal(R, 1.0)
    return R


def correlacion_parcial(x, y, z):
    """r de x e y después de quitarles lo que explica linealmente z (una o varias columnas)."""
    Z = np.column_stack([np.ones(len(x)), np.asarray(z, float)])
    res = lambda v: v - Z @ np.linalg.lstsq(Z, v, rcond=None)[0]
    return pearson(res(np.asarray(x, float)), res(np.asarray(y, float)))


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


def p_valor_r(r, n):
    """Contraste de ρ = 0 (dos colas): t = r·sqrt((n − 2)/(1 − r²)) sigue una t de Student con n − 2 g.l."""
    df = n - 2; t2 = r * r * df / (1 - r * r)
    return beta_inc(df / 2, 0.5, df / (df + t2))


def ic_fisher(r, n, conf=0.95):
    """Intervalo para ρ con la transformación de Fisher: atanh(r) ≈ normal con desviación 1/sqrt(n − 3)."""
    z = NormalDist().inv_cdf((1 + conf) / 2); h = z / math.sqrt(n - 3)
    return math.tanh(math.atanh(r) - h), math.tanh(math.atanh(r) + h)


def vif(X):
    """Factor de inflación de la varianza de cada columna: 1 / (1 − R²_j) = diagonal de la inversa de R."""
    return np.diag(np.linalg.inv(matriz_corr(X)))


class CovOnline:
    """Covarianza en una pasada (Welford) y fusión de dos resúmenes (Chan, Golub y LeVeque)."""
    def __init__(self):
        self.n = 0; self.mx = 0.0; self.my = 0.0; self.C = 0.0; self.Sx = 0.0; self.Sy = 0.0

    def add(self, x, y):
        dx, dy = x - self.mx, y - self.my                  # desviaciones respecto a las medias viejas
        self.n += 1; self.mx += dx / self.n; self.my += dy / self.n
        self.C += dx * (y - self.my)                       # vieja × nueva: exacto, sin restas grandes
        self.Sx += dx * (x - self.mx); self.Sy += dy * (y - self.my)
        return self

    def merge(self, o):
        n = self.n + o.n; dx, dy = o.mx - self.mx, o.my - self.my; f = self.n * o.n / n
        self.C += o.C + dx * dy * f; self.Sx += o.Sx + dx * dx * f; self.Sy += o.Sy + dy * dy * f
        self.mx += dx * o.n / n; self.my += dy * o.n / n; self.n = n
        return self

    def cov(self): return self.C / (self.n - 1)
    def corr(self): return self.C / math.sqrt(self.Sx * self.Sy)
