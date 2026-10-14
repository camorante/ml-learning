"""Media, mediana, varianza y compañía, desde cero (solo NumPy para los arrays)."""
import math
import numpy as np


def media(x):
    x = np.asarray(x, float)
    m = x.sum() / len(x)
    return m + (x - m).sum() / len(x)          # segunda pasada: corrige el redondeo de la primera


def mediana(x):
    s = np.sort(np.asarray(x, float)); n = len(s); k = n // 2
    return s[k] if n % 2 else (s[k - 1] + s[k]) / 2


def varianza(x, ddof=1):
    """Dos pasadas corregida (Björck): suma de cuadrados respecto a la media, menos el error de redondeo de la media."""
    x = np.asarray(x, float); n = len(x)
    if n - ddof <= 0:
        return math.nan
    d = x - media(x)
    return (np.dot(d, d) - d.sum() ** 2 / n) / (n - ddof)


def desviacion(x, ddof=1):
    return math.sqrt(varianza(x, ddof))


def varianza_ingenua(x, ddof=1):
    """La fórmula de una pasada de los libros: (Σx² − (Σx)²/n) / (n − ddof). NO usar: cancelación catastrófica."""
    x = np.asarray(x); n = len(x)
    return (np.sum(x * x) - np.sum(x) ** 2 / n) / (n - ddof)


def cv(x, ddof=1):
    """Coeficiente de variación: desviación típica / media (solo con datos positivos en una escala con cero real)."""
    return desviacion(x, ddof) / media(x)


def zscore(x, ddof=0):
    """Estandarizar: (x − media) / desviación. ddof=0 como StandardScaler y scipy.stats.zscore."""
    x = np.asarray(x, float)
    return (x - media(x)) / desviacion(x, ddof)


def mad(x, escala=1.0):
    """Mediana de las distancias a la mediana. Con escala=1.4826 estima σ si los datos fueran normales (tema 8)."""
    x = np.asarray(x, float)
    return escala * mediana(np.abs(x - mediana(x)))


def media_recortada(x, prop=0.1):
    """Quita el prop de valores más bajos y el prop de los más altos (redondeando hacia abajo) y promedia el resto."""
    s = np.sort(np.asarray(x, float)); k = int(prop * len(s))
    return media(s[k:len(s) - k])


def media_ponderada(x, w):
    x = np.asarray(x, float); w = np.asarray(w, float)
    return np.dot(w, x) / w.sum()


class Welford:
    """Media y varianza en streaming: una pasada, memoria O(1), estable. fusionar() une dos resúmenes (Chan et al.)."""

    def __init__(self):
        self.n, self.m, self.M2 = 0, 0.0, 0.0

    def add(self, x):
        self.n += 1
        d = x - self.m
        self.m += d / self.n
        self.M2 += d * (x - self.m)              # d · (x − media nueva): siempre ≥ 0
        return self

    def fusionar(self, o):
        n = self.n + o.n
        if n == 0:
            return self
        d = o.m - self.m
        self.m += d * o.n / n
        self.M2 += o.M2 + d * d * self.n * o.n / n
        self.n = n
        return self

    def varianza(self, ddof=1):
        return self.M2 / (self.n - ddof) if self.n > ddof else math.nan

    def desviacion(self, ddof=1):
        return math.sqrt(self.varianza(ddof))
