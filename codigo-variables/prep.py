"""Preparar variables desde cero: escalar, codificar, ciclos y ventanas de tiempo."""
import numpy as np

class Estandarizar:
    """z = (x - media) / desviación (con ddof=0, como StandardScaler)."""
    def fit(self, X):
        X = np.asarray(X, float); self.media = X.mean(0); s = X.std(0); self.desv = np.where(s > 0, s, 1.0); return self
    def transform(self, X):
        return (np.asarray(X, float) - self.media) / self.desv

class MinMax:
    def fit(self, X):
        X = np.asarray(X, float); self.lo = X.min(0); r = X.max(0) - self.lo; self.rango = np.where(r > 0, r, 1.0); return self
    def transform(self, X):
        return (np.asarray(X, float) - self.lo) / self.rango

class Robusto:
    """(x - mediana) / rango intercuartílico: los atípicos no mueven la escala."""
    def fit(self, X):
        X = np.asarray(X, float); self.med = np.median(X, 0); q1, q3 = np.percentile(X, [25, 75], axis=0)
        iqr = q3 - q1; self.iqr = np.where(iqr > 0, iqr, 1.0); return self
    def transform(self, X):
        return (np.asarray(X, float) - self.med) / self.iqr

class UnaColumnaPorCategoria:
    """One-hot. Una categoría que no se vio al ajustar queda como todo ceros."""
    def fit(self, x, quitar_primera=False):
        self.cats = sorted(set(x)); self.cols = self.cats[1:] if quitar_primera else self.cats; return self
    def transform(self, x):
        pos = {c: j for j, c in enumerate(self.cols)}; M = np.zeros((len(x), len(self.cols)))
        for i, v in enumerate(x):
            if v in pos: M[i, pos[v]] = 1.0
        return M

def _medias_suavizadas(x, y, m):
    x = np.asarray(x); y = np.asarray(y, float); g = y.mean(); tab = {}
    for c in np.unique(x):
        yc = y[x == c]; tab[c] = (yc.sum() + m * g) / (len(yc) + m)     # n·media_c + m·global, entre n + m
    return tab, g

class MediaPorCategoria:
    """Codificación por la media del objetivo, suavizada hacia la media global con 'm' viajes ficticios.
    fit_transform usa ajuste cruzado: cada fila se codifica con medias calculadas SIN su pliegue."""
    def __init__(self, m=10.0): self.m = m
    def fit(self, x, y):
        self.tab, self.g = _medias_suavizadas(x, y, self.m); return self
    def transform(self, x):
        return np.array([self.tab.get(v, self.g) for v in x])
    def fit_transform(self, x, y, pliegues):
        x = np.asarray(x); out = np.empty(len(x))
        for tr, te in pliegues:
            tab, g = _medias_suavizadas(x[tr], np.asarray(y)[tr], self.m)
            out[te] = [tab.get(v, g) for v in x[te]]
        self.fit(x, y); return out

def ciclo(v, periodo):
    """Hora, día de la semana o mes como punto en un círculo: 23 h y 1 h quedan cerca."""
    a = 2 * np.pi * np.asarray(v, float) / periodo; return np.c_[np.sin(a), np.cos(a)]

def media_pasada(grupo, t, v, ventana):
    """Media de v en (t - ventana, t) para el mismo grupo, solo con filas ANTERIORES. NaN si no hay ninguna."""
    grupo = np.asarray(grupo); t = np.asarray(t, float); v = np.asarray(v, float); out = np.full(len(t), np.nan)
    for g in np.unique(grupo):
        idx = np.where(grupo == g)[0]; idx = idx[np.argsort(t[idx], kind="mergesort")]; tg = t[idx]; vg = v[idx]
        cs = np.r_[0, np.cumsum(vg)]
        lo = np.searchsorted(tg, tg - ventana, side="right"); hi = np.searchsorted(tg, tg, side="left")
        n = hi - lo; out[idx] = np.where(n > 0, (cs[hi] - cs[lo]) / np.maximum(n, 1), np.nan)
    return out
