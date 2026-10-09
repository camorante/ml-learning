"""Métricas y técnicas para clases desbalanceadas, desde cero."""
import numpy as np

def matriz(y, s, t):
    """Verdaderos/falsos positivos y negativos al avisar cuando la puntuación s >= t."""
    y = np.asarray(y, bool); a = np.asarray(s) >= t
    return dict(vp=int(np.sum(a & y)), fp=int(np.sum(a & ~y)), fn=int(np.sum(~a & y)), vn=int(np.sum(~a & ~y)))

def metricas(vp, fp, fn, vn, beta=1.0):
    n = vp + fp + fn + vn; prec = vp / (vp + fp) if vp + fp else 0.0; rec = vp / (vp + fn) if vp + fn else 0.0
    esp = vn / (vn + fp) if vn + fp else 0.0
    fb = (1 + beta**2) * prec * rec / (beta**2 * prec + rec) if prec + rec else 0.0
    den = np.sqrt(float((vp + fp) * (vp + fn) * (vn + fp) * (vn + fn)))
    mcc = (vp * vn - fp * fn) / den if den else 0.0
    return dict(acierto=(vp + vn) / n, precision=prec, exhaustividad=rec, especificidad=esp,
                f=fb, acierto_equilibrado=(rec + esp) / 2, mcc=mcc)

def _ordenar(y, s):
    o = np.argsort(-np.asarray(s, float), kind="mergesort"); y = np.asarray(y, bool)[o]; s = np.asarray(s, float)[o]
    corte = np.r_[np.where(np.diff(s))[0], len(s) - 1]          # último índice de cada valor distinto
    return np.cumsum(y)[corte], np.cumsum(~y)[corte], s[corte]

def curva_roc(y, s):
    vp, fp, umb = _ordenar(y, s)
    return np.r_[0, fp / fp[-1]], np.r_[0, vp / vp[-1]], np.r_[np.inf, umb]

def area(x, y):
    return float(np.sum(np.diff(x) * (y[1:] + y[:-1]) / 2))

def curva_pr(y, s):
    vp, fp, umb = _ordenar(y, s)
    return vp / (vp + fp), vp / vp[-1], umb

def precision_media(y, s):
    """AP = suma de (R_n - R_{n-1}) * P_n: sin interpolar, como scikit-learn."""
    p, r, _ = curva_pr(y, s); return float(np.sum(np.diff(np.r_[0, r]) * p))

def umbral_coste(c_fp, c_fn):
    """Con probabilidades calibradas, avisar si p > c_fp / (c_fp + c_fn)."""
    return c_fp / (c_fp + c_fn)

def corregir_prior(p, factor):
    """Si el entrenamiento multiplicó el peso (o la frecuencia) de los positivos por 'factor',
    las probabilidades salen infladas: se divide la razón de apuestas entre 'factor'."""
    p = np.asarray(p, float); o = p / (1 - p) / factor; return o / (1 + o)

def smote(X, y, k=5, n_nuevos=None, seed=0):
    """Crea positivos sintéticos en el segmento entre un positivo y uno de sus k vecinos positivos."""
    rng = np.random.default_rng(seed); X = np.asarray(X, float); y = np.asarray(y, bool); P = X[y]
    n_nuevos = (len(y) - 2 * len(P)) if n_nuevos is None else n_nuevos
    D = ((P[:, None, :] - P[None, :, :]) ** 2).sum(-1); np.fill_diagonal(D, np.inf); vec = np.argsort(D, 1)[:, :k]
    i = rng.integers(0, len(P), n_nuevos); j = vec[i, rng.integers(0, k, n_nuevos)]; u = rng.random((n_nuevos, 1))
    nuevos = P[i] + u * (P[j] - P[i])
    return np.vstack([X, nuevos]), np.r_[y, np.ones(n_nuevos, bool)]

def brier(y, p):
    return float(np.mean((np.asarray(p) - np.asarray(y, float)) ** 2))
