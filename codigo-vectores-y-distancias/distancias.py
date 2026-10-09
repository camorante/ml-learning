"""Normas, distancias y vecinos más cercanos, desde cero (solo numpy)."""
import numpy as np


def norma(x, p=2.0):
    """Norma Lp de un vector (o de cada fila de una matriz). p=np.inf es la del máximo."""
    a = np.abs(np.asarray(x, float))
    if p == np.inf:
        return a.max(-1)
    if p == 1:
        return a.sum(-1)
    if p == 2:
        return np.sqrt((a * a).sum(-1))
    return (a ** p).sum(-1) ** (1.0 / p)


def minkowski(x, y, p=2.0, w=None):
    """||x - y||_p, con pesos opcionales por componente: (sum w_i |x_i - y_i|^p)^(1/p)."""
    d = np.asarray(x, float) - np.asarray(y, float)
    if w is not None:
        d = d * np.asarray(w, float) ** (1.0 / p)
    return norma(d, p)


def matriz_distancias(X, Y=None, p=2.0, w=None, bloque=2048):
    """D[i, j] = ||X_i - Y_j||_p por difusión (broadcasting), en bloques de filas
    para no crear de golpe un array (n, m, d) gigante."""
    X = np.atleast_2d(np.asarray(X, float))
    Y = X if Y is None else np.atleast_2d(np.asarray(Y, float))
    if w is not None:
        s = np.asarray(w, float) ** (1.0 / p)
        X, Y = X * s, Y * s
    D = np.empty((len(X), len(Y)))
    for i in range(0, len(X), bloque):
        D[i:i + bloque] = norma(X[i:i + bloque, None, :] - Y[None, :, :], p)
    return D


def euclidea_rapida(X, Y=None):
    """Euclídea con ||x||² + ||y||² - 2 x·y: una multiplicación de matrices (rápida),
    pero con cancelación numérica si los puntos están lejos del origen."""
    X = np.atleast_2d(np.asarray(X, float))
    Y = X if Y is None else np.atleast_2d(np.asarray(Y, float))
    d2 = (X * X).sum(1)[:, None] + (Y * Y).sum(1)[None, :] - 2.0 * X @ Y.T
    return np.sqrt(np.maximum(d2, 0.0))          # el redondeo puede dar d² < 0


def estandarizar(X, media=None, desv=None):
    """z = (x - media) / desviación, por columnas (desviación poblacional, como StandardScaler)."""
    X = np.asarray(X, float)
    media = X.mean(0) if media is None else media
    desv = X.std(0) if desv is None else desv
    return (X - media) / np.where(desv > 0, desv, 1.0), media, desv


def mahalanobis(X, Y, VI):
    """Distancia de Mahalanobis con VI = inversa de la covarianza:
    se factoriza VI = L Lᵀ y se usa la euclídea sobre los puntos transformados x·L."""
    L = np.linalg.cholesky(np.asarray(VI, float))
    return matriz_distancias(np.asarray(X, float) @ L, np.asarray(Y, float) @ L, 2)


def vecinos(Xq, X, k=1, p=2.0, excluir_a_si_mismo=False):
    """Índices y distancias de los k puntos de X más cercanos a cada fila de Xq."""
    D = matriz_distancias(Xq, X, p)
    if excluir_a_si_mismo:
        np.fill_diagonal(D, np.inf)
    idx = np.argsort(D, axis=1, kind="stable")[:, :k]
    return idx, np.take_along_axis(D, idx, 1)


def knn_predecir(Xtr, ytr, Xq, k=5, p=2.0):
    """Voto de la mayoría entre los k vecinos (empates: la etiqueta menor)."""
    idx, _ = vecinos(Xq, Xtr, k, p)
    ytr = np.asarray(ytr)
    clases = np.unique(ytr)
    votos = (ytr[idx][:, :, None] == clases[None, None, :]).sum(1)
    return clases[votos.argmax(1)]
