"""Tratamiento de datos que faltan, desde cero."""
import numpy as np

def media_completos(x):
    x = np.asarray(x, float); return np.nanmean(x)

def imputar_media(x):
    x = np.asarray(x, float).copy(); x[np.isnan(x)] = np.nanmean(x); return x

def imputar_grupo_sorteo(x, g, seed=0):
    """'Hot deck': cada hueco recibe el valor de un caso observado al azar del mismo grupo."""
    rng = np.random.default_rng(seed); x = np.asarray(x, float).copy(); g = np.asarray(g)
    for k in np.unique(g):
        m = g == k; obs = x[m & ~np.isnan(x)]; hu = np.where(m & np.isnan(x))[0]
        x[hu] = rng.choice(obs, len(hu), replace=True)
    return x

def imputar_regresion(x, Z, ruido=False, seed=0):
    """Rellena x con una regresión lineal sobre las columnas Z (sin huecos).
    Con ruido=True suma un residuo sorteado, para no aplastar la dispersión."""
    x = np.asarray(x, float).copy(); A = np.column_stack([np.ones(len(x)), Z]); m = np.isnan(x)
    b = np.linalg.lstsq(A[~m], x[~m], rcond=None)[0]; x[m] = A[m] @ b
    if ruido:
        res = x[~m] - A[~m] @ b; x[m] += np.random.default_rng(seed).choice(res, m.sum())
    return x

def nan_euclidean(X, Y):
    """Distancia euclídea ignorando coordenadas con hueco y reescalando por las que quedan (como scikit-learn)."""
    X, Y = np.asarray(X, float), np.asarray(Y, float); p = X.shape[1]
    D = np.full((len(X), len(Y)), np.nan)
    for i in range(len(X)):
        ok = ~np.isnan(X[i]) & ~np.isnan(Y); d2 = np.where(ok, (Y - X[i]) ** 2, 0).sum(1); n = ok.sum(1)
        D[i] = np.where(n > 0, np.sqrt(p / np.maximum(n, 1) * d2), np.nan)
    return D

def imputar_knn(X, k=5):
    """Media de los k donantes más cercanos (distancia nan-euclídea) que sí tienen ese dato."""
    X = np.asarray(X, float); out = X.copy(); M = np.isnan(X)
    for j in range(X.shape[1]):
        rec = np.where(M[:, j])[0]; don = np.where(~M[:, j])[0]
        if not len(rec): continue
        D = nan_euclidean(X[rec], X[don])
        for a, i in enumerate(rec):
            d = D[a]; ok = ~np.isnan(d)
            if not ok.any(): out[i, j] = X[don, j].mean(); continue
            orden = np.argsort(d[ok], kind="stable")[:k]
            out[i, j] = X[don[ok][orden], j].mean()
    return out

def imputar_encadenado(X, max_iter=10, tol=1e-3):
    """Imputación encadenada (MICE determinista) con regresión lineal: empieza con la media y,
    columna a columna (de menos a más huecos), predice cada una con todas las demás."""
    X = np.asarray(X, float); M = np.isnan(X); Xt = np.where(M, np.nanmean(X, 0), X)
    orden = [j for j in np.argsort(M.sum(0), kind="mergesort") if M[:, j].any()]
    lim = tol * np.max(np.abs(X[~M]))
    for _ in range(max_iter):
        prev = Xt.copy()
        for j in orden:
            otras = np.delete(np.arange(X.shape[1]), j); A = np.column_stack([np.ones(len(X)), Xt[:, otras]])
            b = np.linalg.lstsq(A[~M[:, j]], Xt[~M[:, j], j], rcond=None)[0]
            Xt[M[:, j], j] = A[M[:, j]] @ b
        if np.linalg.norm(Xt - prev, ord=np.inf) < lim: break      # norma infinito de matriz (máx. suma por fila), como sklearn
    return Xt

def rubin(estimaciones, varianzas):
    """Reglas de Rubin para m imputaciones: estimación combinada y su varianza total."""
    q = np.asarray(estimaciones, float); u = np.asarray(varianzas, float); m = len(q)
    qb = q.mean(); W = u.mean(); B = q.var(ddof=1); T = W + (1 + 1 / m) * B
    return qb, T, W, B
