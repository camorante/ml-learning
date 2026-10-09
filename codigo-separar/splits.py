"""Divisiones de datos desde cero (mismos índices que scikit-learn)."""
import numpy as np

def kfold(n, k=5, shuffle=False, seed=None):
    idx = np.arange(n)
    if shuffle:
        np.random.RandomState(seed).shuffle(idx)      # mismo generador que sklearn
    sizes = np.full(k, n // k); sizes[: n % k] += 1    # los primeros n % k trozos llevan uno más
    start = 0
    for s in sizes:
        test = idx[start:start + s]
        yield np.setdiff1d(idx, test), np.sort(test)     # sklearn devuelve ambos ordenados
        start += s

def time_series_split(n, k=5, test_size=None, gap=0):
    """Ventana creciente: entrena con el pasado, prueba con el bloque siguiente."""
    test_size = test_size or n // (k + 1)
    for start in range(n - k * test_size, n, test_size):
        yield np.arange(0, start - gap), np.arange(start, start + test_size)

def group_kfold(groups, k=5):
    """Grupos enteros a un solo trozo; reparte los grupos grandes primero, al trozo más ligero."""
    uniq, gidx = np.unique(groups, return_inverse=True)
    size = np.bincount(gidx)
    order = np.argsort(size, kind="stable")[::-1]
    load = np.zeros(k); fold_of = np.zeros(len(uniq), int)
    for g in order:
        f = np.argmin(load); load[f] += size[g]; fold_of[g] = f
    fold = fold_of[gidx]
    for f in range(k):
        yield np.where(fold != f)[0], np.where(fold == f)[0]

def date_split(dates, k=5, gap=0, embargo=0):
    """Para tablas con muchos camiones por día: divide por fechas únicas.
    gap: días quitados del entrenamiento justo antes de la prueba (purga).
    embargo: días quitados justo después de la prueba (solo útil si se entrena también con el futuro)."""
    d = np.asarray(dates); u = np.unique(d)
    for tr_d, te_d in time_series_split(len(u), k, gap=gap):
        yield np.where(np.isin(d, u[tr_d]))[0], np.where(np.isin(d, u[te_d]))[0]

def blocked_kfold_purged(dates, k=5, gap=0):
    """k-fold por bloques de fechas contiguas con purga a ambos lados (entrena también con el futuro)."""
    d = np.asarray(dates); u = np.unique(d)
    for _, te in kfold(len(u), k):
        lo, hi = te.min(), te.max()
        keep = (np.arange(len(u)) < lo - gap) | (np.arange(len(u)) > hi + gap)
        yield np.where(np.isin(d, u[keep]))[0], np.where(np.isin(d, u[te]))[0]

def acc_interval(aciertos, n, z=1.96):
    """Intervalo de Wilson para una proporción de aciertos."""
    p = aciertos / n; den = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / den; h = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / den
    return c - h, c + h

def mae_interval(err_abs, z=1.96, bloques=None):
    """Intervalo normal para el MAE. Con 'bloques' (p. ej. camión o semana) usa la varianza entre bloques,
    que es la honesta cuando los errores de un mismo bloque están correlacionados."""
    e = np.asarray(err_abs, float); m = e.mean()
    if bloques is None:
        se = e.std(ddof=1) / np.sqrt(len(e))
    else:
        b = np.asarray(bloques); ub = np.unique(b)
        mb = np.array([e[b == x].mean() for x in ub]); w = np.array([(b == x).sum() for x in ub])
        se = np.sqrt(np.sum(w**2 * (mb - m)**2) / (np.sum(w)**2) * len(ub) / (len(ub) - 1))
    return m - z * se, m + z * se
