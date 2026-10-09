"""Análisis exploratorio desde cero: resúmenes robustos, forma, correlaciones, redundancia y centinelas."""
import numpy as np

def quantile(x, q):
    """Cuantil con interpolación lineal (el 'tipo 7' de Hyndman-Fan, por defecto en numpy y pandas)."""
    s = np.sort(np.asarray(x, float)); h = (len(s) - 1) * q
    lo = int(np.floor(h)); hi = min(lo + 1, len(s) - 1)
    return s[lo] + (h - lo) * (s[hi] - s[lo])

def resumen(x):
    x = np.asarray(x, float); x = x[~np.isnan(x)]; n = len(x)
    med = quantile(x, 0.5); mad = quantile(np.abs(x - med), 0.5)
    q1, q3 = quantile(x, 0.25), quantile(x, 0.75)
    return dict(n=n, media=x.mean(), desv=x.std(ddof=1), min=x.min(), q1=q1, mediana=med, q3=q3, max=x.max(),
                iqr=q3 - q1, mad=mad, mad_normal=1.4826 * mad, asimetria=skew(x), curtosis=kurt(x))

def skew(x):
    """Asimetría ajustada G1 (la que da pandas): 0 si es simétrica, > 0 si tiene cola a la derecha."""
    x = np.asarray(x, float); n = len(x); d = x - x.mean()
    g1 = np.mean(d**3) / np.mean(d**2) ** 1.5
    return np.sqrt(n * (n - 1)) / (n - 2) * g1

def kurt(x):
    """Curtosis en exceso ajustada G2 (la de pandas): 0 para la normal, > 0 con colas pesadas."""
    x = np.asarray(x, float); n = len(x); d = x - x.mean()
    g2 = np.mean(d**4) / np.mean(d**2) ** 2 - 3
    return (n - 1) / ((n - 2) * (n - 3)) * ((n + 1) * g2 + 6)

def pearson(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float); dx, dy = x - x.mean(), y - y.mean()
    return np.sum(dx * dy) / np.sqrt(np.sum(dx**2) * np.sum(dy**2))

def rangos(x):
    """Rangos 1..n; los empates reciben la media de sus posiciones."""
    x = np.asarray(x, float); orden = np.argsort(x, kind="mergesort"); r = np.empty(len(x)); s = x[orden]; i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and s[j + 1] == s[i]: j += 1
        r[orden[i:j + 1]] = (i + j) / 2 + 1; i = j + 1
    return r

def spearman(x, y):
    return pearson(rangos(x), rangos(y))

def kendall_b(x, y):
    """Tau-b de Kendall, O(n²): pares concordantes menos discordantes, corregido por empates."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    dx = np.sign(x[:, None] - x[None, :]); dy = np.sign(y[:, None] - y[None, :])
    iu = np.triu_indices(len(x), 1); s = np.sum(dx[iu] * dy[iu])
    return s / np.sqrt(np.sum(dx[iu] != 0) * np.sum(dy[iu] != 0))

def vif(X):
    """Factor de inflación de la varianza de cada columna: 1 / (1 - R²) al explicarla con las demás."""
    X = np.asarray(X, float); out = []
    for j in range(X.shape[1]):
        A = np.column_stack([np.ones(len(X)), np.delete(X, j, axis=1)]); y = X[:, j]
        res = y - A @ np.linalg.lstsq(A, y, rcond=None)[0]
        r2 = 1 - res @ res / np.sum((y - y.mean())**2); out.append(1 / (1 - r2))
    return np.array(out)

def centinelas(x, min_rep=10, decimales=3):
    """Valores que se repiten exactamente demasiadas veces en una columna continua (-40, 255, 0, 9999...)."""
    v, c = np.unique(np.round(np.asarray(x, float), decimales), return_counts=True)
    return {float(a): int(b) for a, b in zip(v, c) if b >= min_rep}

def fuera_de_rango(df, reglas):
    """reglas: {columna: (mínimo, máximo)} con límites físicos. Devuelve cuántas filas incumple cada una."""
    return {c: int(((df[c] < lo) | (df[c] > hi)).sum()) for c, (lo, hi) in reglas.items()}

def descomponer_covarianza(x, y, g):
    """Cov total = media de las covarianzas dentro de cada grupo + covarianza entre las medias de los grupos."""
    x, y, g = np.asarray(x, float), np.asarray(y, float), np.asarray(g); n = len(x)
    dentro = entre = 0.0
    for k in np.unique(g):
        m = g == k; w = m.sum() / n
        dentro += w * np.mean((x[m] - x[m].mean()) * (y[m] - y[m].mean()))
        entre += w * (x[m].mean() - x.mean()) * (y[m].mean() - y.mean())
    return dentro, entre, np.mean((x - x.mean()) * (y - y.mean()))
