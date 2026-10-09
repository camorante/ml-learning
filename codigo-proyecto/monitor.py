"""Vigilar un modelo en producción: deriva de los datos (PSI, Kolmogórov-Smirnov) desde cero."""
import numpy as np

def cortes(ref, n=10):
    """Bordes de n tramos con el mismo número de datos de referencia (deciles), abiertos en los extremos."""
    q = np.quantile(ref[~np.isnan(ref)], np.linspace(0, 1, n + 1)[1:-1]); return np.r_[-np.inf, np.unique(q), np.inf]

def psi(ref, act, n=10, eps=1e-4):
    """Índice de estabilidad de la población: suma de (a - e) * ln(a / e) por tramo.
    < 0,1 estable · 0,1-0,25 vigilar · > 0,25 cambio importante (reglas habituales)."""
    ref = np.asarray(ref, float); act = np.asarray(act, float); b = cortes(ref, n)
    e = np.histogram(ref[~np.isnan(ref)], b)[0] / np.sum(~np.isnan(ref))
    a = np.histogram(act[~np.isnan(act)], b)[0] / np.sum(~np.isnan(act))
    e = np.clip(e, eps, None); a = np.clip(a, eps, None)
    return float(np.sum((a - e) * np.log(a / e)))

def ks(ref, act):
    """Estadístico de Kolmogórov-Smirnov: máxima distancia entre las dos funciones de distribución empíricas."""
    x = np.sort(np.asarray(ref, float)); y = np.sort(np.asarray(act, float)); z = np.r_[x, y]
    return float(np.max(np.abs(np.searchsorted(x, z, side="right") / len(x) - np.searchsorted(y, z, side="right") / len(y))))

def ks_p(D, n, m):
    """Valor p asintótico (Kolmogórov): P(D > d) ≈ 2 Σ (-1)^(k-1) exp(-2 k² λ²)."""
    en = np.sqrt(n * m / (n + m)); lam = (en + 0.12 + 0.11 / en) * D
    k = np.arange(1, 101); return float(np.clip(2 * np.sum((-1) ** (k - 1) * np.exp(-2 * k**2 * lam**2)), 0, 1))
