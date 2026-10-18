"""Muestreo y TCL desde cero: error estándar, población finita, tamaño efectivo,
muestreo estratificado (con asignación de Neyman) y efecto de diseño por conglomerados."""
import math
import numpy as np


def media_se(x, N=None):
    """Media muestral y su error estándar s/√n, con corrección por población finita si se da N."""
    x = np.asarray(x, float); n = len(x)
    m = x.sum() / n
    s2 = ((x - m) ** 2).sum() / (n - 1)          # cuasivarianza: divide entre n − 1
    se = math.sqrt(s2 / n)
    if N is not None:
        se *= math.sqrt(1 - n / N)               # fpc: s² estima S² (la de N − 1), así que el factor es 1 − n/N
    return m, se


def autocorr(x, k):
    """Autocorrelación muestral de retardo k (la estimación habitual, con denominador común)."""
    x = np.asarray(x, float); d = x - x.mean()
    return float((d[:-k] * d[k:]).sum() / (d * d).sum()) if k else 1.0


def n_efectivo(n, rho):
    """Tamaño efectivo de n datos con autocorrelaciones rho[k] (k = 1, 2, …):
    Var(x̄) = σ²/n · [1 + 2 Σ (1 − k/n) ρ_k]  ⇒  n_ef = n / [1 + 2 Σ (1 − k/n) ρ_k]."""
    f = 1 + 2 * sum((1 - k / n) * r for k, r in enumerate(rho, start=1) if k < n)
    return n / f


def n_efectivo_ar1(n, phi):
    """Caso AR(1): ρ_k = φ^k."""
    return n_efectivo(n, [phi ** k for k in range(1, n)])


def estratificado(y, h, N_h):
    """Estimador estratificado de la media de la población y su error estándar.
    y: valores de la muestra; h: estrato de cada valor; N_h: tamaño de cada estrato en la población."""
    y, h = np.asarray(y, float), np.asarray(h)
    N_h = np.asarray(N_h, float); W = N_h / N_h.sum()
    m = v = 0.0
    for k, w in enumerate(W):
        yk = y[h == k]; n_k = len(yk)
        mk = yk.mean(); s2k = yk.var(ddof=1)
        m += w * mk
        v += w**2 * s2k / n_k * (1 - n_k / N_h[k])   # fpc dentro de cada estrato
    return m, math.sqrt(v)


def asignar(n, N_h, S_h=None):
    """Reparto de n entre estratos: proporcional (S_h=None) o de Neyman (n_h ∝ N_h·S_h)."""
    p = np.asarray(N_h, float) * (1.0 if S_h is None else np.asarray(S_h, float))
    raw = n * p / p.sum(); out = np.floor(raw).astype(int)
    for k in np.argsort(raw - out)[::-1][: n - out.sum()]:   # restos mayores
        out[k] += 1
    return out


def efecto_diseno(m, icc):
    """Conglomerados de m elementos con correlación intraclase icc: Var se multiplica por esto."""
    return 1 + (m - 1) * icc


def berry_esseen(rho3_sigma3, n, C=0.4748):
    """Cota de Berry–Esseen para la distancia máxima entre la FDA de la media estandarizada y la normal."""
    return C * rho3_sigma3 / math.sqrt(n)
