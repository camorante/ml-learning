"""Probabilidad condicional e independencia, desde cero (solo numpy y math)."""
import math
import numpy as np


def tabla(a, b):
    """Tabla de contingencia: filas = valores de a, columnas = valores de b (recuentos)."""
    fa, ia = np.unique(np.asarray(a), return_inverse=True)
    fb, ib = np.unique(np.asarray(b), return_inverse=True)
    T = np.zeros((len(fa), len(fb)), int)
    np.add.at(T, (ia, ib), 1)
    return T, fa, fb


def conjunta(T):
    """P(fila = i y columna = j): cada casilla entre el total."""
    T = np.asarray(T, float); return T / T.sum()


def marginales(T):
    """P(fila = i) y P(columna = j): sumar la conjunta por filas y por columnas."""
    P = conjunta(T); return P.sum(1), P.sum(0)


def condicional(T, dado="filas"):
    """P(columna | fila) si dado='filas' (cada fila suma 1); P(fila | columna) si dado='columnas'."""
    T = np.asarray(T, float)
    return T / T.sum(1, keepdims=True) if dado == "filas" else T / T.sum(0, keepdims=True)


def prob_total(p_b, p_a_dado_b):
    """P(A) = sum_j P(B=j) P(A | B=j)."""
    return float(np.dot(p_b, p_a_dado_b))


def bayes(p_b, p_a_dado_b):
    """P(B=j | A) a partir de P(B=j) y de P(A | B=j): producto y normalizar."""
    num = np.asarray(p_b, float) * np.asarray(p_a_dado_b, float)
    return num / num.sum()


def esperada(T):
    """Recuentos que habría si filas y columnas fueran independientes: n * P(fila) * P(columna)."""
    T = np.asarray(T, float); return np.outer(T.sum(1), T.sum(0)) / T.sum()


def _q_gamma(a, x):
    """Gamma incompleta superior regularizada Q(a, x) (serie o fracción continua de Lentz)."""
    if x <= 0:
        return 1.0
    lg = a * math.log(x) - x - math.lgamma(a)
    if x < a + 1:                                    # serie para P(a, x)
        s = t = 1.0 / a; k = a
        while abs(t) > 1e-17 * abs(s):
            k += 1; t *= x / k; s += t
        return 1.0 - s * math.exp(lg)
    b = x + 1 - a; c = 1e300; d = 1 / b; h = d       # fracción continua para Q(a, x)
    for i in range(1, 10000):
        an = -i * (i - a); b += 2
        d = an * d + b; d = 1e-300 if abs(d) < 1e-300 else d
        c = b + an / c; c = 1e-300 if abs(c) < 1e-300 else c
        d = 1 / d; h *= d * c
        if abs(d * c - 1) < 1e-16:
            break
    return math.exp(lg) * h


def chi2(T, yates=False):
    """Prueba de independencia: estadístico, grados de libertad y valor p."""
    T = np.asarray(T, float); E = esperada(T); d = np.abs(T - E)
    if yates and T.shape == (2, 2):
        d = np.maximum(d - 0.5, 0)
    est = float((d**2 / E).sum()); gl = (T.shape[0] - 1) * (T.shape[1] - 1)
    return est, gl, _q_gamma(gl / 2, est / 2)


def cramer_v(T):
    """Tamaño del efecto entre 0 (independientes) y 1: sqrt(chi2 / (n (min(r, c) - 1)))."""
    T = np.asarray(T, float); est = chi2(T)[0]
    return math.sqrt(est / (T.sum() * (min(T.shape) - 1)))
