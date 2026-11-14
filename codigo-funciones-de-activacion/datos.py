"""Datos de la flota del tema: consumo (L/100 km) frente a velocidad media del tramo (km/h)."""
import numpy as np

def verdad(v):
    """Curva en U: arranques y paradas en ciudad, mínimo hacia 75 km/h, el aire en autovía."""
    v = np.asarray(v, float)
    return 29 + 0.0045 * (v - 75) ** 2 + 18 * np.exp(-v / 18)

def tramos(n=60, seed=2):
    rng = np.random.default_rng(seed)
    v = np.sort(np.round(rng.uniform(12, 110, n), 1))
    c = np.round(verdad(v) + rng.normal(0, 1.2, n), 1)
    return v, c
