"""Tabla de juguete de la versión sencilla: km diarios de 12 camiones durante 14 días (dos semanas)."""
import numpy as np

DIAS = ["L", "M", "X", "J", "V", "S", "D"] * 2


def tabla(seed=1):
    rng = np.random.default_rng(seed)
    p = np.array([1.0, 1.0, 1.0, 1.0, 0.9, 0.35, 0.03] * 2)          # rutina semanal: forma de la semana
    a = np.r_[rng.uniform(540, 620, 5), rng.uniform(380, 440, 3), rng.uniform(170, 220, 4)]  # km de un laborable
    q = np.array([-0.35, 0, 0, 0, 0, 1.0, 0] * 2)                    # reparto urbano: sábado fuerte, lunes flojo
    b = np.r_[np.zeros(8), rng.uniform(200, 240, 4)]                 # solo los camiones 9 a 12
    verdad = np.outer(a, p) + np.outer(b, q)
    X = np.clip(verdad + rng.normal(0, 22, verdad.shape), 0, None)   # variación diaria (atascos, desvíos)
    anom = [(3, 10, 120.0), (10, 6, 330.0)]   # camión 4: jueves 2.ª semana casi parado; camión 11: sale un domingo
    for i, j, v in anom:
        X[i, j] = v
    return np.round(X), verdad, anom
