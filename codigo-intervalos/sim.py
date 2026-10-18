"""Flota simulada para el tema 12 (intervalos de confianza y bootstrap). Todo con semillas fijas."""
import numpy as np

MU, SIGMA = 32.0, 4.5          # consumo real de los trayectos de este modelo de camión (L/100 km)


def trayectos(n=1000, seed=2336):
    """Consumo (L/100 km) de n trayectos; los 10 primeros son los del ejemplo resuelto."""
    return np.round(np.random.default_rng(seed).normal(MU, SIGMA, n), 1)


DIEZ = trayectos()[:10]                  # 30.9 31.1 37.0 40.7 30.5 30.6 25.2 37.2 30.4 30.4

# Entregas tarde: 1 de 40 en el último mes de un cliente nuevo
TARDE, ENTREGAS = 1, 40

# 12 viajes de un camión de reparto: km y coste (gasóleo + peajes, en €); el último es un viaje largo
KM12 = np.array([42, 55, 61, 48, 73, 88, 51, 59, 67, 46, 57, 395], float)
EUR12 = np.array([31, 40, 47, 33, 52, 71, 36, 45, 50, 32, 41, 318], float)


def camiones(k=8, m=15, seed=21, efecto=2.5, ruido=3.0):
    """k camiones con m trayectos cada uno: cada camión tiene su propio consumo de base."""
    r = np.random.default_rng(seed)
    base = MU + r.normal(0, efecto, k)
    x = np.round(base[:, None] + r.normal(0, ruido, (k, m)), 1)
    return x.ravel(), np.repeat(np.arange(k), m)


def picos(n=20, seed=8):
    """Velocidad punta de n trayectos (km/h): el limitador corta a 90, así que el máximo real es 90."""
    return np.round(np.random.default_rng(seed).uniform(78, 90, n), 1)


def depositos(n=200, seed=160):
    """Consumo por trayecto en dos depósitos; la diferencia real es 1,4 L/100 km."""
    r = np.random.default_rng(seed)
    return np.round(r.normal(31.2, 4.5, n), 1), np.round(r.normal(32.6, 4.5, n), 1)


def carga_consumo(n=200, seed=12):
    """Carga media (t) y consumo (L/100 km) de n camiones, para la banda de confianza y la de predicción."""
    r = np.random.default_rng(seed)
    carga = np.round(r.uniform(4, 26, n), 1)
    return carga, np.round(24 + 0.5 * carga + r.normal(0, 2.6, n), 1)
