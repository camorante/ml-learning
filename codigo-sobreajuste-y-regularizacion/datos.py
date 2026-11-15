"""Datos del tema 7: el consumo de un camión (L/100 km) en cada viaje según su velocidad media (km/h).
El taller instaló un caudalímetro de precisión durante unas semanas: hay pocos viajes medidos y cada medida trae
ruido (carga, viento, conductor). Entrenamiento, validación y prueba se separan ANTES de mirar nada (D1)."""
import numpy as np

V0, V1 = 15.0, 90.0          # velocidades medias posibles (km/h)
RUIDO = 2.0                  # desviación típica del ruido de cada viaje (L/100 km)


def verdad(v):
    """El consumo medio de verdad: alto en ciudad (paradas), mínimo hacia 46 km/h, sube con la resistencia del aire."""
    v = np.asarray(v, float)
    return 20 + 300 / v + 0.0015 * v ** 2


def viajes(n, seed):
    """n viajes: velocidad media al azar entre 15 y 90 km/h (0,1 km/h) y consumo medido = verdad + ruido (0,1 L)."""
    rng = np.random.default_rng(seed)
    v = np.round(rng.uniform(V0, V1, n), 1)
    return v, np.round(verdad(v) + RUIDO * rng.standard_normal(n), 1)


V_POOL, Y_POOL = viajes(300, seed=7)  # todos los viajes medidos para entrenar que podría llegar a haber
V_TR, Y_TR = V_POOL[:20], Y_POOL[:20]  # entrenamiento: los 20 primeros viajes medidos
V_VA, Y_VA = viajes(30, seed=17)      # validación: otros 30, para elegir (épocas, regularización)
V_TE, Y_TE = viajes(200, seed=27)     # prueba: los 200 viajes del mes siguiente, se miran UNA vez al final
MV, SV = V_TR.mean(), V_TR.std()      # escalado de la entrada y de la salida, solo con el entrenamiento (D7)
MY, SY = Y_TR.mean(), Y_TR.std()


def ex(v):
    return ((np.asarray(v, float) - MV) / SV)[:, None]


def ey(y):
    return (np.asarray(y, float) - MY) / SY


def rmse(pred_std, y):
    """Error cuadrático medio, en L/100 km, de unas predicciones hechas en la escala estandarizada."""
    return float(np.sqrt(np.mean((pred_std * SY + MY - np.asarray(y, float)) ** 2)))


REJILLA = np.linspace(V0, V1, 61)     # donde se dibuja la función que aprende la red
