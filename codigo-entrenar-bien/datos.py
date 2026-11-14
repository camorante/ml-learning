"""Datos del tema 6. La zona normal del motor es la misma elipse del tema 3 (régimen en rpm y temperatura
del refrigerante en °C); aquí hay más lecturas y la frontera es borrosa, como en un taller de verdad.
Los camiones de la idea 2 son los del tema 4 (avería en 30 días según los km desde la revisión)."""
import numpy as np

RPM0, RPM_S, TMP0, TMP_S = 1400.0, 350.0, 89.0, 7.0      # centro y semiejes de la elipse «normal» (tema 3)
BORROSO = 0.06                                         # anchura de la frontera, en semiejes


def radio(rpm, tmp):
    """Distancia a la zona normal en «semiejes»: < 1 dentro de la elipse, > 1 fuera."""
    return np.hypot((np.asarray(rpm) - RPM0) / RPM_S, (np.asarray(tmp) - TMP0) / TMP_S)


def p_anomalia(rpm, tmp):
    """Probabilidad de que el taller marque la lectura como anomalía: 50 % justo en el borde de la elipse."""
    return 1 / (1 + np.exp(-(radio(rpm, tmp) - 1) / BORROSO))


def lecturas(n, seed):
    """n lecturas: dos tercios dentro de la elipse y un tercio en el anillo de radio 1–1,9 de alrededor,
    redondeadas como las da el sensor (10 rpm y 0,1 °C). La etiqueta del taller se sortea con p_anomalia."""
    rng = np.random.default_rng(seed)
    dentro = rng.random(n) < 2 / 3
    r = np.where(dentro, np.sqrt(rng.uniform(0, 1, n)), np.sqrt(rng.uniform(1, 1.9 ** 2, n)))
    a = rng.uniform(0, 2 * np.pi, n)
    rpm = np.round(RPM0 + RPM_S * r * np.cos(a), -1)
    tmp = np.round(TMP0 + TMP_S * r * np.sin(a), 1)
    y = (rng.random(n) < p_anomalia(rpm, tmp)).astype(int)
    return np.c_[rpm, tmp], y


X_TR, Y_TR = lecturas(600, seed=61)      # entrenamiento: 600 lecturas revisadas por el taller
X_VA, Y_VA = lecturas(400, seed=73)      # validación: 400 lecturas que la red no ve al entrenar
MEDIA, DESV = X_TR.mean(0), X_TR.std(0)  # el escalado se calcula SOLO con el entrenamiento


def escalar(X):
    return (np.asarray(X, float) - MEDIA) / DESV


def camiones(n, seed):
    """Tema 4: miles de km desde la última revisión y si hubo avería en los 30 días siguientes."""
    rng = np.random.default_rng(seed)
    x = np.round(rng.uniform(0, 60, n), 1)
    y = (rng.random(n) < 1 / (1 + np.exp(-(-5 + 0.08 * x)))).astype(int)
    return x, y
