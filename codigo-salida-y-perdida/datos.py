"""Datos del tema 4: un viaje con tres cabezas de salida y los conjuntos de las ideas."""
import numpy as np

CLASES = ["urbano", "regional", "autovía"]

# El viaje del juguete y del ejemplo resuelto: lo que sacan las tres cabezas y lo que pasó de verdad
VIAJE = dict(consumo_pred=31.2, consumo_real=33.0,          # L/100 km
             z_averia=-1.2, averia=1,                       # logit de «avería en 30 días»; sí hubo avería
             logits=[0.4, 1.6, 0.9], tipo=1)                # urbano, regional, autovía; era regional
SEGURO_Y_EQUIVOCADO = dict(consumo_pred=46.0, z_averia=-4.6, logits=[3.4, -0.8, 0.6])

# Idea 3: consumo de 10 viajes de la misma ruta (L/100 km); el décimo lo leyó mal el sensor del depósito
RUTA = [29.8, 31.2, 32.0, 32.5, 33.1, 33.6, 34.4, 35.0, 36.3]
ATIPICO = 95.0

# Idea 4: 10 camiones, probabilidad de avería en 30 días que dio el modelo y lo que pasó
P_LOTE = [0.08, 0.12, 0.05, 0.20, 0.65, 0.80, 0.10, 0.03, 0.30, 0.01]
Y_LOTE = [0, 0, 0, 0, 1, 1, 0, 0, 0, 1]


def p_verdad(x):
    """Probabilidad real de avería en 30 días según los miles de km desde la última revisión."""
    return 1 / (1 + np.exp(-(-5 + 0.08 * np.asarray(x, float))))


def camiones(n, seed):
    rng = np.random.default_rng(seed)
    x = np.round(rng.uniform(0, 60, n), 1)                  # miles de km desde la revisión
    y = (rng.random(n) < p_verdad(x)).astype(int)
    return x, y


# Costes de la idea 7 (€)
COSTE_AVERIA = 4000      # avería en carretera: grúa, reparación urgente y la entrega perdida
COSTE_REVISION = 300     # revisión preventiva que no hacía falta: taller y medio día parado
