"""Datos de flota del tema: las 20 jornadas revisadas a mano, 1 000 jornadas nuevas
y los viajes de carga y consumo (lo que una recta no puede separar)."""
import numpy as np

# 20 jornadas de conducción: (frenadas bruscas cada 100 km, horas al ralentí en la jornada)
# y la marca del responsable de flota: 1 = revisar con el conductor, 0 = bien.
JORNADAS = [
    (6.8, 1.9, 1), (1.2, 0.8, 0), (8.4, 0.9, 1), (2.9, 1.3, 0), (4.6, 2.6, 1),
    (4.1, 0.6, 0), (3.1, 3.4, 1), (1.8, 2.1, 0), (7.5, 2.8, 1), (5.4, 0.7, 0),
    (5.9, 1.5, 1), (3.6, 1.8, 0), (2.4, 3.9, 1), (0.7, 2.9, 0), (9.1, 1.6, 1),
    (2.5, 0.4, 0), (5.2, 3.5, 1), (6.6, 0.4, 0), (6.3, 1.1, 1), (1.0, 3.2, 0),
]
X = np.array([j[:2] for j in JORNADAS])
y = np.array([j[2] for j in JORNADAS])

# La primera idea del jefe de tráfico: 1 punto por frenada, 0,2 por hora de ralentí, empezar en −6
W_IDEA, B_IDEA = (1.0, 0.2), -6.0
ETA = 0.1


def nuevas(n=1000, seed=5):
    """Jornadas nuevas de la misma flota: la regla «real» es frenadas + 2·horas > 8,
    con algo de ruido (el criterio del responsable no es una fórmula exacta)."""
    rng = np.random.default_rng(seed)
    f = np.round(np.clip(rng.gamma(3, 1.3, n), 0.1, 9.9), 1)
    r = np.round(np.clip(rng.gamma(3, 0.55, n), 0.1, 4.9), 1)
    yy = ((f + 2 * r - 8 + rng.normal(0, 0.5, n)) > 0).astype(int)
    return np.c_[f, r], yy


def carga_consumo(seed=11):
    """30 viajes: carga (t) y consumo (L/100 km). Lo normal es una franja diagonal;
    los que «no cuadran» (1) quedan por encima o por debajo de ella."""
    rng = np.random.default_rng(seed)
    c = np.round(rng.uniform(2, 24, 30), 1)
    base = 24 + 0.55 * c
    desv = np.r_[rng.uniform(-1.3, 1.3, 18), rng.uniform(3.6, 6, 6), -rng.uniform(3.6, 6, 6)]
    q = np.round(base + desv, 1)
    yy = np.r_[np.zeros(18, int), np.ones(12, int)]
    o = rng.permutation(30)
    return np.c_[c, q][o], yy[o]


XOR = (np.array([[0, 0], [0, 1], [1, 0], [1, 1]], float), np.array([0, 1, 1, 0]))
