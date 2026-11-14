"""Datos del tema 5: los 30 viajes de carga y consumo de los temas 1 y 3 (misma función, misma semilla)
y la red 2-2-1 del tema 3, pero colocada mal a propósito: la recta de la neurona B está en 18 L en vez de 21,5."""
import numpy as np
from retro import Densa, MLP


def carga_consumo(seed=11):
    """30 viajes: carga (t) y consumo (L/100 km). Lo normal es una franja diagonal;
    los que «no cuadran» (1) quedan por encima o por debajo de ella. (Copia literal del tema 1.)"""
    rng = np.random.default_rng(seed)
    c = np.round(rng.uniform(2, 24, 30), 1)
    base = 24 + 0.55 * c
    desv = np.r_[rng.uniform(-1.3, 1.3, 18), rng.uniform(3.6, 6, 6), -rng.uniform(3.6, 6, 6)]
    q = np.round(base + desv, 1)
    yy = np.r_[np.zeros(18, int), np.ones(12, int)]
    o = rng.permutation(30)
    return np.c_[c, q][o], yy[o]


# Red del tema 3 (0 errores): A vota si consumo > 26,5 + 0,55·carga; B si consumo < 21,5 + 0,55·carga.
# Aquí B empieza en 18 + 0,55·carga: 3,5 L demasiado abajo, y los 6 viajes que gastan de menos se escapan.
W1_T3, B1_T3 = [[-1.1, 2.0], [1.1, -2.0]], [-53.0, 43.0]
W1_0, B1_0 = [[-1.1, 2.0], [1.1, -2.0]], [-53.0, 36.0]
W2_0, B2_0 = [[10.0, 10.0]], [-5.0]
ETA = 0.03                     # tasa de aprendizaje del paso de ejemplo


def red_inicial():
    return MLP([Densa(W1_0, B1_0, "sigmoide"), Densa(W2_0, B2_0, "identidad")], "bce")
