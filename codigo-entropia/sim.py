"""Datos simulados de la flota para el tema de entropía (semillas fijas)."""
import numpy as np

# Parte diario de cada camión: tipo de incidencia (frecuencias de todo el año y del invierno)
TIPOS = ["sin incidencias", "retraso", "avería", "accidente"]
P_ANUAL = np.array([0.90, 0.06, 0.035, 0.005])
P_INVIERNO = np.array([0.78, 0.12, 0.08, 0.02])

# Mensajes de estado que manda la caja telemática (frecuencias en un día normal)
ESTADOS = ["en ruta", "parado en tráfico", "descargando", "repostando", "en taller", "alerta"]
P_ESTADOS = np.array([0.55, 0.20, 0.12, 0.07, 0.04, 0.02])

# Tipo de avería para el clasificador multiclase
AVERIAS = ["motor", "frenos", "neumáticos", "eléctrica"]


def partes(n=20_000, p=P_INVIERNO, seed=1):
    """n partes diarios con las frecuencias p (índices de TIPOS)."""
    return np.random.default_rng(seed).choice(len(p), n, p=p)


def camiones(n=1000, seed=15):
    """Un camión por fila: km, ruta urbana, matrícula par, temperatura del motor (°C) y si se averió.
    La avería depende de los km, de la ruta y de la temperatura en forma de U (fría o caliente)."""
    r = np.random.default_rng(seed)
    km = np.round(r.uniform(50, 800, n)) * 1000
    urbana = r.random(n) < 0.4
    par = r.random(n) < 0.5
    temp = np.round(r.normal(90, 8, n), 1)
    z = -5.0 + 2.0 * (km > 400_000) + 0.8 * urbana + 0.03 * (temp - 90) ** 2
    averia = r.random(n) < 1 / (1 + np.exp(-z))
    return dict(km=km, urbana=urbana, par=par, temp=temp, averia=averia.astype(int))


def semanas(n=6000, seed=4):
    """Semanas-camión: 3 variables, si hubo avería y el tipo que habría tenido (se usa solo si la hubo)."""
    r = np.random.default_rng(seed)
    X = np.c_[r.normal(0, 1, n), r.normal(0, 1, n), r.normal(0, 1, n)]    # vibración, temp., horas (estandarizadas)
    zb = -2.4 + 1.1 * X[:, 0] + 0.8 * X[:, 2]
    averia = (r.random(n) < 1 / (1 + np.exp(-zb))).astype(int)
    W = np.array([[1.2, 0.0, -0.4], [-0.3, 1.0, 0.2], [0.0, -0.8, 0.9], [-0.9, -0.2, -0.7]])
    lg = X @ W.T + np.array([0.3, 0.0, -0.2, -0.5])
    pr = np.exp(lg - lg.max(1, keepdims=True)); pr /= pr.sum(1, keepdims=True)
    tipo = np.array([r.choice(4, p=q) for q in pr])
    return X, averia, tipo


def velocidades(seed=8):
    """Velocidades (km/h) de un camión que alterna ciudad y autovía: mezcla de dos normales."""
    r = np.random.default_rng(seed)
    ciudad = r.random(5000) < 0.45
    return np.where(ciudad, r.normal(32, 8, 5000), r.normal(84, 6, 5000))
