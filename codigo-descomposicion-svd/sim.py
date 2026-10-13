"""Simulador de la flota de §8: km diarios de 300 camiones durante 4 semanas (28 días, empieza en lunes)."""
import numpy as np


def flota(seed=7, n_anom=20):
    rng = np.random.default_rng(seed)
    d = np.arange(28); dow = d % 7; sem = d // 7
    p = np.select([dow < 4, dow == 4, dow == 5], [1.0, 0.9, 0.35], 0.03)        # rutina semanal
    q = np.select([dow == 5, dow == 0], [1.0, -0.35], 0.0)                         # reparto: sábado sí, lunes flojo
    r = np.where(dow < 5, np.where(sem % 2 == 0, 1.0, -1.0), 0.0)                 # rotación de semanas alternas
    tipo = np.r_[np.zeros(120, int), np.ones(100, int), 2 * np.ones(80, int)]      # larga, regional, reparto
    a = np.select([tipo == 0, tipo == 1], [rng.uniform(520, 640, 300), rng.uniform(350, 450, 300)], rng.uniform(150, 230, 300))
    b = np.where(tipo == 2, rng.uniform(180, 240, 300), 0.0)
    c = np.zeros(300); c[:40] = rng.uniform(120, 180, 40)                           # 40 de larga distancia rotan
    verdad = np.outer(a, p) + np.outer(b, q) + np.outer(c, r)
    X = np.clip(verdad + rng.normal(0, 25, verdad.shape), 0, None)
    anom = np.zeros(X.shape, bool)
    filas = rng.choice(300, n_anom, replace=False)
    for k, i in enumerate(filas):
        if k < 10:                                                                 # avería: medio día parado
            j = rng.choice(np.where(dow < 5)[0]); X[i, j] *= rng.uniform(0.3, 0.5)
        elif k < 16:                                                               # salida en domingo
            j = rng.choice(np.where(dow == 6)[0]); X[i, j] = rng.uniform(150, 250)
        else:                                                                      # saltos de GPS: km × 1,6
            j = rng.choice(np.where(dow < 5)[0]); X[i, j] *= 1.6
        anom[i, j] = True
    return np.round(X), verdad, anom, tipo
