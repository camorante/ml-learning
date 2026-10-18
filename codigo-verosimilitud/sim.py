"""Datos simulados de la flota para el tema de máxima verosimilitud (semillas fijas)."""
import numpy as np

# Alertas de frenada brusca por día de dos camiones durante 14 días (Poisson, semillas elegidas)
ALERTAS_A = np.random.default_rng(38).poisson(2.5, 14)      # ruta urbana: suma 35, media 2,5
ALERTAS_B = np.random.default_rng(1080).poisson(1.0, 14)    # autovía: suma 14, media 1,0


def alertas_largo(n=1000, lam=2.5, seed=7):
    """Un registro largo del camión A: los 14 días del juguete seguidos de n - 14 días más."""
    resto = np.random.default_rng(seed).poisson(lam, n - 14)
    return np.r_[ALERTAS_A, resto]


# Consumo (L/100 km) frente a carga (t) de 12 trayectos: recta + ruido normal de σ = 2
def consumo_carga(seed=4):
    r = np.random.default_rng(seed)
    carga = np.array([4, 6, 7, 9, 10, 12, 13, 15, 17, 19, 21, 24], float)
    consumo = np.round(24 + 0.45 * carga + r.normal(0, 2, carga.size), 1)
    return carga, consumo


# Minutos de descarga en un muelle: nueve paradas, una con un problema con el albarán
DESCARGA = np.array([18, 22, 15, 25, 20, 17, 23, 19, 95], float)


# Entregas: distancia (km) y si llegó tarde (1) o no (0); modelo logístico real
def entregas(n=40, seed=33, medio=170.0, incl=2.4):
    r = np.random.default_rng(seed)
    km = np.round(np.sort(r.uniform(20, 320, n)))
    p = 1 / (1 + np.exp(-incl * (km - medio) / 100))
    return km, (r.random(n) < p).astype(int)


# Conductores con distinta brusquedad: Poisson con λ que varía de un conductor a otro (Gamma)
def alertas_conductores(conductores=40, dias=30, media=2.0, forma=2.0, seed=5):
    r = np.random.default_rng(seed)
    lam = r.gamma(forma, media / forma, conductores)
    return r.poisson(np.repeat(lam, dias)), np.repeat(np.arange(conductores), dias)


# Paradas en el muelle de un cliente (minutos): lognormal como en el tema 8
def paradas(n=60, seed=3):
    return np.round(np.random.default_rng(seed).lognormal(np.log(12), 0.8, n), 1)
