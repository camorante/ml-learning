"""Flota simulada para el tema 10 (covarianza y correlación). Todo con semillas fijas."""
import numpy as np

# Los 8 camiones del ejemplo resuelto: carga media (t) y consumo (L/100 km)
CARGA8 = np.array([6, 9, 11, 14, 16, 19, 21, 24], float)
CONSUMO8 = np.array([27, 30, 29, 33, 31, 35, 36, 35], float)


def juguete(n=120, seed=10):
    """Cuatro parejas de columnas de la flota (un camión por fila, datos de un mes)."""
    r = np.random.default_rng(seed)
    carga = r.uniform(4, 26, n)
    consumo = 24 + 0.5 * carga + r.normal(0, 2.6, n)
    ralenti = r.uniform(5, 80, n)                                   # horas de ralentí al mes
    gasto = 3900 + 4.1 * ralenti + r.normal(0, 140, n)              # € de gasóleo al mes
    exper = r.uniform(1, 30, n)                                     # años de experiencia del conductor
    frenadas = np.clip(9 - 0.18 * exper + r.normal(0, 1.6, n), 0.2, None)   # frenadas bruscas cada 1 000 km
    dia = r.integers(1, 31, n).astype(float)                        # día del mes del repostaje
    consumo2 = 32 + r.normal(0, 2.4, n)
    return dict(carga=(carga, consumo), ralenti=(ralenti, gasto), exper=(exper, frenadas), dia=(dia, consumo2))


def velocidad(n=160, seed=13):
    """Velocidad media del trayecto (km/h) y consumo: curva en U con el mínimo en 50 km/h."""
    r = np.random.default_rng(seed)
    v = r.uniform(10, 90, n)
    c = 29 + 0.005 * (v - 50) ** 2 + r.normal(0, 1.2, n)
    return v, c


def atipico(n=20, seed=33):
    """Dos nubes de 20 camiones: sin relación (edad del conductor y consumo) y con relación (carga y consumo)."""
    r = np.random.default_rng(seed)
    edad = r.uniform(24, 60, n)
    cons = 32 + r.normal(0, 1.6, n)
    carga = r.uniform(6, 24, n)
    cons2 = 24 + 0.5 * carga + r.normal(0, 1.4, n)
    return (edad, cons), (carga, cons2)


def confusion(n=240, seed=13):
    """Edad del camión, tipo de ruta (1 = ciudad) y consumo. La ruta decide el consumo; la edad decide la ruta."""
    r = np.random.default_rng(seed)
    edad = r.uniform(1, 15, n)
    ciudad = (r.random(n) < 1 / (1 + np.exp(-(edad - 8) / 1.6))).astype(int)
    cons = np.where(ciudad == 1, 37.5, 30.0) + r.normal(0, 1.8, n)
    return edad, ciudad, cons


NOMBRES = ["km del mes", "litros", "€ de gasóleo", "carga media", "ralentí", "edad del camión",
           "frenadas bruscas", "letra de la matrícula", "día de nacimiento del conductor"]


def flota(n=500, seed=14):
    """Tabla mensual de n camiones con 9 columnas (las dos últimas son puro azar)."""
    r = np.random.default_rng(seed)
    km = r.uniform(4000, 12000, n)
    carga = r.uniform(6, 24, n)
    edad = r.uniform(1, 15, n)
    ralenti = np.clip(20 + 2.2 * edad + r.normal(0, 12, n), 2, None)
    l100 = 24 + 0.5 * carga + 0.25 * edad + r.normal(0, 2.0, n)
    litros = km * l100 / 100 + 2.5 * ralenti
    euros = litros * r.normal(1.65, 0.03, n)
    frenadas = np.clip(6 - 0.00025 * km + r.normal(0, 1.2, n), 0.1, None)
    letra = r.integers(1, 27, n).astype(float)
    nac = r.integers(1, 32, n).astype(float)
    X = np.column_stack([km, litros, euros, carga, ralenti, edad, frenadas, letra, nac])
    return X
