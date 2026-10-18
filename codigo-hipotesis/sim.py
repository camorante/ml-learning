"""Datos simulados de la flota para el tema 13 (pruebas de hipótesis)."""
import numpy as np

# El juguete: consumo medio del mes siguiente al curso (L/100 km), 10 camiones por grupo.
# Generados con N(32; 1,6) y N(31; 1,6) (el curso ahorra de verdad 1,0) y redondeados.
SIN_CURSO = np.array([31.5, 30.7, 32.2, 30.9, 32.3, 35.3, 32.7, 33.2, 30.7, 30.5])
CON_CURSO = np.array([32.3, 29.5, 29.5, 31.7, 30.3, 32.0, 29.6, 32.1, 27.4, 29.6])
SIGMA, EFECTO = 1.6, 1.0          # dispersión entre camiones y ahorro real (L/100 km)
KM_ANIO, EUR_L = 100_000, 1.65    # km al año por camión y precio del gasóleo


def euros_anio(ahorro_l100):
    """Ahorro en € por camión y año para un ahorro en L/100 km."""
    return ahorro_l100 * KM_ANIO / 100 * EUR_L


def prueba_ab(n, efecto=EFECTO, sigma=SIGMA, mu=32.0, rng=None):
    """Una prueba A/B: n camiones sin curso y n con curso."""
    rng = rng or np.random.default_rng()
    return rng.normal(mu, sigma, n), rng.normal(mu - efecto, sigma, n)


def ralenti(n=2000, p_a=0.18, p_b=0.15, seed=48):
    """Trayectos con más de 10 min de ralentí (1) o no (0), sin aviso (A) y con aviso en la app (B)."""
    rng = np.random.default_rng(seed)
    return (rng.random(n) < p_a).astype(int), (rng.random(n) < p_b).astype(int)


METRICAS = ["consumo", "ralentí", "frenazos", "acelerones", "velocidad media", "exceso de velocidad",
            "km en vacío", "paradas", "tiempo en ruta", "retrasos", "averías", "neumáticos",
            "AdBlue", "temperatura motor", "revoluciones", "giros bruscos", "uso de crucero",
            "horas de motor", "repostajes", "quejas de clientes"]
