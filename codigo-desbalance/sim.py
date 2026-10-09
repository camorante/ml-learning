"""Semanas-camión con una avería rara (~2 %) y cinco señales del equipo."""
import numpy as np, pandas as pd

def semanas(n=20000, seed=6):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(dict(
        vibracion=rng.gamma(2.0, 1.0, n),                 # índice de vibración del motor
        temp_extra=rng.normal(0, 3, n),                   # °C por encima de lo normal
        horas_mant=rng.uniform(0, 900, n),                # horas desde el último mantenimiento
        codigos=rng.poisson(0.4, n),                      # códigos de fallo leves esta semana
        edad=rng.uniform(1, 14, n)))                      # años del camión
    z = (-8.4 + 0.75 * df.vibracion + 0.18 * df.temp_extra + 0.0022 * df.horas_mant
         + 0.7 * df.codigos + 0.09 * df.edad + rng.normal(0, 0.6, n))
    df["averia"] = rng.random(n) < 1 / (1 + np.exp(-z))
    return df
