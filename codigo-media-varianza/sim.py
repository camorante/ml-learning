"""Flota simulada: 300 camiones × 80 viajes, consumo (L/100 km) y ralentí (min), con registros erróneos."""
import numpy as np, pandas as pd


def viajes(seed=1, camiones=300, n=80, p_error=0.004):
    rng = np.random.default_rng(seed)
    base = rng.normal(32, 2.0, camiones)                          # consumo propio de cada camión
    cam = np.repeat(np.arange(camiones), n)
    dia = np.tile(np.arange(n) // 4, camiones)                    # 4 viajes por día, 20 días
    consumo = base[cam] + rng.normal(0, 1.8, len(cam))
    ralenti = rng.gamma(2.0, 9.0, len(cam))                       # minutos de motor al ralentí por viaje
    malo = rng.random(len(cam)) < p_error                         # coma perdida: 33.4 se guarda como 334
    consumo[malo] *= 10
    return pd.DataFrame(dict(camion=cam, dia=dia, consumo=consumo.round(1), ralenti=ralenti.round(0), error=malo)), base
