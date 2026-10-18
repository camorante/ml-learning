"""Población simulada: las 12 000 paradas de un año de una flota de 60 camiones.
Tres tipos de ruta (estratos) con duraciones muy distintas; cada camión tiene su propio
«estilo» (efecto de conglomerado) y sus paradas consecutivas se parecen (autocorrelación)."""
import numpy as np

RUTAS = ["autovía", "ciudad", "puerto"]
CAMIONES = [30, 20, 10]          # camiones por tipo de ruta
MEDIANA = [12, 25, 60]           # minutos
PARADAS = 200                    # paradas por camión en el año
RHO = 0.6                        # autocorrelación entre paradas consecutivas (escala log)


def poblacion(seed=11):
    rng = np.random.default_rng(seed)
    dur, ruta, camion = [], [], []
    c = 0
    for h, (nc, med) in enumerate(zip(CAMIONES, MEDIANA)):
        for _ in range(nc):
            u = rng.normal(0, 0.25)                  # efecto del camión
            e = np.empty(PARADAS); e[0] = rng.normal()
            for t in range(1, PARADAS):              # AR(1) de varianza 1
                e[t] = RHO * e[t - 1] + np.sqrt(1 - RHO**2) * rng.normal()
            x = med * np.exp(u + 0.55 * e)
            dur.append(np.maximum(1, np.round(x)))
            ruta += [h] * PARADAS; camion += [c] * PARADAS; c += 1
    return np.concatenate(dur), np.array(ruta), np.array(camion)
