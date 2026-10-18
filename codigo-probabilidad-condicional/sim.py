import numpy as np, pandas as pd
from cond import chi2


def viajes(n=200_000, seed=9):
    """Viajes simulados: lluvia (20 %), conductor novato (30 %, independiente de la lluvia),
    alerta de frenada brusca según lluvia y conductor, y exceso de velocidad según el conductor."""
    r = np.random.default_rng(seed)
    lluvia = r.random(n) < 0.20
    novato = r.random(n) < 0.30
    p_alerta = np.select([~lluvia & ~novato, ~lluvia & novato, lluvia & ~novato], [0.07, 0.17, 0.19], 0.39)
    alerta = r.random(n) < p_alerta
    exceso = r.random(n) < np.where(novato, 0.30, 0.10)
    return pd.DataFrame(dict(lluvia=lluvia, novato=novato, alerta=alerta, exceso=exceso))
