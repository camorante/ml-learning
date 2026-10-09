import numpy as np, pandas as pd

def flota(n_camiones=60, dias=400, seed=0):
    """Registros diarios de una flota: una fila por camión y día. Devuelve un DataFrame."""
    rng = np.random.default_rng(seed); filas = []
    for c in range(n_camiones):
        sesgo = rng.normal(0, 2.0)                       # eficiencia propia del camión (persistente)
        odo = rng.uniform(50_000, 600_000)               # odómetro inicial: distinto en cada camión
        estado = 0.0                                     # desgaste/mantenimiento: cambia despacio
        for d in range(dias):
            estado = 0.97 * estado + rng.normal(0, 0.45)
            km = max(40, rng.normal(320, 80)); odo += km
            carga = rng.uniform(0, 24); ralenti = 100 * rng.beta(2, 8)
            pendiente = rng.normal(0, 1.2); temp = 15 + 10 * np.sin(2 * np.pi * (d - 100) / 365) + rng.normal(0, 3)
            consumo = (24 + 0.35 * carga + 0.05 * ralenti + 1.5 * pendiente + 0.06 * abs(temp - 18)
                       + sesgo + estado + 0.006 * d + rng.normal(0, 1.2))
            litros_manana = consumo * km / 100 * rng.normal(1, 0.03)   # repostaje del día siguiente: ¡información del futuro!
            filas.append(dict(camion=c, dia=d, km=km, carga=carga, ralenti=ralenti, pendiente=pendiente,
                              temp=temp, odometro=odo, litros_manana=litros_manana, consumo=consumo))
    return pd.DataFrame(filas)
