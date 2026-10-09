"""Viajes de una flota con los problemas típicos que debe encontrar un análisis exploratorio."""
import numpy as np, pandas as pd

TIPOS = {"ligero": dict(v=74, base=17.0, carga=(0.5, 3.5)),
         "mediano": dict(v=62, base=25.0, carga=(2, 9)),
         "pesado": dict(v=52, base=33.0, carga=(8, 24))}

def viajes(n=2400, seed=1, problemas=True):
    rng = np.random.default_rng(seed); filas = []
    tipos = rng.choice(list(TIPOS), n, p=[0.3, 0.4, 0.3])
    for i, t in enumerate(tipos):
        P = TIPOS[t]
        v = float(np.clip(rng.normal(P["v"], 8), 25, 105))
        dist = float(np.exp(rng.normal(np.log(45), 0.75)))                 # muy asimétrica: muchos cortos, pocos larguísimos
        carga = float(rng.uniform(*P["carga"]))
        ral = 100 * float(rng.beta(2, 9))
        temp = float(rng.normal(88, 4))
        # dentro de cada tipo, más velocidad -> más consumo (resistencia del aire)
        cons = P["base"] + 0.0035 * (v - 40) ** 2 * (1 + carga / 30) + 0.45 * carga + 0.06 * ral + rng.normal(0, 1.3)
        km_gps = dist * float(rng.normal(1, 0.004))
        filas.append(dict(tipo=t, distancia_km=dist, km_gps=km_gps, vel_media=v, duracion_h=dist / v,
                          carga_t=carga, ralenti_pct=ral, temp_motor=temp, consumo=cons, litros=cons * dist / 100))
    df = pd.DataFrame(filas)
    if problemas:
        k = rng.choice(n, 14, replace=False); df.loc[k, "vel_media"] = 255.0          # código de error del equipo
        k = rng.choice(n, 60, replace=False); df.loc[k, "temp_motor"] = -40.0         # sensor sin lectura
        k = rng.choice(n, 25, replace=False); df.loc[k, "carga_t"] = -rng.uniform(0.2, 1.5, 25)   # báscula mal calibrada
    return df
