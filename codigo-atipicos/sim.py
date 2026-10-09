"""Viajes con valores atípicos de dos clases: errores y casos raros pero reales."""
import numpy as np, pandas as pd

def viajes(n=800, seed=3):
    rng = np.random.default_rng(seed)
    carga = rng.uniform(0, 24, n)
    ruta = np.where(rng.random(n) < 0.025, "montaña", "llana")
    frio = rng.random(n) < 0.012
    cons = 22 + 0.9 * carga + 7.5 * (ruta == "montaña") + 5.0 * frio + rng.normal(0, 1.6, n)
    df = pd.DataFrame(dict(carga_t=carga.round(2), ruta=ruta, frigorifico=frio, consumo=cons.round(1)))
    df["tipo"] = np.where(ruta == "montaña", "raro: montaña", np.where(frio, "raro: frigorífico", "normal"))
    # errores de registro
    def err(k, tipo, f):
        idx = rng.choice(np.where(df.tipo == "normal")[0], k, replace=False)
        for i in idx: f(i)
        df.loc[idx, "tipo"] = tipo
    err(4, "error: coma desplazada", lambda i: df.__setitem__("consumo", df.consumo.where(df.index != i, round(df.consumo[i] * 10, 1))))
    err(3, "error: galones", lambda i: df.__setitem__("consumo", df.consumo.where(df.index != i, round(df.consumo[i] / 3.785, 1))))
    idx = rng.choice(np.where((df.tipo == "normal") & (df.carga_t < 4))[0], 5, replace=False)   # carga declarada pero viajó vacío
    df.loc[idx, "carga_t"] = rng.uniform(18, 23, 5).round(2); df.loc[idx, "tipo"] = "error: carga mal declarada"
    df["es_error"] = df.tipo.str.startswith("error"); df["es_raro"] = df.tipo.str.startswith("raro")
    return df
