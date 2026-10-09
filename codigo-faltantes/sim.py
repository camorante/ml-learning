"""Descargas de una flota con esperas que a veces no se declaran, y un sensor que falla."""
import numpy as np, pandas as pd

CLIENTES = {"almacén grande": (0.35, 48), "tienda pequeña": (0.40, 18), "obra": (0.25, 32)}

def descargas(n=2000, seed=5):
    rng = np.random.default_rng(seed)
    cli = rng.choice(list(CLIENTES), n, p=[v[0] for v in CLIENTES.values()])
    base = np.array([CLIENTES[c][1] for c in cli])
    palets = np.clip(np.round(rng.gamma(2.2, 4.0, n) * np.where(cli == "tienda pequeña", 0.5, 1.0)), 1, 33)
    espera = base * np.exp(rng.normal(0, 0.45, n)) + 1.1 * palets         # minutos
    df = pd.DataFrame(dict(cliente=cli, palets=palets, espera=espera.round(1)))
    u = rng.random(n)
    # tres formas de que falte la espera (cada una con ~30 % de huecos)
    df["falta_mcar"] = u < 0.30
    p_mar = np.where(cli == "tienda pequeña", 0.55, np.where(cli == "obra", 0.20, 0.10))
    df["falta_mar"] = u < p_mar                                               # depende del cliente (visible)
    p_mnar = 1 / (1 + np.exp((espera - 30) / 8)) * 0.85                       # esperas cortas no se declaran
    df["falta_mnar"] = u < p_mnar
    # sensor de temperatura y averías: el sensor deja de enviar cuando el equipo eléctrico falla
    fallo_elec = rng.random(n) < 0.12
    temp = rng.normal(88, 4, n) + 6 * (rng.random(n) < 0.07)
    p_av = 1 / (1 + np.exp(-(-3.6 + 0.3 * (temp - 88) + 2.8 * fallo_elec)))
    df["temp_motor"] = temp.round(1); df["averia_30d"] = rng.random(n) < p_av
    df["falta_temp"] = rng.random(n) < np.where(fallo_elec, 0.75, 0.04)
    df["fallo_elec"] = fallo_elec
    return df
