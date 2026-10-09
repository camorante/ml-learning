"""Un día de telemetría de un camión, con los errores de registro típicos."""
import numpy as np, pandas as pd

LAT0, LON0 = 40.40, -3.70                      # origen de la proyección local
KM_LAT = 111.32; KM_LON = 111.32 * np.cos(np.radians(LAT0))

def a_latlon(x, y):
    return LAT0 + y / KM_LAT, LON0 + x / KM_LON

def ruta(seed=7):
    """Trayectoria real cada 30 s de 06:00 a 16:00 con dos paradas."""
    rng = np.random.default_rng(seed)
    t = np.arange(6 * 3600, 16 * 3600, 30.0)
    vel = np.zeros(len(t)); rumbo = np.zeros(len(t)); v = 0; h = 0.6
    paradas = [(9.5 * 3600, 10.0 * 3600), (11.0 * 3600, 11.25 * 3600)]   # descarga y repostaje
    for i, ti in enumerate(t):
        parado = any(a <= ti < b for a, b in paradas) or ti < 6 * 3600 + 300
        objetivo = 0 if parado else (82 if (7.2 * 3600 < ti < 9.2 * 3600 or 13 * 3600 < ti < 15 * 3600) else 42)
        v += 0.25 * (objetivo - v) + (0 if parado else rng.normal(0, 3)); v = max(0, v) if not parado else 0
        h += rng.normal(0, 0.06) if v > 5 else 0
        vel[i], rumbo[i] = v, h
    dx = vel * 30 / 3600 * np.cos(rumbo); dy = vel * 30 / 3600 * np.sin(rumbo)
    x, y = np.cumsum(dx), np.cumsum(dy)
    odo = 182_400 + np.cumsum(vel * 30 / 3600)
    comb = 70 - 0.05 * (odo - odo[0])                           # % del depósito (600 L, 30 L/100 km)
    i_rep = np.searchsorted(t, 11.1 * 3600); comb[i_rep:] += 38 # repostaje
    return pd.DataFrame(dict(t_real=t, x=x, y=y, vel=vel, odo=odo, combustible=np.clip(comb, 3, 100)))

def registros(seed=7, deriva_s_por_h=40.0):
    rng = np.random.default_rng(seed + 1); r = ruta(seed)
    lat, lon = a_latlon(r.x.values, r.y.values)
    # reloj del equipo: adelanta 'deriva' segundos por hora desde las 06:00
    t_eq = r.t_real + deriva_s_por_h * (r.t_real - 6 * 3600) / 3600
    m = pd.DataFrame(dict(t_equipo=np.round(t_eq, 1), lat=lat, lon=lon, vel=np.round(r.vel, 1),
                          odo=np.round(r.odo, 2), combustible=np.round(r.combustible, 1)))
    lat_red = rng.exponential(2.0, len(m)) + 0.5                       # retraso de red, s
    m["t_servidor"] = r.t_real + lat_red
    # zona sin cobertura 12:10-12:40: se guardan y se envían intercalados con los nuevos al volver
    sin = (r.t_real >= 12 * 3600 + 600) & (r.t_real < 12 * 3600 + 2400)
    k = np.where(sin)[0]; m.loc[k, "t_servidor"] = 12 * 3600 + 2405 + np.arange(len(k)) * 9.5 + rng.uniform(0, 2, len(k))
    # saltos de GPS: 6 posiciones desplazadas 3-8 km
    m["salto"] = False
    for i in rng.choice(np.where(r.vel > 30)[0], 6, replace=False):
        m.loc[i, "salto"] = True
        a = rng.uniform(0, 2 * np.pi); d = rng.uniform(3, 8)
        m.loc[i, "lat"] += d * np.sin(a) / KM_LAT; m.loc[i, "lon"] += d * np.cos(a) / KM_LON
    # sensor de combustible congelado 13:00-15:00
    fr = (r.t_real >= 13 * 3600) & (r.t_real < 15 * 3600); j = np.where(fr)[0]
    m.loc[j, "combustible"] = m.loc[j[0], "combustible"]
    m["congelado"] = fr.values
    # duplicados por reintento: 5 % de los mensajes se reenvían con otro id y otra hora de llegada
    d = rng.choice(len(m), int(0.05 * len(m)), replace=False)
    dup = m.loc[d].copy(); dup["t_servidor"] += rng.uniform(5, 90, len(d))
    m["reenvio"] = False; dup["reenvio"] = True
    out = pd.concat([m, dup]).sort_values("t_servidor").reset_index(drop=True)
    out.insert(0, "id_mensaje", np.arange(100_000, 100_000 + len(out)))
    return out, r
