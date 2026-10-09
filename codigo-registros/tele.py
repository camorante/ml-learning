"""Limpieza de registros de telemetría desde cero."""
import numpy as np, pandas as pd

R_TIERRA = 6371.0088          # radio medio (km), el que usa scikit-learn/IUGG

def haversine(lat1, lon1, lat2, lon2):
    """Distancia por la superficie de una esfera, en km."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi, dlmb = p2 - p1, np.radians(lon2) - np.radians(lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return 2 * R_TIERRA * np.arcsin(np.sqrt(a))

def distancia_total(df):
    return float(np.sum(haversine(df.lat.values[:-1], df.lon.values[:-1], df.lat.values[1:], df.lon.values[1:])))

def quitar_duplicados(df, clave=("t_equipo", "lat", "lon", "vel", "odo")):
    """Un reenvío repite el contenido con otro id y otra hora de llegada: se identifica por el contenido."""
    return df.sort_values("t_servidor").drop_duplicates(subset=list(clave), keep="first")

def velocidad_implicita(df):
    """km/h necesarios para ir de cada punto al siguiente en el tiempo que marca el equipo."""
    d = haversine(df.lat.values[:-1], df.lon.values[:-1], df.lat.values[1:], df.lon.values[1:])
    dt = np.diff(df.t_equipo.values) / 3600
    return d / np.maximum(dt, 1e-9)

def marcar_saltos(df, vmax=160.0):
    """Un salto aislado obliga a ir rapidísimo para llegar a él Y para volver. Se marca el punto
    si la velocidad desde el anterior y hacia el siguiente superan vmax; se repite hasta que no quede ninguno."""
    df = df.copy(); df["salto_detectado"] = False; idx = np.arange(len(df))
    while True:
        sub = df.iloc[idx]; v = velocidad_implicita(sub)
        malo = np.zeros(len(sub), bool); malo[1:-1] = (v[:-1] > vmax) & (v[1:] > vmax)
        if not malo.any(): break
        score = np.zeros(len(sub)); score[1:-1] = v[:-1] + v[1:]     # el peor: el que obliga a ir más rápido
        peor = int(np.argmax(np.where(malo, score, -1)))
        df.iloc[idx[peor], df.columns.get_loc("salto_detectado")] = True; idx = np.delete(idx, peor)
    return df

def tramos_congelados(t, valor, odo, min_min=30.0, min_km=5.0):
    """Tramos en los que 'valor' no cambia ni una décima durante al menos min_min minutos
    mientras el odómetro avanza al menos min_km (parado, es normal que no cambie)."""
    t, valor, odo = map(np.asarray, (t, valor, odo)); out = []; i = 0
    while i < len(valor):
        j = i
        while j + 1 < len(valor) and valor[j + 1] == valor[i]: j += 1
        dur = (t[j] - t[i]) / 60
        if dur >= min_min and odo[j] - odo[i] >= min_km: out.append((float(t[i]), float(t[j]), float(dur), float(odo[j] - odo[i])))
        i = j + 1
    return out

def theil_sen(x, y):
    """Pendiente robusta: mediana de las pendientes entre todas las parejas de puntos."""
    x, y = np.asarray(x, float), np.asarray(y, float); i, j = np.triu_indices(len(x), 1)
    ok = x[j] != x[i]; s = np.median((y[j] - y[i])[ok] / (x[j] - x[i])[ok])
    return s, np.median(y - s * x)

def estimar_deriva(t_equipo, t_servidor, ventana_s=1800):
    """El retraso de red solo puede sumar: el borde inferior de (servidor - equipo) sigue al error del reloj.
    Se toma el mínimo por ventana y se ajusta una recta robusta. Devuelve s de deriva por hora y el desfase."""
    t_equipo, d = np.asarray(t_equipo, float), np.asarray(t_servidor, float) - np.asarray(t_equipo, float)
    b = np.floor((t_equipo - t_equipo.min()) / ventana_s)
    xs, ys = [], []
    for k in np.unique(b):
        m = b == k; i = np.argmin(d[m]); xs.append(t_equipo[m][i]); ys.append(d[m][i])
    s, c = theil_sen(np.array(xs), np.array(ys))
    return -s * 3600 / (1 + s), c, (np.array(xs), np.array(ys))
