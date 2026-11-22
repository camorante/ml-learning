"""Datos del tema 11: el diario de un viaje, como una secuencia de fichas (tokens). La caja negra del camión
apunta lo que pasa: sale de un depósito, carga, descarga, frena, para en un atasco… A veces se enciende un
código de avería (los del tema 10). Al final pasa por el taller, donde el mecánico apunta el ÚLTIMO código que
se encendió, y vuelve al depósito del que salió.

La tarea es la de un modelo de lenguaje: predecir la ficha siguiente. Casi todas son imprevisibles (los sucesos
del viaje son al azar), pero dos se pueden acertar siempre si se mira bien hacia atrás:
  · después de «regreso» va el depósito de la salida, que está al principio, lejos;
  · después de «taller» va el último código de avería encendido, que puede no ser el único."""
import numpy as np

ESPECIALES = ["salida", "regreso", "taller", "fin"]
DEPOSITOS = ["Madrid", "Zaragoza", "Valencia", "Sevilla"]
CODIGOS = ["P0217", "P0299", "C0035", "P0562"]          # sobretemperatura, turbo, sensor de rueda, tensión baja
SUCESOS = ["carga", "descarga", "ralentí", "frenada", "atasco", "repostaje", "peaje", "parada"]
VOCAB = ESPECIALES + DEPOSITOS + CODIGOS + SUCESOS
V = len(VOCAB)                                         # 20 fichas
ID = {t: i for i, t in enumerate(VOCAB)}
TMAX = 40                                              # el viaje más largo posible


def viaje(rng):
    """salida D · 6–26 sucesos con 1–3 códigos distintos entre ellos · taller C · 1–6 sucesos · regreso D · fin"""
    d = DEPOSITOS[rng.integers(4)]
    n1 = rng.integers(6, 27)
    medio = [SUCESOS[k] for k in rng.integers(0, 8, n1)]
    nc = rng.integers(1, 4)
    cods = [CODIGOS[k] for k in rng.choice(4, nc, replace=False)]
    for c, p in zip(cods, np.sort(rng.choice(n1, nc, replace=False))):
        medio[p] = c                                    # los códigos sustituyen sucesos, en orden de aparición
    n2 = rng.integers(1, 7)
    cola = [SUCESOS[k] for k in rng.integers(0, 8, n2)]
    return ["salida", d] + medio + ["taller", cods[-1]] + cola + ["regreso", d, "fin"]


def conjunto(n, seed):
    """n viajes como matriz de índices (n, TMAX), rellenada con «fin», y sus longitudes."""
    rng = np.random.default_rng(seed)
    vs = [viaje(rng) for _ in range(n)]
    X = np.full((n, TMAX), ID["fin"], dtype=np.int64)
    L = np.array([len(v) for v in vs])
    for i, v in enumerate(vs):
        X[i, :len(v)] = [ID[t] for t in v]
    return X, L


def datos():
    """Entrenamiento (2 000 viajes), validación (500) y prueba (1 000), con semillas fijas."""
    return conjunto(2000, 11), conjunto(500, 111), conjunto(1000, 1111)


def clave(X, L):
    """Las posiciones que se pueden acertar: la ficha que sigue a «regreso» y la que sigue a «taller».
    Devuelve dos listas de (viaje, posición de la pregunta) y la distancia hasta la respuesta escondida."""
    reg, tal = [], []
    for i in range(len(X)):
        v = X[i, :L[i]].tolist()
        reg.append((i, v.index(ID["regreso"]), v.index(ID["regreso"]) - 1))      # distancia hasta el depósito (pos. 1)
        t = v.index(ID["taller"]); ult = max(p for p in range(t) if VOCAB[v[p]] in CODIGOS)
        tal.append((i, t, t - ult))
    return reg, tal


if __name__ == "__main__":
    (X, L), (Xv, Lv), (Xt, Lt) = datos()
    print(f"vocabulario: {V} fichas · viajes de {L.min()} a {L.max()} fichas (media {L.mean():.1f})")
    print("ejemplo:", " ".join(VOCAB[k] for k in X[0, :L[0]]))
    reg, tal = clave(Xt, Lt)
    dr = np.array([r[2] for r in reg]); dt = np.array([t[2] for t in tal])
    nc = np.array([sum(VOCAB[k] in CODIGOS for k in Xt[i, :Lt[i]]) - 1 for i in range(len(Xt))])
    print(f"prueba: depósito a {dr.min()}–{dr.max()} fichas de «regreso» (media {dr.mean():.1f});"
          f" último código a {dt.min()}–{dt.max()} de «taller» (media {dt.mean():.1f})")
    print(f"viajes con 1, 2 y 3 códigos: {[int((nc == k).sum()) for k in (1, 2, 3)]}")
    print(f"acierto adivinando al azar: depósito {1/4:.0%}, código {1/4:.0%};"
          f" eligiendo un código encendido al azar: {np.mean(1 / nc):.1%}")
