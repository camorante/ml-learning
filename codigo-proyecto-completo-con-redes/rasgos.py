"""Preparación de las entradas: escalado con las medias del entrenamiento (D7) y rasgos hechos a mano para el modelo de
referencia (6 resúmenes por sensor: media, desviación, máximo, mínimo, pendiente y últimos 7 días menos los 21 de antes)."""
import numpy as np


class Escalador:
    """Resta la media y divide por la desviación, ambas calculadas SOLO con el entrenamiento."""
    def __init__(s, A, ejes):
        s.m, s.d = A.mean(ejes, keepdims=True), A.std(ejes, keepdims=True)

    def __call__(s, A):
        return (A - s.m) / s.d


def a_mano(X):
    """X: (n, 28, 3) → (n, 18) resúmenes por sensor."""
    t = np.arange(X.shape[1]) - (X.shape[1] - 1) / 2
    pend = (X * t[None, :, None]).sum(1) / (t ** 2).sum()                 # pendiente por día (mínimos cuadrados)
    R = [X.mean(1), X.std(1), X.max(1), X.min(1), pend, X[:, -7:].mean(1) - X[:, :-7].mean(1)]
    return np.concatenate(R, 1)


NOMBRES = [f"{r}_{c}" for r in ["media", "desv", "max", "min", "pend", "ult7"] for c in ["vib", "temp", "aceite"]]


def secuencia(X, esc):
    """(n, 28, 3) → (n, 6, 28): los 3 sensores escalados y, además, cada uno menos su media de los 21 primeros días
    (lo que se ha movido respecto a lo normal de ESE camión)."""
    s = esc(X).transpose(0, 2, 1)
    return np.concatenate([s, s - s[:, :, :21].mean(2, keepdims=True)], 1).astype(np.float32)


def preparar(F, esc=None):
    """Entradas de la red: secuencia (n, 6, 28) y tabla (n, 5), escaladas; modelo y depósito como enteros."""
    if esc is None:
        esc = dict(seq=Escalador(F["X"], (0, 1)), tab=Escalador(F["tab"], 0))
    return dict(seq=secuencia(F["X"], esc["seq"]), tab=esc["tab"](F["tab"]).astype(np.float32),
                mod=F["mod"].astype(np.int64), dep=F["dep"].astype(np.int64), y=F["y"]), esc
