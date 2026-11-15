"""Dropout por dentro: la red del juguete entrenada con p = 0.3 (sin weight decay). Cada máscara apaga cada neurona oculta
con probabilidad 0.3 para TODAS las velocidades a la vez: es una de las 2¹²⁸ redes «adelgazadas». En evaluación no se
apaga ninguna (las encendidas ya se multiplicaron por 1/0.7 al entrenar). ¿Se parece la red completa a la media?"""
import json, os
import numpy as np
from datos import V_TR, Y_TR, V_VA, Y_VA, ex, ey, SY, MY, REJILLA
from mlp import entrenar, adelante

P_DROP = 0.3


def adelgazada(P, X, m1, m2, p=P_DROP):
    """La red con las máscaras m1 y m2 (64 ceros y unos cada una) fijas, en la escala del dropout invertido."""
    h1 = np.tanh(X @ P[0].T + P[1]) * m1 / (1 - p)
    h2 = np.tanh(h1 @ P[2].T + P[3]) * m2 / (1 - p)
    return (h2 @ P[4].T + P[5])[:, 0]


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    G = ex(REJILLA)
    r = entrenar(ex(V_TR), ey(Y_TR), ex(V_VA), ey(Y_VA), epocas=3000, lr=0.01, p=P_DROP)
    P, rng = r["P"], np.random.default_rng(5)
    M = (rng.random((2000, 2, 64)) >= P_DROP).astype(float)
    F = np.array([adelgazada(P, G, m[0], m[1]) for m in M]) * SY + MY
    fe = adelante(P, G)[0] * SY + MY
    print(f"neuronas encendidas en las 30 primeras máscaras: entre {int(M[:30].sum((1, 2)).min())} y {int(M[:30].sum((1, 2)).max())} de 128"
          f" (de media se esperan {128 * (1 - P_DROP):.1f})")
    d30, d2000 = np.abs(F[:30].mean(0) - fe), np.abs(F.mean(0) - fe)
    print(f"red completa (evaluación) frente a la media de las redes adelgazadas: diferencia máxima {d30.max():.3f} L/100 km"
          f" con 30 y {d2000.max():.3f} con 2 000")
    print(f"dispersión de las adelgazadas (desviación típica media en la rejilla): {F.std(0).mean():.2f} L/100 km")
    k = int(np.argmin(np.abs(REJILLA - 85)))
    print(f"a {REJILLA[k]:.2f} km/h: red completa {fe[k]:.2f}; las 30 adelgazadas, de {F[:30, k].min():.2f} a {F[:30, k].max():.2f}")
    json.dump(dict(f=np.round(F[:30], 2).tolist(), fe=np.round(fe, 2).tolist(), m=M[:30].astype(int).tolist()),
              open("apagadas.json", "w", encoding="utf-8"), separators=(",", ":"))
