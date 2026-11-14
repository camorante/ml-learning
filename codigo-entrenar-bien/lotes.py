"""Tamaño del lote con una neurona de dos números (w, b): los 2 000 camiones del tema 4 (avería en 30 días
según los km desde la revisión). El gradiente de un lote es una estimación por muestreo del gradiente de verdad."""
import json, os
import numpy as np
from datos import camiones
from red import sigmoide, softplus

x, y = camiones(2000, seed=100)
u = (x - 30) / 30                                              # km centrados y escalados, como en el tema 4
N = len(y)


def perdida(w, b, j=slice(None)):
    z = b + w * u[j]
    return float(np.mean(y[j] * softplus(-z) + (1 - y[j]) * softplus(z)))


def grad(w, b, j=slice(None)):
    r = sigmoide(b + w * u[j]) - y[j]
    return np.array([np.mean(r * u[j]), np.mean(r)])


def camino(B, pasos=300, lr=2.0, inicio=(-6.0, -2.5), seed=2):
    """Descenso con lotes de B camiones, barajando en cada época. Devuelve los (w, b) visitados."""
    rng = np.random.default_rng(seed); th = np.array(inicio); P = [th.copy()]; orden, k = rng.permutation(N), 0
    for _ in range(pasos):
        if k + B > N:
            orden, k = rng.permutation(N), 0
        j = orden[k:k + B]; k += B
        th = th - lr * grad(*th, j); P.append(th.copy())
    return np.array(P)


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    th = np.array([-6.0, -2.5])                                # punto de partida (el del tema 4)
    for _ in range(20000):                                     # el fondo, con el gradiente exacto
        th -= 2.0 * grad(*th)
    print(f"{N} camiones, {y.sum()} averías · fondo del valle: w = {th[0]:.4f}, b = {th[1]:.4f}, pérdida {perdida(*th):.4f}")
    g0 = grad(-6.0, -2.5)
    per = np.array([[(sigmoide(-2.5 - 6.0 * ui) - yi) * ui, sigmoide(-2.5 - 6.0 * ui) - yi] for ui, yi in zip(u, y)])
    s2 = per.var(axis=0).sum()                                 # varianza de un ejemplo (traza)
    print(f"gradiente exacto en (−6, −2.5): ({g0[0]:.4f}, {g0[1]:.4f}), longitud {np.linalg.norm(g0):.4f}")
    rng = np.random.default_rng(7); salida = {"fondo": th.round(4).tolist(), "lotes": {}}
    print("  lote   pasos/época   ruido medido (2 000 lotes)   σ/√B·√((N−B)/(N−1))   pérdida tras 300 pasos (tasa 2)   distancia al fondo")
    for B in (1, 4, 16, 64, 256, 2000):
        est = np.array([grad(-6.0, -2.5, rng.choice(N, B, replace=False)) for _ in range(2000)])
        ruido = np.sqrt(((est - g0) ** 2).sum(1).mean())
        teoria = np.sqrt(s2 / B * (N - B) / (N - 1))
        P = camino(B)
        print(f"  {B:4d}   {N // B:6d}        {ruido:10.4f}                 {teoria:10.4f}            {perdida(*P[-1]):.4f}"
              f"               {np.linalg.norm(P[-1] - th):.3f}")
        abanico = est[:30]
        salida["lotes"][B] = dict(camino=P.round(4).tolist(), abanico=abanico.round(4).tolist(), ruido=round(ruido, 4))
    salida["g0"] = g0.round(4).tolist()
    W, Bb = np.linspace(-8, 8, 61), np.linspace(-6, 3, 46)
    salida["rejilla"] = dict(w=W.round(3).tolist(), b=Bb.round(3).tolist(),
                             L=[[round(perdida(w, b), 4) for w in W] for b in Bb])
    json.dump(salida, open("lotes.json", "w", encoding="utf-8"), separators=(",", ":"))
