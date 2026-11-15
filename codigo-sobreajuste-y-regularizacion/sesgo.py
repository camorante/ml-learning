"""Sesgo y varianza: la misma receta de entrenamiento con 20 muestras distintas de 20 viajes. Tres modelos: uno rígido
(sin capa oculta: una recta), la grande sin freno (1-64-64-1) y la grande con freno (AdamW λ = 1 y dropout 0.2).
Error cuadrático medio frente a la curva de verdad = sesgo² + varianza (medidos en la rejilla, en (L/100 km)²)."""
import json, os
import numpy as np
from datos import viajes, verdad, V_VA, Y_VA, REJILLA
from mlp import entrenar, adelante

MODELOS = {"rigida": dict(capas=(1, 1)), "grande": dict(capas=(1, 64, 64, 1)), "frenada": dict(capas=(1, 64, 64, 1), wd=1, p=0.2)}

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    f0, res = verdad(REJILLA), {}
    for k, cfg in MODELOS.items():
        F = []
        for s in range(20):
            v, y = viajes(20, seed=100 + s)                    # otra muestra de 20 viajes
            mv, sv, my, sy = v.mean(), v.std(), y.mean(), y.std()
            e = lambda u: ((u - mv) / sv)[:, None]
            r = entrenar(e(v), (y - my) / sy, e(V_VA), (Y_VA - my) / sy, epocas=3000, lr=0.01, **cfg)
            F.append(adelante(r["P"], e(REJILLA))[0] * sy + my)
        F = np.array(F)
        b2, var = float(np.mean((F.mean(0) - f0) ** 2)), float(np.mean(F.var(0)))
        res[k] = dict(f=np.round(F[:8], 2).tolist(), b2=round(b2, 3), var=round(var, 3))
        print(f"{k:8s} sesgo² {b2:6.3f}   varianza {var:6.3f}   suma {b2 + var:6.3f}   (media en la rejilla de 61 velocidades, 20 muestras)")
    json.dump(res, open("sesgo.json", "w", encoding="utf-8"), separators=(",", ":"))
