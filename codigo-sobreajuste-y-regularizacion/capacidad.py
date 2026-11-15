"""Capacidad frente a datos: redes 1-m-m-1 sin regularizar (3 000 épocas, Adam 0.01, lote completo) con los primeros
n viajes medidos. El escalado se recalcula con los n viajes de cada caso. Error en L/100 km al terminar."""
import json, os
import numpy as np
from datos import V_POOL, Y_POOL, V_VA, Y_VA, REJILLA
from mlp import entrenar, adelante

MS, NS = [2, 8, 64], [20, 50, 100, 300]

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    res = {}
    print("  m   parámetros |  n viajes | entren.  valid.")
    for m in MS:
        for n in NS:
            v, y = V_POOL[:n], Y_POOL[:n]
            mv, sv, my, sy = v.mean(), v.std(), y.mean(), y.std()
            e = lambda u: ((u - mv) / sv)[:, None]
            r = entrenar(e(v), (y - my) / sy, e(V_VA), (Y_VA - my) / sy, capas=(1, m, m, 1), epocas=3000, lr=0.01)
            tf, vf = float(np.sqrt(r["tr"][-1]) * sy), float(np.sqrt(r["va"][-1]) * sy)
            res[f"{m}|{n}"] = dict(tf=round(tf, 3), vf=round(vf, 3), f=np.round(adelante(r["P"], e(REJILLA))[0] * sy + my, 2).tolist())
            print(f" {m:3d}   {m * 2 + m * (m + 1) + m + 1:6d}     |   {n:4d}    | {tf:6.3f}  {vf:6.3f}")
    json.dump(res, open("capacidad.json", "w", encoding="utf-8"), separators=(",", ":"))
