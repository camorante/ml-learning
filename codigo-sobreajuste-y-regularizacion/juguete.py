"""El juguete: la red 1-64-64-1 con 20 viajes, entrenada 3 000 épocas con AdamW (tasa 0.01, lote completo) para cada
combinación de weight decay λ y dropout p. Siempre los mismos pesos iniciales (semilla 0) y las mismas máscaras (semilla 1).
Guarda las curvas, la mejor época de validación y la función aprendida (al final y en la mejor época)."""
import json, os
import numpy as np
from datos import V_TR, Y_TR, V_VA, Y_VA, ex, ey, SY, MY, REJILLA
from mlp import entrenar, adelante

WDS, PS, EPOCAS = [0, 0.1, 0.3, 1, 3], [0, 0.1, 0.2, 0.3, 0.5], 3000
rm = lambda mse: float(np.sqrt(mse) * SY)                       # error cuadrático medio → RMSE en L/100 km

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    X, y, Xv, yv, G = ex(V_TR), ey(Y_TR), ex(V_VA), ey(Y_VA), ex(REJILLA)
    out = {}
    print(f"red 1-64-64-1 ({64 * 2 + 64 * 65 + 65} parámetros), 20 viajes de entrenamiento, 30 de validación")
    print("  λ     p   | mejor época  valid. ahí | valid. final  entren. final")
    for wd in WDS:
        for p in PS:
            r = entrenar(X, y, Xv, yv, epocas=EPOCAS, lr=0.01, wd=wd, p=p, rejilla=G)
            f = lambda P: [round(float(v) * SY + MY, 2) for v in adelante(P, G)[0]]
            e = r["mejor"]
            out[f"{wd:g}|{p:g}"] = dict(tr=[round(rm(v), 3) for v in r["tr"][::10]], va=[round(rm(v), 3) for v in r["va"][::10]],
                                        mejor=e, vm=round(rm(r["va"][e]), 3), tm=round(rm(r["tr"][e]), 3),
                                        vf=round(rm(r["va"][-1]), 3), tf=round(rm(r["tr"][-1]), 3), fm=f(r["Pm"]), ff=f(r["P"]))
            o = out[f"{wd:g}|{p:g}"]
            print(f"  {wd:<4g}  {p:<3g} |   {e:5d}      {o['vm']:.3f}   |   {o['vf']:.3f}        {o['tf']:.3f}")
    json.dump(out, open("juguete.json", "w", encoding="utf-8"), separators=(",", ":"))
