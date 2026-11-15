"""Weight decay y el tamaño de los pesos: los 4 096 pesos de la capa oculta del medio (64 × 64) al final del juguete
sin dropout, para cada λ de AdamW. Histograma de |w| y la suma de los cuadrados de todos los pesos."""
import json, os
import numpy as np
from datos import V_TR, Y_TR, V_VA, Y_VA, ex, ey
from mlp import entrenar, iniciar

BORDES = [0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1, 1.5, 100]

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    P0 = iniciar([1, 64, 64, 1], 0)
    res = {"ini": dict(h=np.histogram(np.abs(P0[2]), BORDES)[0].tolist(), s2=round(float(sum((p ** 2).sum() for p in P0[::2])), 1),
                       mx=round(float(np.abs(P0[2]).max()), 3))}
    print(f"al empezar: suma de w² = {res['ini']['s2']}, |w| máximo en la capa del medio = {res['ini']['mx']}")
    for wd in (0, 0.1, 0.3, 1, 3):
        r = entrenar(ex(V_TR), ey(Y_TR), ex(V_VA), ey(Y_VA), epocas=3000, lr=0.01, wd=wd)
        W = r["P"][2]
        s2 = float(sum((p ** 2).sum() for p in r["P"][::2]))
        res[f"{wd:g}"] = dict(h=np.histogram(np.abs(W), BORDES)[0].tolist(), s2=round(s2, 1), mx=round(float(np.abs(W).max()), 3))
        print(f"λ = {wd:<3g}: suma de w² (todas las capas) = {s2:8.1f}   |w| máximo en la capa del medio = {np.abs(W).max():.3f}"
              f"   pesos con |w| < 0.05: {int((np.abs(W) < 0.05).sum())} de 4096")
    json.dump(dict(bordes=BORDES, **res), open("pesos.json", "w", encoding="utf-8"), separators=(",", ":"))
