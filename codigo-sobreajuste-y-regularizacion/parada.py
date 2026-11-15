"""Parada temprana con paciencia sobre el juguete sin regularizar: se vigila la validación en cada época, se guarda la
mejor y se para cuando lleva `paciencia` épocas sin mejorar. Fotos de la función aprendida a lo largo del entrenamiento."""
import json, os
import numpy as np
from datos import V_TR, Y_TR, V_VA, Y_VA, ex, ey, SY, MY, REJILLA
from mlp import entrenar, adelante

FOTOS = list(range(0, 600, 20)) + list(range(600, 3001, 100))


def parada_temprana(va, paciencia):
    """Recorre la curva de validación como si llegara época a época. Devuelve (época en la que para, mejor época)."""
    mejor = 0
    for ep in range(1, len(va)):
        if va[ep] < va[mejor]:
            mejor = ep
        elif ep - mejor >= paciencia:
            return ep, mejor
    return len(va) - 1, mejor


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    G = ex(REJILLA)
    r = entrenar(ex(V_TR), ey(Y_TR), ex(V_VA), ey(Y_VA), epocas=3000, lr=0.01, rejilla=G, fotos=FOTOS)
    rm = lambda m: float(np.sqrt(m) * SY)
    pacs = {}
    for pac in (10, 50, 100, 300):
        ep, mj = parada_temprana(r["va"], pac)
        pacs[pac] = [ep, mj]
        print(f"paciencia {pac:3d}: para en la época {ep:4d} y se queda con la {mj:3d} (validación {rm(r['va'][mj]):.3f}, entrenamiento {rm(r['tr'][mj]):.3f})")
    print(f"sin parar: época 3000, validación {rm(r['va'][-1]):.3f}, entrenamiento {rm(r['tr'][-1]):.3f}")
    json.dump(dict(fotos=FOTOS, pac=pacs, tr=[round(rm(v), 3) for v in r["tr"]], va=[round(rm(v), 3) for v in r["va"]], f=[np.round(r["fotos"][e] * SY + MY, 2).tolist() for e in FOTOS]),
              open("parada.json", "w", encoding="utf-8"), separators=(",", ":"))
