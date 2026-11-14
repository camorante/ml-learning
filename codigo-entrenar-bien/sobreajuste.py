"""Una curva de validación que sube: una red grande (2-64-1) con solo 40 lecturas de entrenamiento.
El sobreajuste y cómo frenarlo son el tema 7; aquí solo aprendemos a reconocerlo en las curvas."""
import json, os
import numpy as np
from datos import X_TR, Y_TR, X_VA, Y_VA, escalar
from entrenar import entrenar

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    X, y = escalar(X_TR[:40]), Y_TR[:40]                     # 40 lecturas (el escalado sigue siendo el de las 600)
    r = entrenar("adam", 0.01, 40, X, y, escalar(X_VA), Y_VA, epocas=1500, m=64)
    va = np.array(r["va"]); e = int(va.argmin())
    print(f"2-64-1 ({64 * 4 + 1} parámetros) con 40 lecturas, Adam 0.01, lote completo, 1500 épocas")
    for k in sorted({0, e, 200, 500, 1000, 1500}):
        print(f"   época {k:4d}: entrenamiento {r['tr'][k]:.3f}   validación {va[k]:.3f}")
    print(f"la validación es mínima en la época {e} ({va[e]:.3f}); al final, {va[-1]:.3f} con {r['tr'][-1]:.4f} en entrenamiento")
    json.dump(dict(tr=[round(v, 4) for v in r["tr"]], va=[round(v, 4) for v in r["va"]], mejor=e),
              open("sobreajuste.json", "w", encoding="utf-8"), separators=(",", ":"))
