"""La tasa no tiene por qué ser fija. SGD con momento 0,9, lotes de 32, 60 épocas: tasa constante, coseno,
calentamiento + coseno y por escalones. Y una tasa tan grande que lo hace estallar."""
import json, math, os
import numpy as np
from datos import X_TR, Y_TR, X_VA, Y_VA, escalar
from entrenar import entrenar, epoca_meta

X, Xv = escalar(X_TR), escalar(X_VA)
coseno = lambda f: 0.5 * (1 + math.cos(math.pi * f))                     # de 1 a 0 en media onda
calent = lambda f: min(1.0, f / 0.05) * coseno(f)                        # 5 % de los pasos subiendo, luego coseno
escalones = lambda f: 0.1 ** int(f / (1 / 3))                            # ÷10 a 1/3 y a 2/3 del camino

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    casos = [("constante 0.1", 0.1, None), ("constante 1", 1.0, None), ("constante 10", 10.0, None),
             ("coseno desde 1", 1.0, coseno), ("calentamiento + coseno desde 1", 1.0, calent),
             ("escalones desde 1", 1.0, escalones), ("calentamiento + coseno desde 3", 3.0, calent)]
    out = {}
    for nombre, lr, prog in casos:
        r = entrenar("momento", lr, 32, X, Y_TR, Xv, Y_VA, epocas=60, programa=prog)
        va = r["va"]; ok = len(va) == 61
        em = epoca_meta(va, 0.25)
        print(f"{nombre:32s} " + (f"validación final {va[-1]:.3f}  mínima {min(va):.3f}  acierto {r['acc']:.1%}"
                                  f"  baja de 0.25 en la época {em}" if ok else
                                  f"se dispara en la época {len(va) - 1}: pérdida {r['tr'][-1]:.3g}"))
        out[nombre] = dict(tr=[round(v, 4) for v in r["tr"]], va=[round(v, 4) for v in va])
    f = np.linspace(0, 1, 7)
    print("factor de la tasa en f = 0, 1/6, …, 1:  coseno", np.round([coseno(x) for x in f], 3),
          " calentamiento", np.round([calent(x) for x in f], 3))
    json.dump(out, open("programas.json", "w", encoding="utf-8"), separators=(",", ":"))
