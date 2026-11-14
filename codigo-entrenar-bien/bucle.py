"""El bucle paso a paso con el entrenamiento del juguete (Adam, tasa 0.1, lotes de 32): lo que pasa en cada
uno de los primeros 38 pasos (dos épocas) y al final de cada época. Guarda bucle.json para los dibujos."""
import json, os
import numpy as np
import red
from datos import X_TR, Y_TR, X_VA, Y_VA, escalar
from entrenar import entrenar

X, Xv = escalar(X_TR), escalar(X_VA)
fotos, lotes, Ll, Lt, grads = [], [], [], [], []


def mirar(ep, k, j, L, P):
    if ep <= 2:
        fotos.append([np.round(p, 4).tolist() for p in P]); lotes.append(j.tolist())
        Ll.append(round(L, 4)); Lt.append(round(red.perdida(P, X, Y_TR), 4))


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    P0 = red.iniciar(8, seed=0)
    r = entrenar("adam", 0.1, 32, X, Y_TR, Xv, Y_VA, epocas=60, cada_paso=mirar)
    print(f"600 lecturas, lotes de 32 → {len(lotes) // 2} pasos por época (el último lote tiene {len(lotes[18])} lecturas)")
    print(f"antes de empezar: pérdida {r['tr'][0]:.4f} (ln 2 = {np.log(2):.4f} sería decir 50 % a todo)")
    for s in (0, 1, 2, 18, 37):
        print(f"   paso {s + 1:2d}: pérdida del lote {Ll[s]:.4f}   de las 600 tras el paso {Lt[s]:.4f}")
    for e in (1, 2, 5, 10, 20, 60):
        print(f"   época {e:2d}: entrenamiento {r['tr'][e]:.4f}   validación {r['va'][e]:.4f}")
    print(f"acierto en validación al final: {r['acc']:.1%} · pasos totales {r['pasos']}")
    # el ejemplo resuelto: los tres primeros gradientes del peso que va de la neurona 1 a la salida
    _, G1 = red.gradiente(P0, X[lotes[0]], Y_TR[lotes[0]])
    print(f"peso w = W2[0, 0] al empezar: {P0[2][0, 0]:.6f}; gradiente del primer lote {G1[2][0, 0]:.6f}")
    json.dump(dict(fotos=fotos, lotes=lotes, Ll=Ll, Lt=Lt, P0=[np.round(p, 4).tolist() for p in P0]),
              open("bucle.json", "w", encoding="utf-8"), separators=(",", ":"))
