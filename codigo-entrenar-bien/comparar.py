"""Entrenamientos reales de la red 2-8-1 (tanh) con la zona normal del motor: optimizador × tasa × tamaño del
lote, 60 épocas, siempre desde los mismos pesos iniciales y con el mismo barajado. Guarda rejilla.json."""
import json, os
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from datos import X_TR, Y_TR, X_VA, Y_VA, escalar
from entrenar import entrenar, epoca_meta

OPTS = ["sgd", "momento", "adam"]
TASAS = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0]
LOTES = [1, 32, 600]
META = 0.25                                  # pérdida de validación que consideramos «ya ha aprendido la zona»


def uno(args):
    opt, lr, lote = args
    r = entrenar(opt, lr, lote, escalar(X_TR), Y_TR, escalar(X_VA), Y_VA, epocas=60)
    return dict(opt=opt, lr=lr, lote=lote, tr=[round(v, 4) for v in r["tr"]], va=[round(v, 4) for v in r["va"]],
                acc=r["acc"], pasos=r["pasos"], P=[np.round(p, 4).tolist() for p in r["P"]])


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    trabajos = [(o, lr, b) for o in OPTS + ["rmsprop"] for b in LOTES for lr in TASAS if o != "rmsprop" or b == 32]
    with ProcessPoolExecutor() as ex:
        R = list(ex.map(uno, trabajos))
    json.dump(R, open("rejilla.json", "w", encoding="utf-8"), separators=(",", ":"))
    fin = lambda r: f"{r['va'][-1]:6.3f}" if len(r["va"]) == 61 else "∞"           # ∞: se disparó y se paró
    print(f"{len(R)} entrenamientos de 60 épocas con 600 lecturas; pérdida de validación al final (60 épocas)")
    for b in LOTES:
        print(f"lotes de {b} ({-(-600 // b)} pasos por época)   tasa:" + "".join(f"{lr:>7g}" for lr in TASAS))
        for o in OPTS + (["rmsprop"] if b == 32 else []):
            fila = [r for r in R if r["opt"] == o and r["lote"] == b]
            print(f"   {o:8s}" + " " * 25 + "".join(f"{fin(r):>7s}" for r in fila))
    print(f"\ncon lotes de 32 y la tasa que deja la menor pérdida de validación al final:")
    for o in OPTS + ["rmsprop"]:
        r = min((r for r in R if r["opt"] == o and r["lote"] == 32), key=lambda r: r["va"][-1])
        em = epoca_meta(r["va"], META)
        print(f"   {o:8s} tasa {r['lr']:<5g} validación {r['va'][-1]:.3f}   acierto {r['acc']:.1%}"
              f"   baja de {META} en la época {em:2d} ({em * 19:3d} pasos)")
