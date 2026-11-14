"""Escalar las entradas: la misma red y los mismos pasos con las lecturas en bruto (rpm y °C) o estandarizadas.
Primero la curvatura de una sola neurona en su punto de partida; después, entrenamientos reales."""
import json, os
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from datos import X_TR, Y_TR, X_VA, Y_VA, escalar
from entrenar import entrenar

TASAS = [0.0001, 0.001, 0.01, 0.1, 1.0]


def kappa(X):
    """Hessiana de la entropía cruzada media de una neurona sigmoide en w = 0: ¼·E[x̃ x̃ᵀ], con x̃ = (1, x).
    Su número de condición (curvatura máxima / mínima) dice cuánto se alarga el valle."""
    Xt = np.c_[np.ones(len(X)), X]
    ev = np.linalg.eigvalsh(Xt.T @ Xt / len(X) / 4)
    return ev.max() / ev.min(), ev


def uno(args):
    opt, lr, bruto = args
    f = (lambda A: np.asarray(A, float)) if bruto else escalar
    r = entrenar(opt, lr, 32, f(X_TR), Y_TR, f(X_VA), Y_VA, epocas=60)
    return dict(opt=opt, lr=lr, bruto=bruto, tr=[round(v, 4) for v in r["tr"]], va=[round(v, 4) for v in r["va"]], acc=r["acc"])


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    print(f"lecturas: rpm de media {X_TR[:, 0].mean():.0f} y desviación {X_TR[:, 0].std():.0f};"
          f" temperatura de media {X_TR[:, 1].mean():.1f} °C y desviación {X_TR[:, 1].std():.2f} °C")
    for nombre, X in [("en bruto", X_TR), ("estandarizadas", escalar(X_TR))]:
        k, ev = kappa(X)
        print(f"una neurona, {nombre:15s}: curvaturas {np.array2string(ev, precision=3)}  →  número de condición {k:,.0f}")
    trabajos = [(o, lr, b) for b in (True, False) for o in ("sgd", "adam") for lr in TASAS]
    with ProcessPoolExecutor() as ex:
        R = list(ex.map(uno, trabajos))
    print("red 2-8-1, lotes de 32, 60 épocas · pérdida de validación al final · tasa:" + "".join(f"{lr:>8g}" for lr in TASAS))
    for b in (True, False):
        for o in ("sgd", "adam"):
            fila = [r for r in R if r["bruto"] == b and r["opt"] == o]
            print(f"   {o:5s} {'en bruto' if b else 'estandarizadas':15s}" + " " * 36
                  + "".join(f"{r['va'][-1]:8.3f}" if len(r["va"]) == 61 else "       ∞" for r in fila))
    best = {f"{o}_{'b' if b else 'e'}": min((r for r in R if r["bruto"] == b and r["opt"] == o), key=lambda r: r["va"][-1])
            for b in (True, False) for o in ("sgd", "adam")}
    for k, r in best.items():
        print(f"mejor {k}: tasa {r['lr']:g}, validación {r['va'][-1]:.3f}, acierto {r['acc']:.1%}")
    json.dump(best, open("escala.json", "w", encoding="utf-8"), separators=(",", ":"))
