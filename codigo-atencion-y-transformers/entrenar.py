"""Entrena el Transformer diminuto del tema (d = 16, 2 cabezas, un bloque) y dos variantes:
sin codificación posicional y con una sola cabeza. Mide los aciertos en las dos preguntas que se pueden acertar
y adónde mira la atención desde «regreso» y desde «taller». Guarda el principal en viajero.pt y el que no
sabe de posiciones en viajero_sin_posicion.pt."""
import numpy as np
import torch
from datos import datos, clave, VOCAB, CODIGOS, DEPOSITOS
from modelo import entrenar, n_param, aciertos

(X, L), (Xv, Lv), (Xt, Lt) = datos()
reg, tal = clave(Xt, Lt)


def mide(m):
    with torch.no_grad():
        Z, W = m(torch.tensor(Xt), pesos=True)
    pred, W = Z.argmax(-1).numpy(), W.numpy()                       # W: (viajes, cabezas, T, T)
    # peso que pone cada cabeza, desde la pregunta, en la ficha que esconde la respuesta
    a_dep = W[[i for i, _, _ in reg], :, [t for _, t, _ in reg], 1].mean(0)
    a_cod = W[[i for i, _, _ in tal], :, [t for _, t, _ in tal], [t - dd for _, t, dd in tal]].mean(0)
    return aciertos(pred, Xt, reg), aciertos(pred, Xt, tal), a_dep, a_cod


resultados = {}
for nombre, kw in [("principal: 2 cabezas, con posición", {}), ("sin codificación posicional", {"con_posicion": False}),
                   ("una sola cabeza", {"h": 1})]:
    m, curva, seg = entrenar(X, L, Xv, Lv, **kw)
    ad, ac, wd, wc = mide(m)
    print(f"{nombre} · {n_param(m)} parámetros")
    print(f"  pérdida de validación por época 1, 10, 25, 50, 100: {', '.join(f'{curva[e - 1]:.3f}' for e in (1, 10, 25, 50, 100))}")
    print(f"  aciertos en 1 000 viajes de prueba: depósito tras «regreso» {ad:.1%} · último código tras «taller» {ac:.1%}")
    print(f"  peso medio de cada cabeza en la ficha clave: depósito {np.round(wd, 3).tolist()} · código {np.round(wc, 3).tolist()}")
    if len(resultados) < 2:
        torch.save(m.state_dict(), "viajero.pt" if not resultados else "viajero_sin_posicion.pt")
    resultados[nombre] = (ad, ac)
