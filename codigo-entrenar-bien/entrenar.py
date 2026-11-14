"""El bucle de entrenamiento: barajar → para cada lote: adelante, pérdida, atrás, actualizar → repetir cada época."""
import numpy as np
import red
from optim import SGD, Adam, RMSprop

OPT = {"sgd": lambda lr: SGD(lr), "momento": lambda lr: SGD(lr, momentum=0.9), "adam": lambda lr: Adam(lr),
       "rmsprop": lambda lr: RMSprop(lr)}


def entrenar(opt, lr, lote, X, y, Xv, yv, epocas=60, m=8, programa=None, seed_init=0, seed_barajar=1, cada_paso=None):
    """Devuelve las pérdidas de entrenamiento y de validación al final de cada época (la 0 es antes de empezar),
    el acierto en validación y los parámetros finales. programa(f) multiplica la tasa, con f = fracción del
    entrenamiento ya hecha. cada_paso(época, k, índices, pérdida_del_lote, P) se llama tras cada actualización."""
    P = red.iniciar(m, X.shape[1], seed=seed_init)
    o = OPT[opt](lr)
    rng = np.random.default_rng(seed_barajar)
    n = len(y); pasos_epoca = -(-n // lote); total = epocas * pasos_epoca; t = 0
    tr, va = [red.perdida(P, X, y)], [red.perdida(P, Xv, yv)]
    for ep in range(1, epocas + 1):
        orden = rng.permutation(n)                          # barajar en cada época
        for k in range(pasos_epoca):
            j = orden[k * lote:(k + 1) * lote]              # el lote: las siguientes `lote` lecturas
            L, G = red.gradiente(P, X[j], y[j])             # adelante + pérdida + atrás
            if programa:
                o.lr = lr * programa(t / total)
            o.step(P, G)                                    # actualizar
            t += 1
            if cada_paso:
                cada_paso(ep, k, j, L, P)
        tr.append(red.perdida(P, X, y)); va.append(red.perdida(P, Xv, yv))
        if not np.isfinite(tr[-1]) or tr[-1] > 50:          # se ha disparado: no tiene sentido seguir
            break
    return dict(tr=tr, va=va, acc=red.acierto(P, Xv, yv), P=P, pasos=t)


def epoca_meta(va, meta):
    """Primera época en la que la pérdida de validación baja de la meta (None si nunca)."""
    return next((e for e, v in enumerate(va) if v < meta), None)
