"""Entrenamiento real: RNN, LSTM y GRU de 16 unidades con la pregunta de memoria, para cada hueco y tres semillas
(en paralelo, un proceso por red). Se guardan las redes de la semilla 0 y todos los aciertos en comparar.json.
También las dos redes mínimas de una sola unidad (hueco de 25 lecturas) que usan el juguete y el ejemplo."""
import json, os, time
from multiprocessing import Pool
import torch
from datos import HUECOS, datos
from modelos import entrenar, acierto, n_param

os.chdir(os.path.dirname(os.path.abspath(__file__)))


def una(args):
    tipo, H, G, seed, kw = args
    kw = dict(kw); extra = kw.pop("extra", "")
    torch.set_num_threads(1)
    (X, Y, _), (Xv, Yv, _), (Xt, Yt, _) = datos(G)
    m, curva = entrenar(tipo, H, X, Y, Xv, Yv, seed=seed, **kw)
    if seed == 0 and not extra:
        torch.save(m.state_dict(), f"red_{tipo}{H}_G{G}.pt")
    return dict(tipo=tipo, H=H, G=G, seed=seed, param=n_param(m), prueba=acierto(m, Xt, Yt), curva=curva, extra=extra)


if __name__ == "__main__":
    t0 = time.time()
    trabajos = [(t, 16, G, s, {}) for G in sorted(HUECOS, reverse=True) for t in ("LSTM", "GRU", "RNN") for s in (0, 1, 2)]
    trabajos += [(t, 1, 25, 0, dict(lr=0.01, sesgo_olvido=5.0)) for t in ("LSTM", "RNN")]
    # dos controles: una RNN con tantos parámetros como la LSTM, y la LSTM con el sesgo de olvido de PyTorch (0)
    trabajos += [("RNN", 32, 50, s, dict(extra="RNN de 32 unidades")) for s in (0, 1, 2)]
    trabajos += [("LSTM", 16, 100, s, dict(sesgo_olvido=None, extra="LSTM con sesgo de olvido 0")) for s in (0, 1, 2)]
    with Pool(15) as p:
        R = p.map(una, trabajos, chunksize=1)
    json.dump(R, open("comparar.json", "w", encoding="utf-8"))
    print(f"aciertos en 1 000 tramos de prueba (mitad con frenada brusca al salir) · {time.time() - t0:.0f} s")
    print("semilla 0 (mínimo–máximo de las semillas 0, 1 y 2)")
    print(f"{'hueco':18s}" + "".join(f"{t:24s}" for t in ("RNN", "LSTM", "GRU")))
    for G in HUECOS:
        fila = f"{G:3d} ({G / 10:4.1f} s)".ljust(18)
        for t in ("RNN", "LSTM", "GRU"):
            a = [r["prueba"] for r in R if r["tipo"] == t and r["H"] == 16 and r["G"] == G and not r["extra"]]
            fila += f"{a[0]:.1%} ({min(a):.1%}–{max(a):.1%})".ljust(24)
        print(fila)
    print("parámetros:", {t: next(r["param"] for r in R if r["tipo"] == t and r["H"] == 16) for t in ("RNN", "LSTM", "GRU")})
    for e in ("RNN de 32 unidades", "LSTM con sesgo de olvido 0"):
        a = [r for r in R if r["extra"] == e]
        print(f"{e} ({a[0]['param']} parámetros), hueco {a[0]['G']}: " + ", ".join(f"{r['prueba']:.1%}" for r in a) + " (semillas 0, 1, 2)")
    for r in R:
        if r["H"] == 1:
            print(f"{r['tipo']} de una sola unidad ({r['param']} parámetros), hueco 25: {r['prueba']:.1%}")
