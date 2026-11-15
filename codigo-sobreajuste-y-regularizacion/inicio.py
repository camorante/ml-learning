"""La señal a través de 20 capas de 256 neuronas según cómo se inicializan los pesos (sesgos a 0), con 1 000 entradas
N(0, 1). Xavier lleva la ganancia que recomienda PyTorch (5/3 con tanh, 1 con ReLU). Se mide la desviación típica de la salida de cada capa. Con bn=True, cada capa normaliza su suma con las
estadísticas del lote (normalización por lotes, γ = 1, β = 0) antes de la activación."""
import json, os
import numpy as np
from mlp import xavier_uniform, he_normal
from bn import BatchNorm1d

ACT = {"tanh": np.tanh, "relu": lambda a: np.maximum(a, 0)}
INI = {"pequenos": lambda o, i, r, g: r.normal(0, 0.01, (o, i)), "xavier": lambda o, i, r, g: xavier_uniform(o, i, r, g),
       "he": lambda o, i, r, g: he_normal(o, i, r), "grandes": lambda o, i, r, g: r.normal(0, 1, (o, i))}


def senal(act, ini, bn=False, capas=20, ancho=256, seed=0):
    rng = np.random.default_rng(seed)
    h, out = rng.standard_normal((1000, ancho)), []
    for _ in range(capas):
        a = h @ INI[ini](ancho, ancho, rng, 5 / 3 if act == "tanh" else 1.0).T   # Xavier: ganancia 5/3 con tanh
        if bn:
            a = BatchNorm1d(ancho).forward(a)
        h = ACT[act](a)
        out.append(float(h.std()))
    return out


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    res = {}
    print("desviación típica de la salida de las capas 1, 5, 10 y 20")
    for act in ACT:
        for ini in INI:
            for bn in (False, True):
                s = senal(act, ini, bn)
                res[f"{act}|{ini}|{int(bn)}"] = [float(f"{v:.3g}") for v in s]
                print(f"  {act:4s} {ini:8s} {'con BN' if bn else 'sin BN'}: " + "  ".join(f"{s[k - 1]:9.3g}" for k in (1, 5, 10, 20)))
    json.dump(res, open("inicio.json", "w", encoding="utf-8"), separators=(",", ":"))
