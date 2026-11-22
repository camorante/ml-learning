"""Cuántos parámetros tiene un bloque Transformer, comprobado con nn.TransformerEncoderLayer, y cuánto ocupa la
matriz de atención según la longitud de la secuencia (el coste cuadrático)."""
import torch
from torch import nn
from modelo import Viajero, n_param


def bloque(d, f):
    """atención: Wq, Wk, Wv, Wo (d×d) con sus sesgos · FFN d→f→d con sesgos · dos normas (γ y β)"""
    return 4 * (d * d + d), d * f + f + f * d + d, 2 * 2 * d


print("parámetros de un bloque (atención + FFN + normas) · fórmula y nn.TransformerEncoderLayer")
for nombre, d, h, f in [("este tema", 16, 2, 64), ("Transformer base (2017)", 512, 8, 2048), ("GPT-2 pequeño", 768, 12, 3072)]:
    a, ff, ln = bloque(d, f)
    capa = nn.TransformerEncoderLayer(d, h, dim_feedforward=f, batch_first=True)
    print(f"  {nombre:24s} d={d:3d} h={h:2d} FFN {f:4d}: atención {a:9,} + FFN {ff:9,} + normas {ln:5,} = {a + ff + ln:9,}"
          f" · PyTorch {n_param(capa):9,}")
m = Viajero()
partes = {"embedding 20×16": m.E, "bloque": [m.ln1, m.q, m.k, m.v, m.o, m.ln2, m.f1, m.f2], "norma final": m.lnf, "salida 16→20": m.sal}
print("el modelo del tema:", ", ".join(f"{k} {sum(n_param(x) for x in (v if isinstance(v, list) else [v]))}" for k, v in partes.items()),
      f"= {n_param(m)} (la posición es fija: no se aprende)")
a, ff, ln = bloque(768, 3072)
gpt2 = 12 * (a + ff + ln) + 50257 * 768 + 1024 * 768 + 2 * 768
print(f"GPT-2 pequeño: 12 bloques {12 * (a + ff + ln):,} + fichas 50 257×768 {50257 * 768:,} + posiciones 1 024×768"
      f" {1024 * 768:,} + norma final 1 536 = {gpt2:,}")
print("matriz de atención n×n de una cabeza en una capa (2 bytes por número):")
for n in (40, 1024, 8192, 131072):
    b = 2 * n * n
    print(f"  n = {n:7,}: {n * n:15,} pesos · {b / 2**20:12,.3f} MiB")
