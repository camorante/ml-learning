"""Verificación de activaciones.py: valores contra torch.nn.functional, derivadas contra el
autograd de PyTorch y contra diferencias centrales (float64)."""
import warnings
import numpy as np
import torch
import torch.nn.functional as F
from activaciones import ACTIVACIONES, escalon, sigmoide, gelu_tanh

torch.set_default_dtype(torch.float64)
rng = np.random.default_rng(0)
z = np.r_[rng.normal(0, 4, 100_000), np.linspace(-40, 40, 8001)]
z = z[np.abs(z) > 1e-6]                              # fuera del pliegue de ReLU y Leaky ReLU en 0
REF = {"sigmoide": torch.sigmoid, "tanh": torch.tanh, "relu": torch.relu,
       "leaky_relu": lambda t: F.leaky_relu(t, 0.01), "gelu": lambda t: F.gelu(t)}
h = 1e-5
print(f"{len(z):,} puntos en [−40, 40]           valor vs torch   derivada vs autograd   vs dif. centrales")
for nombre, (f, df) in ACTIVACIONES.items():
    t = torch.tensor(z, requires_grad=True)
    y = REF[nombre](t); y.sum().backward()           # el gradiente de la suma = derivada punto a punto
    e_val = np.max(np.abs(f(z) - y.detach().numpy()))
    e_der = np.max(np.abs(df(z) - t.grad.numpy()))
    e_num = np.max(np.abs(df(z) - (f(z + h) - f(z - h)) / (2 * h))[np.abs(z) > 2 * h])
    print(f"  {nombre:11s}                       {e_val:9.1e}        {e_der:9.1e}             {e_num:9.1e}")
t = torch.tensor(z)
print(f"  gelu (aprox. tanh) vs F.gelu(approximate='tanh'): "
      f"{np.max(np.abs(gelu_tanh(z) - F.gelu(t, approximate='tanh').numpy())):.1e};"
      f"  distancia a la GELU exacta: {np.max(np.abs(gelu_tanh(z) - ACTIVACIONES['gelu'][0](z))):.1e}")
print(f"  escalón vs torch.heaviside(z, 0): {np.max(np.abs(escalon(z) - torch.heaviside(t, torch.tensor(0.)).numpy())):.1e}")
def avisos(f):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always"); r = f()
    return r, [x.category.__name__ for x in w]
(r1, w1), (r2, w2) = avisos(lambda: sigmoide(-800.0)), avisos(lambda: 1 / (1 + np.exp(np.float64(800))))
print(f"  sigmoide(−800): estable = {float(r1):.1e}, avisos {w1};  directa 1/(1+exp(800)) = {float(r2):.1e}, avisos {w2}")
zz = np.linspace(-10, 10, 200_001)
for nombre in ("sigmoide", "tanh", "relu"):
    d = ACTIVACIONES[nombre][1](zz)
    print(f"  {nombre:8s}: derivada máxima {d.max():.4f};  en z = −5: {d[np.searchsorted(zz, -5)]:.4f};"
          f"  en z = 5: {d[np.searchsorted(zz, 5)]:.4f}")
