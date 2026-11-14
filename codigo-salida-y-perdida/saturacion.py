"""Por qué no error cuadrático sobre probabilidades: el empujón en z para un camión que SÍ se averió,
y una neurona sigmoide entrenada desde un mal comienzo con cada pérdida (PyTorch, autograd)."""
import numpy as np
import torch
from datos import camiones
from perdidas import sigmoide

print(" z     p        CE=−log p   empujón CE = p−1   MSE=(p−1)²   empujón MSE = 2(p−1)p(1−p)")
for z in (-6, -4, -2, 0, 2):
    p = sigmoide(z)
    print(f"{z:3d}  {p:.4f}   {-np.log(p):8.3f}     {p - 1:+.4f}          {(p - 1) ** 2:.4f}       {2 * (p - 1) * p * (1 - p):+.5f}")

x, y = camiones(40, seed=4)                       # 40 camiones: miles de km desde la revisión y avería sí/no
u = torch.tensor((x - 30) / 30); t = torch.tensor(y, dtype=torch.float64)
print(f"\n40 camiones, {int(y.sum())} averías. Neurona p = σ(b + w·u), u = (km − 30 000)/30 000.")
print("Empieza al revés (w = −6: «más km, menos averías»), descenso de gradiente con paso 0,5:")
for nombre in ("entropía cruzada", "error cuadrático sobre p"):
    w = torch.tensor(-6.0, dtype=torch.float64, requires_grad=True)
    b = torch.tensor(-2.5, dtype=torch.float64, requires_grad=True)
    fila = []
    for paso in range(1, 2001):
        z = b + w * u
        L = torch.nn.functional.binary_cross_entropy_with_logits(z, t) if nombre.startswith("entropía") \
            else torch.mean((torch.sigmoid(z) - t) ** 2)
        w.grad = b.grad = None; L.backward()
        if paso == 1: g0 = w.grad.item()
        with torch.no_grad(): w -= 0.5 * w.grad; b -= 0.5 * b.grad
        if paso in (10, 100, 2000): fila.append(f"paso {paso}: w = {w.item():+.2f}")
    print(f"  {nombre:25s} gradiente inicial en w {g0:+.4f}   " + "   ".join(fila))
