"""Un paso de entrenamiento con los 30 viajes: la red 2-2-1 mal colocada, sus 9 gradientes,
θ ← θ − η∇L con η = 0.03, y el mismo paso hecho por PyTorch (autograd + torch.optim.SGD)."""
import numpy as np
import torch
from datos import carga_consumo, red_inicial, ETA

X, y = carga_consumo()
red = red_inicial()
errores = lambda r: int(np.sum((r.adelante(X)[:, 0] > 0) != y))     # p > 0.5  ⇔  z > 0
L0, g = red.gradientes(X, y)
nombres = ["wA·carga", "wA·consumo", "wB·carga", "wB·consumo", "bA", "bB", "vA", "vB", "c"]
grads = np.r_[g[0][0].ravel(), g[0][1], g[1][0].ravel(), g[1][1]]
print(f"antes: pérdida media {L0:.4f}, {errores(red)} errores de 30")
for n, v in zip(nombres, grads):
    print(f"  ∂L/∂{n:11s} = {v:+.5f}")
th0 = red.vector()
red.paso(X, y, ETA)
L1 = red.perdida_media(X, y)
print(f"después de un paso (η = {ETA}): pérdida media {L1:.4f}, {errores(red)} errores de 30")
W, b = red.capas[0].W, red.capas[0].b
print(f"  recta B: consumo = {-b[1] / W[1, 1]:.2f} + {-W[1, 0] / W[1, 1]:.3f}·carga   (antes 18 + 0.55·carga)")

# lo mismo con PyTorch
torch.set_default_dtype(torch.float64)
net = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Sigmoid(), torch.nn.Linear(2, 1))
with torch.no_grad():
    net[0].weight[:] = torch.tensor(red_inicial().capas[0].W); net[0].bias[:] = torch.tensor(red_inicial().capas[0].b)
    net[2].weight[:] = torch.tensor(red_inicial().capas[1].W); net[2].bias[:] = torch.tensor(red_inicial().capas[1].b)
opt = torch.optim.SGD(net.parameters(), lr=ETA)
perd = torch.nn.BCEWithLogitsLoss()
Xt, yt = torch.tensor(X), torch.tensor(y, dtype=torch.float64)
loss = perd(net(Xt)[:, 0], yt); opt.zero_grad(); loss.backward()
gt = torch.cat([net[0].weight.grad.ravel(), net[0].bias.grad, net[2].weight.grad.ravel(), net[2].bias.grad]).numpy()
opt.step()
tht = torch.cat([p.detach().ravel() for p in (net[0].weight, net[0].bias, net[2].weight, net[2].bias)]).numpy()
print(f"PyTorch: pérdida {loss.item():.4f} → {perd(net(Xt)[:, 0], yt).item():.4f};"
      f" diferencia máxima en gradientes {np.max(np.abs(gt - grads)):.1e}, en pesos nuevos {np.max(np.abs(tht - red.vector())):.1e}")
