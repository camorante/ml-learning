"""Neuronas ReLU muertas: un paso de aprendizaje demasiado grande las apaga para siempre."""
import numpy as np
import torch
from datos import tramos

torch.set_default_dtype(torch.float64)
v, c = tramos()
X = torch.tensor(v / 100)[:, None]; Y = torch.tensor((c - 35) / 10)[:, None]

def entrenar(act, lr, pasos=2000, semilla=0):
    torch.manual_seed(semilla)
    net = torch.nn.Sequential(torch.nn.Linear(1, 20), act, torch.nn.Linear(20, 1))
    with torch.no_grad():
        al_empezar = int(((net[0](X) <= 0).all(0)).sum())
    opt = torch.optim.SGD(net.parameters(), lr=lr)
    for _ in range(pasos):
        opt.zero_grad(); ((net(X) - Y) ** 2).mean().backward(); opt.step()
    with torch.no_grad():
        Z = net[0](X)                                         # entrada de cada neurona en los 60 tramos
        muertas = int(((Z <= 0).all(0)).sum())                 # nunca positiva: gradiente exactamente 0
        rmse = float(((net(X) - Y) ** 2).mean().sqrt()) * 10
    opt.zero_grad(); ((net(X) - Y) ** 2).mean().backward()
    g = net[0].weight.grad.abs().ravel()
    return al_empezar, muertas, rmse, g

print("de 20 neuronas: apagadas (entrada <= 0 en los 60 tramos) al empezar / al acabar; con gradiente 0 al acabar")
for lr in (0.05, 0.3, 0.6):
    for nombre, act in (("ReLU", torch.nn.ReLU()), ("Leaky ReLU", torch.nn.LeakyReLU(0.01))):
        res = [entrenar(act, lr, semilla=s) for s in range(10)]
        m0, m, e = (np.mean([r[k] for r in res]) for k in range(3))
        cero = np.mean([int((r[3] == 0).sum()) for r in res])
        print(f"  lr = {lr:<4} {nombre:10s}: apagadas {m0:4.1f} / {m:4.1f},  gradiente 0: {cero:4.1f},"
              f"  error {e:.2f} L/100 km   (media de 10 semillas)")
# con la entrada centrada, (v − 60)/30, nacen muertas muchas menos
Xc = torch.tensor((v - 60) / 30)[:, None]
nacen = []
for s in range(10):
    torch.manual_seed(s); nacen.append(int(((torch.nn.Linear(1, 20)(Xc) <= 0).all(0)).sum()))
print(f"entrada centrada: apagadas al empezar {np.mean(nacen):.1f} de 20 (media de 10 semillas)")
