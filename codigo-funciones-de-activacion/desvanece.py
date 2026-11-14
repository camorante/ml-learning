"""Tamaño del gradiente que llega a cada capa de un perceptrón de 10 capas ocultas (PyTorch)."""
import torch

torch.set_default_dtype(torch.float64)
ACT = {"sigmoide": torch.nn.Sigmoid, "tanh": torch.nn.Tanh, "relu": torch.nn.ReLU}

def red(act, capas=10, ancho=32):
    m = []
    for _ in range(capas):
        l = torch.nn.Linear(ancho, ancho)
        if act == "relu": torch.nn.init.kaiming_normal_(l.weight, nonlinearity="relu")   # He (2015)
        else: torch.nn.init.xavier_normal_(l.weight)                                    # Glorot (2010)
        torch.nn.init.zeros_(l.bias); m += [l, ACT[act]()]
    return torch.nn.Sequential(*m, torch.nn.Linear(ancho, 1))

X, y = torch.randn(256, 32, generator=torch.Generator().manual_seed(1)), torch.ones(256, 1)
print("norma del gradiente de los pesos de cada capa oculta (1 = la primera)")
for act in ACT:
    norma = torch.zeros(10)
    for semilla in range(20):                        # media de 20 inicializaciones
        torch.manual_seed(semilla); net = red(act)
        ((net(X) - y) ** 2).mean().backward()
        norma += torch.tensor([l.weight.grad.norm() for l in net if isinstance(l, torch.nn.Linear)][:10]) / 20
    fila = "  ".join(f"{g:.1e}" for g in norma[[0, 2, 4, 6, 9]])
    print(f"  {act:8s} capas 1, 3, 5, 7, 10: {fila}   primera / última = {norma[0] / norma[9]:.1e}")
