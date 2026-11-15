"""Las tres redes del tema, en PyTorch, y un bucle de entrenamiento con semillas fijas (Adam, lotes de 32)."""
import numpy as np
import torch
from torch import nn

torch.set_default_dtype(torch.float64)


class CNN(nn.Module):
    """conv 1→4 (5) · ReLU · max-pool 2 · conv 4→8 (5) · ReLU · máximo global · lineal 8→1."""
    def __init__(s):
        super().__init__()
        s.c1, s.c2, s.out = nn.Conv1d(1, 4, 5), nn.Conv1d(4, 8, 5), nn.Linear(8, 1)

    def mapas(s, x):                                   # x: (n, 64) → mapas de la capa 1 (n, 4, 60)
        return torch.relu(s.c1(x[:, None, :]))

    def forward(s, x):
        h = nn.functional.max_pool1d(s.mapas(x), 2)    # (n, 4, 30)
        h = torch.relu(s.c2(h))                        # (n, 8, 26)
        return s.out(h.amax(-1))[:, 0]                 # logit de «frenada brusca»


class Densa(nn.Module):
    """lineal 64→32 · ReLU · lineal 32→1: un peso por cada lectura y neurona."""
    def __init__(s, m=32):
        super().__init__()
        s.net = nn.Sequential(nn.Linear(64, m), nn.ReLU(), nn.Linear(m, 1))

    def forward(s, x):
        return s.net(x)[:, 0]


class UnFiltro(nn.Module):
    """La red del juguete: un filtro de 5 pesos · máximo global · lineal 1→1 (sin ReLU: el máximo ya decide)."""
    def __init__(s):
        super().__init__()
        s.c, s.out = nn.Conv1d(1, 1, 5), nn.Linear(1, 1)

    def forward(s, x):
        return s.out(s.c(x[:, None, :]).amax(-1))[:, 0]


def n_param(m):
    return sum(p.numel() for p in m.parameters())


def entrenar(modelo, X, y, Xv, yv, epocas=40, lr=0.01, lote=32, seed=0):
    """Adam, entropía cruzada binaria. Devuelve el modelo y las curvas (pérdida y acierto de validación por época)."""
    torch.manual_seed(seed)
    m = modelo()
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    X, y, Xv, yv = (torch.tensor(a, dtype=torch.float64) for a in (X, y, Xv, yv))
    g = torch.Generator().manual_seed(seed + 1)
    perd = nn.functional.binary_cross_entropy_with_logits
    curva = []
    for _ in range(epocas):
        orden = torch.randperm(len(y), generator=g)
        for k in range(0, len(y), lote):
            j = orden[k:k + lote]
            opt.zero_grad(); perd(m(X[j]), y[j]).backward(); opt.step()
        with torch.no_grad():
            z = m(Xv); curva.append((perd(z, yv).item(), ((z > 0).double() == yv).double().mean().item()))
    return m, curva


def acierto(m, X, y):
    with torch.no_grad():
        return float(((m(torch.tensor(X)) > 0).numpy() == np.asarray(y)).mean())


def prob(m, X):
    with torch.no_grad():
        return torch.sigmoid(m(torch.tensor(np.atleast_2d(X)))).numpy()
