"""Los modelos del tema 10 en PyTorch (float64) y el bucle de entrenamiento.
Emb: código visto → embedding de d números → capa lineal → 20 logits (softmax: ¿qué otro código hay?).
Tabla: código visto → one-hot de 20 → capa lineal → 20 logits (sin embedding: una fila libre por código)."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

torch.set_default_dtype(torch.float64)


def sin_si_mismo(z, a):
    """Un código no puede acompañarse a sí mismo en una visita: su propio logit vale −∞ (probabilidad 0)."""
    return z.masked_fill(Fn.one_hot(a, z.shape[1]).bool(), float("-inf"))


class Emb(nn.Module):
    def __init__(self, V, d):
        super().__init__()
        self.E = nn.Embedding(V, d)           # la tabla de embeddings: V filas de d números, N(0, 1) al empezar
        self.out = nn.Linear(d, V)            # un vector de salida (y un sesgo) por código

    def forward(self, a):
        return sin_si_mismo(self.out(self.E(a)), a)


class Tabla(nn.Module):
    def __init__(self, V):
        super().__init__()
        self.V = V
        self.out = nn.Linear(V, V)            # one-hot × W: cada código tiene su propia fila de 20 logits

    def forward(self, a):
        return sin_si_mismo(self.out(Fn.one_hot(a, self.V).double()), a)


def n_param(m):
    return sum(p.numel() for p in m.parameters())


def evaluar(m, P):
    """Entropía cruzada media (nats) y aciertos «el otro código está entre los 3 más probables»."""
    with torch.no_grad():
        z = m(torch.tensor(P[:, 0])); y = torch.tensor(P[:, 1])
        ce = Fn.cross_entropy(z, y).item()
        top3 = (z.topk(3, dim=1).indices == y[:, None]).any(1).double().mean().item()
    return ce, top3


def entrenar(m, P_tr, P_te, epocas=60, lr=0.02, lote=64, seed=0, fotos=(), wd=0.0):
    """Adam con lotes de `lote` pares barajados con semilla fija. Devuelve el historial y las «fotos» de la
    tabla de embeddings en las épocas pedidas (época 0 = antes de entrenar)."""
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=wd)
    A, Y = torch.tensor(P_tr[:, 0]), torch.tensor(P_tr[:, 1])
    hist, snaps = [], {}
    for ep in range(epocas + 1):
        if ep in fotos:
            snaps[ep] = m.E.weight.detach().numpy().copy()
        if ep:
            perm = torch.randperm(len(A), generator=g)
            for i in range(0, len(A), lote):
                idx = perm[i:i + lote]
                loss = Fn.cross_entropy(m(A[idx]), Y[idx])
                opt.zero_grad(); loss.backward(); opt.step()
        hist.append((ep, *evaluar(m, P_tr), *evaluar(m, P_te)))
    return hist, snaps


def coseno(E):
    """Matriz de similitudes coseno entre filas, con el mismo orden de sumas que el JS."""
    V, d = E.shape
    n = np.zeros(V)
    for i in range(V):
        a = 0.0
        for k in range(d):
            a += E[i, k] * E[i, k]
        n[i] = np.sqrt(a)
    S = np.zeros((V, V))
    for i in range(V):
        for j in range(V):
            a = 0.0
            for k in range(d):
                a += E[i, k] * E[j, k]
            S[i, j] = a / (n[i] * n[j])
    return S


def vecinos(S, i, k=3):
    """Los k códigos más parecidos a i por coseno (sin él mismo); empates: el índice menor primero."""
    orden = sorted((j for j in range(len(S)) if j != i), key=lambda j: (-S[i, j], j))
    return orden[:k]


def pureza(E, sistema, k=3):
    """Fracción de los k vecinos por coseno que son del mismo sistema principal, sobre todos los códigos."""
    S = coseno(E)
    ok = sum(sistema[j] == sistema[i] for i in range(len(E)) for j in vecinos(S, i, k))
    return int(ok), len(E) * k
