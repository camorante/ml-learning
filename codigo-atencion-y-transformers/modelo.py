"""El Transformer diminuto del tema en PyTorch (las mismas cuentas que atencion.py) y su entrenamiento:
predecir la ficha siguiente del diario del viaje, con máscara causal, Adam con calentamiento y semillas fijas."""
import time
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from datos import V, TMAX

torch.set_default_dtype(torch.float64)


def posiciones(n, d):
    p = torch.arange(n)[:, None].double()
    w = 1.0 / 10000 ** (torch.arange(0, d, 2).double() / d)
    PE = torch.zeros(n, d)
    PE[:, 0::2], PE[:, 1::2] = torch.sin(p * w), torch.cos(p * w)
    return PE


class Lectora(nn.Module):
    """La alternativa del tema 9: embedding (V×d) → LSTM de H unidades, ficha a ficha → lineal H→V.
    Como en el tema 9, la puerta de olvido empieza casi abierta (sesgo 3) para que la memoria dure."""
    def __init__(s, d=16, H=20, sesgo_olvido=3.0):
        super().__init__()
        s.E, s.rec, s.sal = nn.Embedding(V, d), nn.LSTM(d, H, batch_first=True), nn.Linear(H, V)
        with torch.no_grad():
            s.rec.bias_ih_l0[H:2 * H] = sesgo_olvido; s.rec.bias_hh_l0[H:2 * H] = 0.0

    def forward(s, ids, causal=True):
        return s.sal(s.rec(s.E(ids))[0])


class Viajero(nn.Module):
    """embedding (V×d) + posición → LN → atención de h cabezas → residual → LN → FFN d→4d→d → residual → LN → d→V"""
    def __init__(s, d=16, h=2, con_posicion=True):
        super().__init__()
        s.d, s.h, s.con_posicion = d, h, con_posicion
        s.E = nn.Embedding(V, d)
        s.ln1, s.ln2, s.lnf = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        s.q, s.k, s.v, s.o = (nn.Linear(d, d) for _ in range(4))
        s.f1, s.f2 = nn.Linear(d, 4 * d), nn.Linear(4 * d, d)
        s.sal = nn.Linear(d, V)
        s.register_buffer("PE", posiciones(TMAX, d))

    def forward(s, ids, causal=True, pesos=False):
        n, T = ids.shape
        X = s.E(ids) + (s.PE[:T] if s.con_posicion else 0)
        A = s.ln1(X)
        dh = s.d // s.h
        sep = lambda Z: Z.view(n, T, s.h, dh).transpose(1, 2)          # (n, h, T, dh)
        Q, K, Vv = sep(s.q(A)), sep(s.k(A)), sep(s.v(A))
        S = Q @ K.transpose(-1, -2) / dh ** 0.5
        if causal:
            S = S.masked_fill(torch.triu(torch.ones(T, T, dtype=torch.bool), 1), float("-inf"))
        W = S.softmax(-1)
        H = X + s.o((W @ Vv).transpose(1, 2).reshape(n, T, s.d))
        Y = H + s.f2(F.relu(s.f1(s.ln2(H))))
        Z = s.sal(s.lnf(Y))
        return (Z, W) if pesos else Z


def n_param(m):
    return sum(p.numel() for p in m.parameters())


def a_numpy(m):
    """Los pesos del modelo con la convención de atencion.py (X @ W: se trasponen los de nn.Linear)."""
    t = lambda x: x.detach().numpy().copy()
    P = {"E": t(m.E.weight), "Wout": t(m.sal.weight).T, "bout": t(m.sal.bias)}
    for n, lin in (("q", m.q), ("k", m.k), ("v", m.v), ("o", m.o)):
        P["W" + n], P["b" + n] = t(lin.weight).T, t(lin.bias)
    P["W1"], P["b1"], P["W2"], P["b2"] = t(m.f1.weight).T, t(m.f1.bias), t(m.f2.weight).T, t(m.f2.bias)
    for n, ln in (("1", m.ln1), ("2", m.ln2), ("f", m.lnf)):
        P["g" + n], P["be" + n] = t(ln.weight), t(ln.bias)
    return P


def perdida(Z, X, L):
    """Entropía cruzada de predecir X[:, t+1] desde la posición t, solo dentro de cada viaje."""
    T = X.shape[1]
    dentro = torch.arange(T - 1)[None, :] < (L[:, None] - 1)
    return F.cross_entropy(Z[:, :-1][dentro], X[:, 1:][dentro])


def entrenar(X, L, Xv, Lv, d=16, h=2, con_posicion=True, epocas=100, lr=3e-3, calentamiento=200, lote=32, seed=0, crear=None):
    """Adam con calentamiento lineal del ritmo de aprendizaje (tema 6) durante los primeros pasos.
    Devuelve el modelo, la pérdida de validación por época y los segundos que ha tardado."""
    torch.manual_seed(seed)
    m = crear() if crear else Viajero(d, h, con_posicion)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda paso: min(1.0, (paso + 1) / calentamiento))
    X, L, Xv, Lv = (torch.tensor(a) for a in (X, L, Xv, Lv))
    g = torch.Generator().manual_seed(seed + 1)
    curva, t0 = [], time.perf_counter()
    for _ in range(epocas):
        orden = torch.randperm(len(X), generator=g)
        for k in range(0, len(X), lote):
            j = orden[k:k + lote]
            opt.zero_grad(); perdida(m(X[j]), X[j], L[j]).backward(); opt.step(); sched.step()
        with torch.no_grad():
            curva.append(perdida(m(Xv), Xv, Lv).item())
    return m, curva, time.perf_counter() - t0


def aciertos(pred, X, preguntas):
    """Fracción de preguntas (viaje, posición) en las que la ficha más probable es la siguiente de verdad."""
    return float(np.mean([pred[i, t] == X[i, t + 1] for i, t, _ in preguntas]))
