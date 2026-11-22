"""El autoencoder del tema en PyTorch y su entrenamiento con semillas fijas (Adam, lotes de 64, error cuadrático medio).
codificador 64 → 32 (tanh) → k   ·   decodificador k → 32 (tanh) → 64"""
import numpy as np
import torch
from torch import nn

torch.set_default_dtype(torch.float64)


class AE(nn.Module):
    def __init__(s, k=4, m=32, n=64, lineal=False):
        super().__init__()
        act = nn.Identity() if lineal else nn.Tanh()
        s.cod = nn.Sequential(nn.Linear(n, m), act, nn.Linear(m, k)) if m else nn.Linear(n, k)
        s.dec = nn.Sequential(nn.Linear(k, m), act, nn.Linear(m, n)) if m else nn.Linear(k, n)

    def forward(s, x):
        return s.dec(s.cod(x))


def entrenar(X, k=4, m=32, epocas=150, lr=3e-3, lote=64, seed=0, ruido=0.0, lineal=False, n=None):
    """Entrena con X como entrada y como objetivo. Con ruido > 0, la entrada lleva ruido extra (denoising)
    y el objetivo es la ventana sin ese ruido extra. Devuelve el modelo y el error medio de entrenamiento por época."""
    torch.manual_seed(seed)
    net = AE(k, m, n or X.shape[1], lineal)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    X = torch.tensor(X)
    g = torch.Generator().manual_seed(seed + 1)
    curva = []
    for _ in range(epocas):
        orden, tot = torch.randperm(len(X), generator=g), 0.0
        for i in range(0, len(X), lote):
            xb = X[orden[i:i + lote]]
            xin = xb + ruido * torch.randn(xb.shape, generator=g) if ruido else xb
            opt.zero_grad()
            perd = ((net(xin) - xb) ** 2).mean()
            perd.backward(); opt.step()
            tot += perd.item() * len(xb)
        curva.append(tot / len(X))
    return net, curva


def reconstruir(net, X):
    with torch.no_grad():
        return net(torch.tensor(np.atleast_2d(X))).numpy()


def codigo(net, X):
    with torch.no_grad():
        return net.cod(torch.tensor(np.atleast_2d(X))).numpy()


def error(net, X):
    """Error de reconstrucción de cada ventana: la media de los 64 cuadrados (m/s²)²."""
    return ((reconstruir(net, X) - X) ** 2).mean(1)


def metricas(err, malo, umbral):
    """Alarmas con err > umbral. malo: 1 si la ventana tiene avería. Devuelve un dict con los recuentos."""
    al = err > umbral
    vp, fp = int((al & malo).sum()), int((al & ~malo).sum())
    fn = int((~al & malo).sum())
    prec = vp / (vp + fp) if vp + fp else 1.0
    rec = vp / (vp + fn)
    return {"vp": vp, "fp": fp, "fn": fn, "precision": prec, "recall": rec,
            "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0}
