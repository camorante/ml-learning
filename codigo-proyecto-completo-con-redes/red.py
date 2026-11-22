"""La red de varias entradas del tema y su entrenamiento: Adam, pesos de clase y parada temprana con paciencia."""
import copy
import numpy as np
import torch
from torch import nn

torch.set_num_threads(4)


class Red(nn.Module):
    """Rama de secuencia (una GRU de 16 unidades que lee los 28 días: tema 9), embeddings de modelo y depósito
    (tema 10) y la tabla escalada; todo se concatena y pasa por una cabeza densa (tema 3) que da un logit.
    `ramas` elige qué entradas usa (para la ablación); "cnn" cambia la GRU por dos convoluciones 1D con máximo y
    media globales (tema 8); `n_cam` > 0 añade un embedding por camión (la trampa de §2)."""
    def __init__(s, ramas=("seq", "emb", "tab"), n_cam=0):
        super().__init__()
        s.ramas, s.n_cam, d = ramas, n_cam, 0
        if "seq" in ramas:
            s.gru = nn.GRU(6, 16, batch_first=True); d += 16
        if "cnn" in ramas:
            s.c1, s.c2 = nn.Conv1d(6, 8, 5), nn.Conv1d(8, 8, 5); d += 16
        if "emb" in ramas:
            s.e_mod, s.e_dep = nn.Embedding(8, 3), nn.Embedding(10, 3); d += 6
        if "tab" in ramas:
            d += 5
        if n_cam:
            s.e_cam = nn.Embedding(n_cam + 1, 4); d += 4          # fila 0: camión desconocido
        s.h1, s.out, s.drop = nn.Linear(d, 16), nn.Linear(16, 1), nn.Dropout(0.2)

    def partes(s, b):
        P = []
        if "seq" in s.ramas:
            _, hn = s.gru(b["seq"].transpose(1, 2))                # lee (n, 28, 6) día a día
            P.append(hn[0])                                        # estado tras el último día (n, 16)
        if "cnn" in s.ramas:
            h = torch.relu(s.c2(torch.relu(s.c1(b["seq"]))))       # (n, 8, 20)
            P += [h.amax(-1), h.mean(-1)]
        if "emb" in s.ramas:
            P += [s.e_mod(b["mod"]), s.e_dep(b["dep"])]
        if "tab" in s.ramas:
            P.append(b["tab"])
        if s.n_cam:
            P.append(s.e_cam(b["cam"]))
        return torch.cat(P, 1)

    def forward(s, b):
        return s.out(s.drop(torch.relu(s.h1(s.partes(b)))))[:, 0]


def tensores(E, idx=None):
    idx = slice(None) if idx is None else idx
    return {k: torch.as_tensor(v[idx]) for k, v in E.items()}


def logits(m, E, lote=4096):
    m.eval()
    with torch.no_grad():
        return np.concatenate([m(tensores(E, slice(k, k + lote))).numpy() for k in range(0, len(E["y"]), lote)])


def entrenar(Etr, Eva, seed=0, ramas=("seq", "emb", "tab"), n_cam=0, w=None, lr=1e-3, lote=256, epocas=100, paciencia=12):
    """Adam con pérdida logística ponderada: cada avería pesa w (por defecto, sanas/averías del entrenamiento).
    Para cuando la pérdida de validación lleva `paciencia` épocas sin mejorar y devuelve la mejor época."""
    torch.manual_seed(seed)
    y = Etr["y"]
    w = float((1 - y).sum() / y.sum()) if w is None else w
    m = Red(ramas, n_cam)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    perd = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(w))
    g = torch.Generator().manual_seed(seed + 1)
    T = tensores(Etr)
    pond = lambda z, y: float(perd(torch.as_tensor(z), torch.as_tensor(y)))
    curva, mejor, est = [], (np.inf, 0), None
    for ep in range(1, epocas + 1):
        m.train()
        orden = torch.randperm(len(y), generator=g)
        for k in range(0, len(y), lote):
            j = orden[k:k + lote]
            b = {c: v[j] for c, v in T.items()}
            opt.zero_grad(); perd(m(b), b["y"]).backward(); opt.step()
        ztr, zva = logits(m, Etr), logits(m, Eva)
        curva.append((pond(ztr, y), pond(zva, Eva["y"])))
        if curva[-1][1] < mejor[0]:
            mejor, est = (curva[-1][1], ep), copy.deepcopy(m.state_dict())
        elif ep - mejor[1] >= paciencia:
            break
    m.load_state_dict(est)
    return m, dict(curva=curva, mejor=mejor[1], w=w)


def prob(m, E, w):
    """Probabilidad corregida: el peso w multiplica las probabilidades a favor (odds) por w; se deshace restando ln w."""
    return 1 / (1 + np.exp(-(logits(m, E) - np.log(w))))
