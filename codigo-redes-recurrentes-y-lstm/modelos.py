"""Las redes del tema, en PyTorch, y un bucle de entrenamiento con semillas fijas (Adam, lotes de 32, recorte
del gradiente). Las tres leen el tramo lectura a lectura y responden con su último estado oculto."""
import numpy as np
import torch
from torch import nn

torch.set_default_dtype(torch.float64)
ESCALA = 4.0                                   # m/s²: las lecturas entran divididas por 4, casi todas en [−1, 1]
CAPAS = {"RNN": nn.RNN, "LSTM": nn.LSTM, "GRU": nn.GRU}


class Lectora(nn.Module):
    """celda recurrente (RNN con tanh, LSTM o GRU) de H unidades · lineal H→1 sobre el último estado."""
    def __init__(s, tipo="LSTM", H=16, sesgo_olvido=3.0):
        super().__init__()
        s.tipo, s.rec, s.out = tipo, CAPAS[tipo](1, H, batch_first=True), nn.Linear(H, 1)
        if tipo != "RNN" and sesgo_olvido is not None:
            # al empezar, la puerta que conserva la memoria casi abierta: la de olvido de la LSTM (orden i, f, g, o)
            # y la de actualización z de la GRU (orden r, z, n) ocupan el segundo bloque de H filas
            with torch.no_grad():
                s.rec.bias_ih_l0[H:2 * H] = sesgo_olvido; s.rec.bias_hh_l0[H:2 * H] = 0.0

    def estados(s, x):                        # x: (n, T) en m/s² → estados ocultos h_1 … h_T: (n, T, H)
        return s.rec(x[:, :, None] / ESCALA)[0]

    def forward(s, x):
        return s.out(s.estados(x)[:, -1])[:, 0]           # logit de «hubo frenada brusca al salir»


def n_param(m):
    return sum(p.numel() for p in m.parameters())


def entrenar(tipo, H, X, y, Xv, yv, epocas=60, lr=0.003, lote=32, recorte=1.0, seed=0, sesgo_olvido=3.0):
    """Adam con entropía cruzada binaria y recorte de la norma del gradiente. Se queda con la época de mejor
    acierto en validación. Devuelve el modelo y la curva (acierto de validación por época)."""
    torch.manual_seed(seed)
    m = Lectora(tipo, H, sesgo_olvido)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    X, y, Xv, yv = (torch.tensor(a, dtype=torch.float64) for a in (X, y, Xv, yv))
    g = torch.Generator().manual_seed(seed + 1)
    perd = nn.functional.binary_cross_entropy_with_logits
    curva, mejor, estado = [], -1.0, None
    for _ in range(epocas):
        orden = torch.randperm(len(y), generator=g)
        for k in range(0, len(y), lote):
            j = orden[k:k + lote]
            opt.zero_grad(); perd(m(X[j]), y[j]).backward()
            if recorte:
                nn.utils.clip_grad_norm_(m.parameters(), recorte)
            opt.step()
        with torch.no_grad():
            a = ((m(Xv) > 0).double() == yv).double().mean().item()
        curva.append(a)
        if a > mejor:
            mejor, estado = a, {k: v.clone() for k, v in m.state_dict().items()}
    m.load_state_dict(estado)
    return m, curva


def acierto(m, X, y):
    with torch.no_grad():
        return float(((m(torch.tensor(X)) > 0).numpy() == np.asarray(y)).mean())
