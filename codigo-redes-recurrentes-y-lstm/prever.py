"""Memoria corta: predecir la velocidad del camión dentro de 3 segundos a partir de lo que lleva leído.
Una salida en cada paso (muchos a muchos). Aquí basta con recordar los últimos segundos: la RNN y la LSTM
aciertan casi lo mismo, y las dos mejoran a «seguirá igual» y a «seguirá la tendencia»."""
import os
import numpy as np
import torch
from torch import nn

torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)
os.chdir(os.path.dirname(os.path.abspath(__file__)))
T, AV = 120, 3                 # 120 s de velocidad, una lectura por segundo; se predice 3 s por delante


def trayecto(rng):
    """Velocidad en km/h: el camión persigue una velocidad objetivo que cambia (parar, calle, avenida,
    carretera) con aceleración limitada y un poco de ruido del sensor."""
    v, obj, out = rng.uniform(0, 60), rng.choice([0, 30, 50, 80]), []
    for _ in range(T + AV):
        if rng.random() < 0.06:
            obj = rng.choice([0, 30, 50, 80])
        a = np.clip(0.35 * (obj - v), -6.0, 4.0)          # km/h por segundo: frena más fuerte de lo que acelera
        v = max(0.0, v + a + rng.normal(0, 0.6))
        out.append(v)
    out = np.array(out)
    return out + rng.normal(0, 0.4, len(out))


def conjunto(n, seed):
    rng = np.random.default_rng(seed)
    V = np.array([trayecto(rng) for _ in range(n)])
    return V[:, :T], V[:, AV:AV + T]                         # entrada v_1…v_120 y objetivo v_4…v_123


class Prevision(nn.Module):
    def __init__(s, tipo, H=16):
        super().__init__()
        s.rec = {"RNN": nn.RNN, "LSTM": nn.LSTM}[tipo](1, H, batch_first=True); s.out = nn.Linear(H, 1)

    def forward(s, v):                                     # (n, T) km/h → (n, T) km/h previstos
        return 40 + 40 * s.out(s.rec((v[:, :, None] - 40) / 40)[0])[:, :, 0]


def entrenar(tipo, X, Y, epocas=30, seed=0):
    torch.manual_seed(seed)
    m = Prevision(tipo); opt = torch.optim.Adam(m.parameters(), lr=0.005)
    X, Y = torch.tensor(X), torch.tensor(Y); g = torch.Generator().manual_seed(seed + 1)
    for _ in range(epocas):
        orden = torch.randperm(len(X), generator=g)
        for k in range(0, len(X), 32):
            j = orden[k:k + 32]; opt.zero_grad()
            nn.functional.mse_loss(m(X[j]), Y[j]).backward()
            nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
    return m


X_TR, Y_TR = conjunto(1000, 91)
X_TE, Y_TE = conjunto(300, 92)
if __name__ == "__main__":
    ini = 10                                               # se evalúa a partir del segundo 10
    err = lambda P: float(np.mean(np.abs(P[:, ini:] - Y_TE[:, ini:])))
    print(f"error medio (km/h) al prever la velocidad {AV} s por delante, 300 trayectos de prueba:")
    print(f"  «seguirá igual» (la última lectura):       {err(X_TE):.2f}")
    tend = X_TE + AV * np.c_[np.zeros(len(X_TE)), np.diff(X_TE, axis=1)]
    print(f"  «seguirá la tendencia» (último cambio × 3): {err(tend):.2f}")
    P = {}
    for tipo in ("RNN", "LSTM"):
        m = entrenar(tipo, X_TR, Y_TR)
        with torch.no_grad():
            P[tipo] = m(torch.tensor(X_TE)).numpy()
        print(f"  {tipo:4s} de 16 unidades:                    {err(P[tipo]):.2f}")
        torch.save(m.state_dict(), f"prever_{tipo}.pt")
