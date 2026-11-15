"""Entrenamiento y evaluación con normalización por lotes: una red 1-32-1 con BatchNorm1d antes de la tanh, entrenada en
PyTorch con lotes de 5 viajes. En modo entrenamiento, la predicción de un viaje depende de con qué otros viajes vaya en el
lote; en modo evaluación (model.eval()), se usan las estadísticas móviles y la respuesta es una sola."""
import json, os
import numpy as np
import torch
from datos import V_TR, Y_TR, ex, ey, SY, MY

torch.manual_seed(0)
net = torch.nn.Sequential(torch.nn.Linear(1, 32), torch.nn.BatchNorm1d(32), torch.nn.Tanh(), torch.nn.Linear(32, 1))
X, y = torch.tensor(ex(V_TR), dtype=torch.float32), torch.tensor(ey(Y_TR), dtype=torch.float32)
opt = torch.optim.Adam(net.parameters(), lr=0.01)
g = torch.Generator().manual_seed(1)
for ep in range(300):
    net.train()
    for j in torch.randperm(20, generator=g).split(5):            # 4 lotes de 5 viajes por época
        opt.zero_grad()
        torch.nn.functional.mse_loss(net(X[j])[:, 0], y[j]).backward()
        opt.step()
LOTES = int(net[1].num_batches_tracked)                           # lotes vistos al entrenar

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    v85 = torch.tensor(ex([85.0]), dtype=torch.float32)
    rng = np.random.default_rng(3)
    otros = [rng.choice(20, 4, replace=False) for _ in range(20)]
    bn = net[1]
    rm, rv = bn.running_mean.clone(), bn.running_var.clone()
    net.train()
    with torch.no_grad():
        tr = [float(net(torch.cat([v85, X[o]]))[0, 0]) * SY + MY for o in otros]   # el viaje de 85 km/h con 4 viajes al azar
    bn.running_mean.copy_(rm); bn.running_var.copy_(rv)            # deshacer lo que esas pasadas movieron las estadísticas
    net.eval()
    with torch.no_grad():
        ev = float(net(v85)[0, 0]) * SY + MY
    print(f"viaje a 85 km/h, modo entrenamiento en 20 lotes distintos: de {min(tr):.2f} a {max(tr):.2f} L/100 km"
          f" (media {np.mean(tr):.2f}); modo evaluación: {ev:.2f}")
    net.train()
    try:
        net(v85)
    except ValueError as e:
        print("un lote de 1 en modo entrenamiento:", str(e).split(",")[0])
    print(f"estadísticas móviles de la neurona 1: media {float(rm[0]):.4f}, varianza {float(rv[0]):.4f}"
          f" (tras {LOTES} lotes de entrenamiento)")
    json.dump(dict(tr=[round(v, 2) for v in tr], ev=round(ev, 2)), open("modos.json", "w", encoding="utf-8"), separators=(",", ":"))
