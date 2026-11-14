"""Las redes del juguete: velocidad -> consumo con 1 o 3 capas ocultas de 3 neuronas,
sin activación, con sigmoide, con tanh y con ReLU (PyTorch, CPU, float64)."""
import json
import numpy as np
import torch
from datos import tramos

torch.set_default_dtype(torch.float64)
v, c = tramos()
vn, cn = tramos(200, seed=7)                     # 200 tramos nuevos para medir el error
X = torch.tensor(v / 100)[:, None]               # velocidad en cientos de km/h
Y = torch.tensor((c - 35) / 10)[:, None]          # consumo centrado y escalado
ACT = {"ninguna": torch.nn.Identity, "sigmoide": torch.nn.Sigmoid, "tanh": torch.nn.Tanh, "relu": torch.nn.ReLU}

def red(capas, act):
    m, d = [], 1
    for _ in range(capas):
        m += [torch.nn.Linear(d, 3), ACT[act]()]; d = 3
    return torch.nn.Sequential(*m, torch.nn.Linear(d, 1))

def entrenar(capas, act, semillas=30):
    mejor = None
    for s in range(semillas):
        torch.manual_seed(s); net = red(capas, act)
        opt = torch.optim.Adam(net.parameters(), lr=0.02)
        for _ in range(3000):
            opt.zero_grad(); loss = ((net(X) - Y) ** 2).mean(); loss.backward(); opt.step()
        opt = torch.optim.LBFGS(net.parameters(), max_iter=500, line_search_fn="strong_wolfe")
        def cierre():
            opt.zero_grad(); l = ((net(X) - Y) ** 2).mean(); l.backward(); return l
        opt.step(cierre)
        with torch.no_grad():
            mse = float(((net(X) - Y) ** 2).mean())
        if mejor is None or mse < mejor[0]:
            mejor = (mse, net)
    return mejor[1]

recta = np.polyfit(v, c, 1)                                  # mínimos cuadrados con una recta
print(f"recta de mínimos cuadrados: consumo = {recta[1]:.3f} + {recta[0]:.4f}·v   "
      f"error {np.sqrt(np.mean((np.polyval(recta, v) - c) ** 2)):.3f} L/100 km"
      f" (tramos nuevos {np.sqrt(np.mean((np.polyval(recta, vn) - cn) ** 2)):.3f})")
out = {}
for capas in (1, 3):
    for act in ACT:
        net = entrenar(capas, act)
        with torch.no_grad():
            pred = net(X).numpy().ravel() * 10 + 35
            pn = net(torch.tensor(vn / 100)[:, None]).numpy().ravel() * 10 + 35
            pg = net(torch.tensor([[0.20], [0.75], [1.05]])).numpy().ravel() * 10 + 35
        rmse = float(np.sqrt(np.mean((pred - c) ** 2))); rmse_n = float(np.sqrt(np.mean((pn - cn) ** 2)))
        n_par = sum(p.numel() for p in net.parameters())
        print(f"{capas} capa(s), {act:8s}: {n_par:2d} números, error {rmse:.3f} (tramos nuevos {rmse_n:.3f}),"
              f" a 20/75/105 km/h: {pg[0]:.1f} {pg[1]:.1f} {pg[2]:.1f}")
        capas_w = [[l.weight.detach().numpy().tolist(), l.bias.detach().numpy().tolist()]
                   for l in net if isinstance(l, torch.nn.Linear)]
        out[f"{capas}-{act}"] = dict(rmse=rmse, rmse_n=rmse_n, n=n_par, capas=capas_w)
json.dump(out, open("red_toy.json", "w"))
