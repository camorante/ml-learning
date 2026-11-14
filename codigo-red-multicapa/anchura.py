"""Zona normal del motor con una capa oculta de m neuronas sigmoide (m = 1…8), entrenada fuera de la
página con PyTorch (Adam + L-BFGS, 12 arranques). Entradas estandarizadas con la elipse «normal»."""
import json
import numpy as np
import torch
from datos import motor, motor_nuevas, RPM0, RPM_S, TMP0, TMP_S
from red import MLP, Densa, n_param

torch.set_default_dtype(torch.float64)
X, y = motor(); Xn, yn = motor_nuevas()
esc = lambda A: np.c_[(A[:, 0] - RPM0) / RPM_S, (A[:, 1] - TMP0) / TMP_S]
Xt, Yt = torch.tensor(esc(X)), torch.tensor(y, dtype=torch.float64)


def entrenar(m, semillas=12):
    mejor = None
    for s in range(semillas):
        torch.manual_seed(s)
        net = torch.nn.Sequential(torch.nn.Linear(2, m), torch.nn.Sigmoid(), torch.nn.Linear(m, 1))
        perdida = lambda: torch.nn.functional.binary_cross_entropy_with_logits(net(Xt)[:, 0], Yt) \
            + 1e-4 * sum((p ** 2).sum() for n, p in net.named_parameters() if "weight" in n)
        opt = torch.optim.Adam(net.parameters(), lr=0.05)
        for _ in range(1500):
            opt.zero_grad(); L = perdida(); L.backward(); opt.step()
        opt = torch.optim.LBFGS(net.parameters(), max_iter=1000, line_search_fn="strong_wolfe")
        def cierre():
            opt.zero_grad(); L = perdida(); L.backward(); return L
        opt.step(cierre)
        with torch.no_grad():
            L = perdida().item()
        if mejor is None or L < mejor[0]:
            mejor = (L, net)
    n = mejor[1]
    return MLP([Densa(n[0].weight.detach().numpy(), n[0].bias.detach().numpy()),
                Densa(n[2].weight.detach().numpy(), n[2].bias.detach().numpy())]), mejor[0]


print(f"{len(y)} lecturas revisadas ({y.sum()} anomalías), {len(yn)} nuevas ({yn.sum()} fuera de la zona)")
print(" m   parámetros   pérdida   errores (de 120)   acierto en nuevas")
out = {}
for m in (1, 2, 3, 4, 5, 6, 8):
    red, L = entrenar(m)
    e = int(np.sum(red.predict(esc(X)) != y)); a = float(np.mean(red.predict(esc(Xn)) == yn))
    assert red.n_param == n_param([2, m, 1]) == 4 * m + 1
    print(f"{m:2d}   {red.n_param:6d}      {L:.4f}      {e:4d}            {100 * a:5.1f} %")
    c1, c2 = red.capas
    out[m] = {"W": np.round(c1.W, 4).tolist(), "b": np.round(c1.b, 4).tolist(),
              "v": np.round(c2.W[0], 4).tolist(), "c": round(float(c2.b[0]), 4), "e": e, "a": round(a, 4)}
json.dump(out, open("anchura.json", "w", encoding="utf-8"))
