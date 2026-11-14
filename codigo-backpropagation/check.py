"""retro.py contra el autograd de PyTorch con los mismos pesos: 500 redes al azar
(1 a 4 capas ocultas, 1 a 16 neuronas, sigmoide/tanh/ReLU, BCE/CE/MSE, lotes de 1 a 64), en float64."""
import numpy as np
import torch
from retro import Densa, MLP, gradiente_vector

torch.set_default_dtype(torch.float64)
rng = np.random.default_rng(5)
TACT = {"sigmoide": torch.sigmoid, "tanh": torch.tanh, "relu": torch.relu}
peor_abs, peor_rel, n_par = 0.0, 0.0, []
for k in range(500):
    act, perd = rng.choice(["sigmoide", "tanh", "relu"]), rng.choice(["bce", "ce", "mse"])
    d, K, n = int(rng.integers(1, 9)), (int(rng.integers(2, 6)) if perd == "ce" else 1), int(rng.integers(1, 65))
    tam = [d] + [int(rng.integers(1, 17)) for _ in range(rng.integers(1, 5))] + [K]
    W = [rng.normal(0, 1 / np.sqrt(a), (b, a)) for a, b in zip(tam, tam[1:])]
    B = [rng.normal(0, 0.5, b) for b in tam[1:]]
    X = rng.normal(0, 1, (n, d))
    y = rng.integers(0, K, n) if perd == "ce" else (rng.integers(0, 2, n) if perd == "bce" else rng.normal(0, 1, n))
    red = MLP([Densa(w, b, act if i < len(W) - 1 else "identidad") for i, (w, b) in enumerate(zip(W, B))], perd)
    g = gradiente_vector(red, X, y)
    # PyTorch: mismos pesos, mismo cálculo hacia delante, gradientes por autograd
    Wt = [torch.tensor(w, requires_grad=True) for w in W]; Bt = [torch.tensor(b, requires_grad=True) for b in B]
    H = torch.tensor(X)
    for i, (w, b) in enumerate(zip(Wt, Bt)):
        H = H @ w.T + b
        if i < len(Wt) - 1: H = TACT[act](H)
    yt = torch.tensor(y)
    L = (torch.nn.functional.binary_cross_entropy_with_logits(H[:, 0], yt.double()) if perd == "bce" else
         torch.nn.functional.cross_entropy(H, yt) if perd == "ce" else torch.mean((H[:, 0] - yt) ** 2))
    L.backward()
    gt = torch.cat([torch.cat([w.grad.ravel(), b.grad]) for w, b in zip(Wt, Bt)]).numpy()
    peor_abs = max(peor_abs, np.max(np.abs(g - gt)))
    peor_rel = max(peor_rel, np.linalg.norm(g - gt) / max(np.linalg.norm(gt), 1e-300))
    n_par.append(len(g))
print(f"PyTorch {torch.__version__}: 500 redes de {min(n_par)} a {max(n_par)} parámetros")
print(f"  error absoluto máximo en un gradiente: {peor_abs:.1e}")
print(f"  error relativo máximo ‖g − g_torch‖ / ‖g_torch‖: {peor_rel:.1e}")
