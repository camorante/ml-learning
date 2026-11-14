"""Sin activación, una red de varias capas es una sola capa lineal: comprobación numérica."""
import numpy as np
import torch

torch.set_default_dtype(torch.float64)
torch.manual_seed(0)
dims = [5, 16, 16, 2, 16, 3]                         # 5 capas lineales, con un cuello de botella de 2
capas = [torch.nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:])]
red = torch.nn.Sequential(*capas)                    # sin ninguna activación entre capas

# W = W5·W4·W3·W2·W1  y  b = W5(W4(W3(W2 b1 + b2) + b3) + b4) + b5
W, b = np.eye(5), np.zeros(5)
for l in capas:
    Wl, bl = l.weight.detach().numpy(), l.bias.detach().numpy()
    W, b = Wl @ W, Wl @ b + bl
X = torch.randn(1000, 5)
with torch.no_grad():
    err = np.max(np.abs(red(X).numpy() - (X.numpy() @ W.T + b)))
print(f"5 capas lineales ({sum(p.numel() for p in red.parameters())} números) = una capa de {W.size + b.size}:"
      f" W {W.shape}, error máximo {err:.1e}")
print(f"rango de W: {np.linalg.matrix_rank(W)} (el cuello de 2 neuronas limita la red entera a rango 2)")

# Con ReLU entre capas ya no hay una W que valga: la mejor capa lineal (mínimos cuadrados) deja error
red_relu = torch.nn.Sequential(*sum([[l, torch.nn.ReLU()] for l in capas[:-1]], []), capas[-1])
with torch.no_grad():
    Y = red_relu(X).numpy()
A = np.c_[X.numpy(), np.ones(len(X))]
coef, *_ = np.linalg.lstsq(A, Y, rcond=None)
r2 = 1 - np.sum((A @ coef - Y) ** 2) / np.sum((Y - Y.mean(0)) ** 2)
print(f"con ReLU: la mejor capa lineal solo explica R² = {r2:.3f} de la salida (sin ReLU, R² = 1)")
