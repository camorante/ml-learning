"""La red del tema 3 (2 entradas → m neuronas → 1 salida) con tanh en la capa oculta y entropía cruzada
binaria desde el logit. La pasada hacia atrás (el gradiente) es la retropropagación del tema 5."""
import numpy as np


def softplus(t):
    return np.maximum(t, 0) + np.log1p(np.exp(-np.abs(t)))


def sigmoide(z):
    e = np.exp(-np.abs(z))
    return np.where(z >= 0, 1 / (1 + e), e / (1 + e))


def iniciar(m=8, d=2, seed=0):
    """Como nn.Linear de PyTorch: pesos y sesgos ~ U(−1/√entradas, 1/√entradas). Orden: W1, b1, W2, b2."""
    rng = np.random.default_rng(seed)
    a1, a2 = 1 / np.sqrt(d), 1 / np.sqrt(m)
    return [rng.uniform(-a1, a1, (m, d)), rng.uniform(-a1, a1, m),
            rng.uniform(-a2, a2, (1, m)), rng.uniform(-a2, a2, 1)]


def logit(P, X):
    W1, b1, W2, b2 = P
    return (np.tanh(X @ W1.T + b1) @ W2.T + b2)[:, 0]


def perdida(P, X, y):
    """Entropía cruzada binaria media (nats), calculada desde el logit de forma estable."""
    z = logit(P, X)
    return float(np.mean(y * softplus(-z) + (1 - y) * softplus(z)))


def acierto(P, X, y):
    return float(np.mean((logit(P, X) > 0) == y))


def gradiente(P, X, y):
    """Pérdida media del lote y su gradiente respecto a cada parámetro (en el mismo orden que P)."""
    W1, b1, W2, b2 = P
    H = np.tanh(X @ W1.T + b1)                       # (n, m)
    z = (H @ W2.T + b2)[:, 0]                        # (n,)
    L = float(np.mean(y * softplus(-z) + (1 - y) * softplus(z)))
    dz = (sigmoide(z) - y) / len(y)                  # ∂L/∂z = (p − y)/n (tema 4)
    dH = dz[:, None] * W2                            # hacia atrás por la capa de salida
    dA = dH * (1 - H ** 2)                           # tanh'(a) = 1 − tanh(a)²  (tema 2)
    return L, [dA.T @ X, dA.sum(0), dz[None, :] @ H, np.array([dz.sum()])]
