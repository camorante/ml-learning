"""Optimizadores desde cero en NumPy con las convenciones EXACTAS de torch.optim (sin weight decay).
Cada uno recibe la lista de parámetros P y la de gradientes G (mismo orden y forma) y actualiza P en su sitio."""
import numpy as np


class SGD:
    """v ← μ·v + (1 − d)·g   (en el primer paso, v ← g sin amortiguar, como PyTorch)
    paso = g + μ·v si nesterov, si no v;   θ ← θ − η·paso.   Con μ = 0 es el SGD de siempre."""

    def __init__(self, lr, momentum=0.0, dampening=0.0, nesterov=False):
        self.lr, self.mu, self.d, self.nesterov, self.v = lr, momentum, dampening, nesterov, None

    def step(self, P, G):
        if self.mu == 0:
            for p, g in zip(P, G):
                p -= self.lr * g
            return
        if self.v is None:
            self.v = [g.copy() for g in G]
        else:
            for v, g in zip(self.v, G):
                v *= self.mu
                v += (1 - self.d) * g
        for p, g, v in zip(P, G, self.v):
            p -= self.lr * (g + self.mu * v if self.nesterov else v)


class RMSprop:
    """s ← α·s + (1 − α)·g²;   θ ← θ − η·g / (√s + ε).   ε va FUERA de la raíz (TensorFlow 1 la ponía dentro)."""

    def __init__(self, lr=0.01, alpha=0.99, eps=1e-8):
        self.lr, self.a, self.eps, self.s = lr, alpha, eps, None

    def step(self, P, G):
        if self.s is None:
            self.s = [np.zeros_like(g) for g in G]
        for p, g, s in zip(P, G, self.s):
            s *= self.a
            s += (1 - self.a) * g * g
            p -= self.lr * g / (np.sqrt(s) + self.eps)


class Adam:
    """m ← β1·m + (1 − β1)·g;   v ← β2·v + (1 − β2)·g²;   t ← t + 1
    m̂ = m/(1 − β1ᵗ),  v̂ = v/(1 − β2ᵗ);   θ ← θ − η·m̂ / (√v̂ + ε)   (ε sumado tras la raíz y tras corregir v)."""

    def __init__(self, lr=0.001, betas=(0.9, 0.999), eps=1e-8):
        self.lr, (self.b1, self.b2), self.eps, self.t, self.m, self.v = lr, betas, eps, 0, None, None

    def step(self, P, G):
        if self.m is None:
            self.m, self.v = [np.zeros_like(g) for g in G], [np.zeros_like(g) for g in G]
        self.t += 1
        c1, c2 = 1 - self.b1 ** self.t, 1 - self.b2 ** self.t
        for p, g, m, v in zip(P, G, self.m, self.v):
            m *= self.b1
            m += (1 - self.b1) * g
            v *= self.b2
            v += (1 - self.b2) * g * g
            p -= self.lr * (m / c1) / (np.sqrt(v / c2) + self.eps)
