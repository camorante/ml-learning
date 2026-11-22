"""Autoencoder desde cero en NumPy, hacia delante y hacia atrás:
x → lineal → tanh → lineal (código z) → lineal → tanh → lineal → x̂ ;  pérdida = media de (x̂ − x)²"""
import numpy as np


class Lineal:
    def __init__(self, W, b):                    # W: (salidas, entradas), como nn.Linear
        self.W, self.b = W.copy(), b.copy()

    def adelante(self, x):
        self.x = x
        return x @ self.W.T + self.b

    def atras(self, g):
        self.dW, self.db = g.T @ self.x, g.sum(0)
        return g @ self.W


class Tanh:
    def adelante(self, x):
        self.y = np.tanh(x)
        return self.y

    def atras(self, g):
        return g * (1 - self.y ** 2)             # tanh'(a) = 1 − tanh(a)²


class Autoencoder:
    """capas: [(W1, b1), (W2, b2), (W3, b3), (W4, b4)]; tanh tras la 1.ª y la 3.ª. El código es la salida de la 2.ª."""

    def __init__(self, pesos):
        L = [Lineal(W, b) for W, b in pesos]
        self.cod = [L[0], Tanh(), L[1]]
        self.dec = [L[2], Tanh(), L[3]]
        self.lineales = L

    def codigo(self, x):
        for c in self.cod:
            x = c.adelante(x)
        return x

    def adelante(self, x):
        self.z = self.codigo(x)
        h = self.z
        for c in self.dec:
            h = c.adelante(h)
        return h

    def perdida(self, x, objetivo=None):
        """Error cuadrático medio sobre las n × 64 casillas. Sin objetivo, el objetivo es la propia entrada."""
        self.objetivo = x if objetivo is None else objetivo
        self.xr = self.adelante(x)
        return np.mean((self.xr - self.objetivo) ** 2)

    def atras(self):
        g = 2 * (self.xr - self.objetivo) / self.xr.size     # ∂pérdida/∂x̂
        for c in reversed(self.cod + self.dec):
            g = c.atras(g)
        return [(c.dW, c.db) for c in self.lineales], g       # gradientes de cada capa y ∂pérdida/∂x

    def paso(self, x, lr):
        perd = self.perdida(x)
        grads, _ = self.atras()
        for c, (dW, db) in zip(self.lineales, grads):
            c.W -= lr * dW; c.b -= lr * db
        return perd
