"""Retropropagación desde cero en NumPy: una red multicapa (MLP) que guarda lo que necesita en la pasada
hacia delante y reparte el gradiente capa a capa hacia atrás. Misma convención que el tema 3:
W tiene forma (m, d), una fila por neurona; los ejemplos van en filas, Z = X Wᵀ + b."""
import numpy as np


def sigmoide(z):
    z = np.asarray(z, float); e = np.exp(-np.abs(z))
    return np.where(z >= 0, 1 / (1 + e), e / (1 + e))


# activación y su derivada local, escrita con lo que ya se guardó (z y h = φ(z))
ACT = {"sigmoide": (sigmoide, lambda z, h: h * (1 - h)),
       "tanh": (np.tanh, lambda z, h: 1 - h ** 2),
       "relu": (lambda z: np.maximum(z, 0.0), lambda z, h: (z > 0).astype(float)),
       "identidad": (lambda z: np.asarray(z, float), lambda z, h: np.ones_like(z))}


def softplus(t):
    return np.maximum(t, 0) + np.log1p(np.exp(-np.abs(t)))


def softmax(Z):
    E = np.exp(Z - Z.max(1, keepdims=True)); return E / E.sum(1, keepdims=True)


# pérdida media sobre el lote, calculada desde la última z, y su gradiente ∂L/∂Z de la última capa
PERDIDA = {
    "bce": (lambda Z, y: np.mean(y * softplus(-Z[:, 0]) + (1 - y) * softplus(Z[:, 0])),
            lambda Z, y: (sigmoide(Z[:, 0]) - y)[:, None] / len(y)),                     # (p − y)/n
    "ce": (lambda Z, y: np.mean(np.log(np.exp(Z - Z.max(1, keepdims=True)).sum(1)) + Z.max(1) - Z[np.arange(len(y)), y]),
           lambda Z, y: (softmax(Z) - np.eye(Z.shape[1])[y]) / len(y)),                   # (p − y)/n
    "mse": (lambda Z, y: np.mean((Z[:, 0] - y) ** 2),
            lambda Z, y: 2 * (Z[:, 0] - y)[:, None] / len(y)),
}


class Densa:
    def __init__(self, W, b, act="sigmoide"):
        self.W, self.b, self.act = np.atleast_2d(np.asarray(W, float)), np.asarray(b, float).ravel(), act

    def adelante(self, X):
        self.X = X                                   # se guarda: la señal que entra (para ∇W)
        self.Z = X @ self.W.T + self.b
        self.H = ACT[self.act][0](self.Z)            # se guarda: z y h dan la pendiente local
        return self.H

    def atras(self, dH):
        """Recibe ∂L/∂H (n, m). Deja ∂L/∂W y ∂L/∂b y devuelve ∂L/∂X para la capa anterior."""
        dZ = dH * ACT[self.act][1](self.Z, self.H)   # δ = (error que vuelve) ⊙ φ'(z)
        self.dW = dZ.T @ self.X                      # (señal que entra) × (error que vuelve), sumado en el lote
        self.db = dZ.sum(0)
        return dZ @ self.W                           # Wᵀ δ, en filas: los caminos que se separan se suman


class MLP:
    def __init__(self, capas, perdida="bce"):
        self.capas, self.perdida = list(capas), perdida
        assert self.capas[-1].act == "identidad", "la última capa da logits; la pérdida aplica σ o softmax"

    def adelante(self, X):
        H = np.atleast_2d(np.asarray(X, float))
        for c in self.capas:
            H = c.adelante(H)
        return H

    def perdida_media(self, X, y):
        return PERDIDA[self.perdida][0](self.adelante(X), y)

    def gradientes(self, X, y):
        """Una pasada hacia delante y una hacia atrás: la pérdida y [(∂L/∂W, ∂L/∂b) de cada capa]."""
        Z = self.adelante(X)
        L = PERDIDA[self.perdida][0](Z, y)
        d = PERDIDA[self.perdida][1](Z, y)           # ∂L/∂z de la salida: (p − y)/n
        for c in reversed(self.capas):
            d = c.atras(d)
        return L, [(c.dW, c.db) for c in self.capas]

    # los parámetros como un solo vector (para diferencias finitas y comparaciones)
    def vector(self):
        return np.concatenate([np.r_[c.W.ravel(), c.b] for c in self.capas])

    def cargar(self, v):
        i = 0
        for c in self.capas:
            m, d = c.W.shape
            c.W = v[i:i + m * d].reshape(m, d).copy(); i += m * d
            c.b = v[i:i + m].copy(); i += m

    def paso(self, X, y, eta):
        """Un paso de descenso de gradiente: θ ← θ − η ∇L."""
        L, g = self.gradientes(X, y)
        for c, (dW, db) in zip(self.capas, g):
            c.W = c.W - eta * dW; c.b = c.b - eta * db
        return L


def gradiente_vector(red, X, y):
    _, g = red.gradientes(X, y)
    return np.concatenate([np.r_[dW.ravel(), db] for dW, db in g])


def diferencias_centrales(red, X, y, h=1e-6):
    """2 pasadas hacia delante por parámetro: (L(θ + h eᵢ) − L(θ − h eᵢ)) / 2h."""
    v0 = red.vector(); g = np.zeros_like(v0)
    for i in range(len(v0)):
        e = np.zeros_like(v0); e[i] = h
        red.cargar(v0 + e); Lp = red.perdida_media(X, y)
        red.cargar(v0 - e); Lm = red.perdida_media(X, y)
        g[i] = (Lp - Lm) / (2 * h)
    red.cargar(v0)
    return g


def error_relativo(a, b):
    return np.linalg.norm(a - b) / max(np.linalg.norm(a), np.linalg.norm(b), 1e-300)
