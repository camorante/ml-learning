"""Una red multicapa (MLP) desde cero en numpy: capas densas = producto de matrices + sesgo + activación.
Solo la pasada hacia delante; el entrenamiento (retropropagación) es el tema 5."""
import numpy as np


def sigmoide(z):
    z = np.asarray(z, float)
    e = np.exp(-np.abs(z))
    return np.where(z >= 0, 1 / (1 + e), e / (1 + e))


ACT = {"sigmoide": sigmoide, "tanh": np.tanh, "relu": lambda z: np.maximum(z, 0.0),
       "identidad": lambda z: np.asarray(z, float)}


class Densa:
    """Una capa: m neuronas que miran las mismas d entradas. W tiene forma (m, d): una fila por neurona."""

    def __init__(self, W, b, act="sigmoide"):
        self.W, self.b, self.act = np.atleast_2d(np.asarray(W, float)), np.asarray(b, float).ravel(), act
        assert self.W.shape[0] == self.b.shape[0], "una fila de W y un sesgo por neurona"

    def __call__(self, X):
        """X: (n, d) → Z = X Wᵀ + b: (n, m) → H = φ(Z). Todas las neuronas y todos los ejemplos a la vez."""
        Z = np.asarray(X, float) @ self.W.T + self.b
        return Z, ACT[self.act](Z)

    @property
    def n_param(self):
        return self.W.size + self.b.size                # m·d pesos + m sesgos = m·(d + 1)


class MLP:
    def __init__(self, capas):
        self.capas = list(capas)
        for a, b in zip(self.capas, self.capas[1:]):
            assert a.W.shape[0] == b.W.shape[1], "las salidas de una capa son las entradas de la siguiente"

    def forward(self, X, todo=False):
        """Pasada hacia delante. Con todo=True devuelve también (Z, H) de cada capa."""
        H, pasos = np.atleast_2d(np.asarray(X, float)), []
        for c in self.capas:
            Z, H = c(H)
            pasos.append((Z, H))
        return (H, pasos) if todo else H

    def predict_proba(self, X):
        p = self.forward(X)[:, 0]
        return np.c_[1 - p, p]                          # mismo formato que scikit-learn: [P(0), P(1)]

    def predict(self, X):
        return (self.forward(X)[:, 0] > 0.5).astype(int)

    @property
    def n_param(self):
        return sum(c.n_param for c in self.capas)

    @property
    def tamanos(self):
        return [self.capas[0].W.shape[1]] + [c.W.shape[0] for c in self.capas]

    @classmethod
    def desde_sklearn(cls, m):
        """MLPClassifier guarda coefs_[l] con forma (d, m): la traspuesta de nuestra W."""
        acts = {"logistic": "sigmoide", "tanh": "tanh", "relu": "relu", "identity": "identidad"}
        L = len(m.coefs_)
        return cls(Densa(W.T, b, acts[m.activation] if l < L - 1 else "sigmoide")
                   for l, (W, b) in enumerate(zip(m.coefs_, m.intercepts_)))


def n_param(tamanos):
    """Parámetros de una red con capas de tamaños [d, m1, m2, …, salida]: Σ (entradas + 1) · neuronas."""
    return sum((a + 1) * b for a, b in zip(tamanos, tamanos[1:]))
