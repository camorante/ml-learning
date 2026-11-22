"""Capa de embedding desde cero en NumPy, hacia delante y hacia atrás, y el modelo del tema:
código visto → embedding → capa lineal → softmax sin el propio código → entropía cruzada."""
import numpy as np


class Embedding:
    """Una tabla E de V filas × d columnas. Mirar el índice i es devolver la fila i (= one-hot(i) @ E)."""

    def __init__(self, E):
        self.E = E

    def adelante(self, idx):
        self.idx = idx
        return self.E[idx]                              # (n, d): una fila por índice, sin multiplicar nada

    def atras(self, dout):
        """Solo las filas que se han mirado reciben gradiente; si una fila sale dos veces, se suman."""
        dE = np.zeros_like(self.E)
        np.add.at(dE, self.idx, dout)                   # dE[idx[k]] += dout[k], también con índices repetidos
        return dE


class EmbeddingBolsa:
    """Una «bolsa» de índices por ejemplo (los códigos de una visita) → la media de sus filas.
    bolsas: lista de listas de índices. Como torch.nn.EmbeddingBag(mode="mean")."""

    def __init__(self, E):
        self.E = E

    def adelante(self, bolsas):
        self.bolsas = bolsas
        return np.stack([self.E[b].mean(0) for b in bolsas])

    def atras(self, dout):
        dE = np.zeros_like(self.E)
        for b, g in zip(self.bolsas, dout):
            np.add.at(dE, b, g / len(b))                # cada fila de la bolsa recibe 1/len del gradiente
        return dE


def softmax_sin(z, a):
    """Softmax por filas con el logit del propio código a −∞ (un código no se acompaña a sí mismo)."""
    z = z.copy(); z[np.arange(len(a)), a] = -np.inf
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


class Modelo:
    """z = E[a] @ W.T + b; pérdida = entropía cruzada media de «el otro código es y»."""

    def __init__(self, E, W, b):
        self.emb, self.W, self.b = Embedding(E.copy()), W.copy(), b.copy()

    def perdida(self, a, y):
        self.a, self.y = a, y
        self.h = self.emb.adelante(a)                   # (n, d)
        self.p = softmax_sin(self.h @ self.W.T + self.b, a)
        return -np.mean(np.log(self.p[np.arange(len(y)), y]))

    def atras(self):
        n = len(self.y)
        dz = self.p.copy(); dz[np.arange(n), self.y] -= 1; dz /= n     # softmax + entropía cruzada (tema 4)
        dW, db = dz.T @ self.h, dz.sum(0)
        dE = self.emb.atras(dz @ self.W)                                 # tema 5: la regla de la cadena
        return dE, dW, db

    def paso(self, a, y, lr):
        """Un paso de descenso de gradiente (sin Adam): actualiza las filas miradas, W y b."""
        L = self.perdida(a, y); dE, dW, db = self.atras()
        self.emb.E -= lr * dE; self.W -= lr * dW; self.b -= lr * db
        return L
