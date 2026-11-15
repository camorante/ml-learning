"""Normalización por lotes (Ioffe y Szegedy, 2015) desde cero en NumPy, con las convenciones de nn.BatchNorm1d:
en entrenamiento normaliza con la media y la varianza SESGADA del lote (÷ n) y actualiza las estadísticas móviles con
momento 0.1, guardando la varianza INSESGADA (÷ (n − 1)); en evaluación usa solo las estadísticas móviles."""
import numpy as np


class BatchNorm1d:
    def __init__(self, d, momentum=0.1, eps=1e-5):
        self.gamma, self.beta = np.ones(d), np.zeros(d)              # escala y desplazamiento aprendibles
        self.run_mean, self.run_var = np.zeros(d), np.ones(d)         # estadísticas móviles (lo que usa la evaluación)
        self.mom, self.eps, self.train = momentum, eps, True

    def forward(self, x):
        if self.train:
            n = x.shape[0]
            mu, var = x.mean(0), x.var(0)                             # varianza sesgada: ÷ n
            self.run_mean = (1 - self.mom) * self.run_mean + self.mom * mu
            self.run_var = (1 - self.mom) * self.run_var + self.mom * var * n / (n - 1)   # insesgada: ÷ (n − 1)
        else:
            mu, var = self.run_mean, self.run_var
        self.xh = (x - mu) / np.sqrt(var + self.eps)
        self.inv = 1 / np.sqrt(var + self.eps)
        return self.gamma * self.xh + self.beta

    def backward(self, dy):
        """Gradientes respecto a la entrada, γ y β (modo entrenamiento: la media y la varianza dependen de x)."""
        self.dgamma, self.dbeta = (dy * self.xh).sum(0), dy.sum(0)
        dxh = dy * self.gamma
        if not self.train:
            return dxh * self.inv
        n = dy.shape[0]
        return self.inv / n * (n * dxh - dxh.sum(0) - self.xh * (dxh * self.xh).sum(0))
