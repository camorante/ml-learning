import numpy as np
from red import MLP, Densa, n_param


def sierra(L):
    """L capas ocultas de 2 neuronas ReLU. Cada capa calcula t(u) = 2·relu(u) − 4·relu(u − ½):
    dobla [0, 1] por la mitad. La combinación 2·h1 − 4·h2 va en los pesos de la capa siguiente."""
    capas = [Densa([[1.0], [1.0]], [0.0, -0.5], "relu")]
    capas += [Densa([[2.0, -4.0], [2.0, -4.0]], [0.0, -0.5], "relu") for _ in range(L - 1)]
    return MLP(capas + [Densa([[2.0, -4.0]], [0.0], "identidad")])


def trozos(f, n=2 ** 15):
    u = np.linspace(0, 1, n + 1)[:, None]
    s = np.diff(f(u)[:, 0]) * n                     # pendiente en cada intervalo de la rejilla
    return 1 + int(np.sum(np.abs(np.diff(s)) > 1e-6))


print(" L   trozos   parámetros (profunda)   una capa: neuronas   parámetros")
for L in (1, 2, 3, 4, 6, 8, 10):
    red = sierra(L)
    k = trozos(red.forward)
    assert red.n_param == n_param(red.tamanos) == 6 * L + 1
    m = k - 1                                        # una capa ReLU con m neuronas da como mucho m + 1 trozos
    print(f"{L:2d}   {k:6d}   {red.n_param:10d}              {m:10d}           {n_param([1, m, 1]):6d}")
print("valores en u = 0, 1/8, 1/4, 3/8, 1/2 con L = 3:", sierra(3).forward(np.array([[0], [1 / 8], [1 / 4], [3 / 8], [1 / 2]]))[:, 0])
