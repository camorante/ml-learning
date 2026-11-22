"""Una celda RNN (tanh) y una celda LSTM desde cero en NumPy, hacia delante y hacia atrás a lo largo de toda la
secuencia (retropropagación en el tiempo). Mismas convenciones que PyTorch: X es (n, T, D), los pesos de entrada
son (filas, D), los recurrentes (filas, H), y la LSTM apila sus cuatro bloques en el orden i, f, g, o."""
import numpy as np


def sigmoide(z):
    return 1 / (1 + np.exp(-z))


# ---------------------------------------------------------------- RNN: h_t = tanh(W x_t + U h_{t-1} + b)
def rnn_adelante(X, W, U, b, h0):
    n, T, _ = X.shape
    Hs = np.zeros((n, T + 1, U.shape[0])); Hs[:, 0] = h0           # Hs[:, t + 1] = h_t; Hs[:, 0] = h_0
    for t in range(T):
        Hs[:, t + 1] = np.tanh(X[:, t] @ W.T + Hs[:, t] @ U.T + b)
    return Hs[:, 1:], (X, W, U, Hs)


def rnn_atras(dH, cache):
    """dH: ∂L/∂h_t que llega de arriba en cada paso (n, T, H). Devuelve ∂L/∂X, ∂W, ∂U, ∂b, ∂h_0."""
    X, W, U, Hs = cache
    dX, dW, dU, db = np.zeros_like(X), np.zeros_like(W), np.zeros_like(U), np.zeros(U.shape[0])
    dh = np.zeros_like(Hs[:, 0])                                    # lo que llega del paso siguiente
    for t in reversed(range(X.shape[1])):
        dh = dh + dH[:, t]
        da = dh * (1 - Hs[:, t + 1] ** 2)                           # a través de tanh: tanh' = 1 − h²
        dW += da.T @ X[:, t]; dU += da.T @ Hs[:, t]; db += da.sum(0)
        dX[:, t] = da @ W
        dh = da @ U                                                 # hacia h_{t-1}: el producto que se desvanece
    return dX, dW, dU, db, dh


# ---------------------------------------------------------------- LSTM
def lstm_adelante(X, W, U, b, h0, c0):
    """z = W x_t + U h_{t-1} + b, partido en cuatro bloques de H: i, f, o con sigmoide y g con tanh.
    c_t = f ⊙ c_{t-1} + i ⊙ g ;  h_t = o ⊙ tanh(c_t)."""
    n, T, _ = X.shape; H = U.shape[1]
    Hs = np.zeros((n, T + 1, H)); Cs = np.zeros((n, T + 1, H)); Hs[:, 0], Cs[:, 0] = h0, c0
    P = np.zeros((n, T, 4, H))                                      # puertas i, f, g, o de cada paso
    for t in range(T):
        z = (X[:, t] @ W.T + Hs[:, t] @ U.T + b).reshape(n, 4, H)
        i, f, g, o = sigmoide(z[:, 0]), sigmoide(z[:, 1]), np.tanh(z[:, 2]), sigmoide(z[:, 3])
        Cs[:, t + 1] = f * Cs[:, t] + i * g
        Hs[:, t + 1] = o * np.tanh(Cs[:, t + 1])
        P[:, t] = np.stack([i, f, g, o], 1)
    return Hs[:, 1:], Cs[:, 1:], (X, W, U, Hs, Cs, P)


def lstm_atras(dH, cache, dC=None):
    """dH (y opcionalmente dC): gradientes que llegan de arriba sobre h_t (y c_t). Devuelve ∂X, ∂W, ∂U, ∂b, ∂h_0, ∂c_0."""
    X, W, U, Hs, Cs, P = cache
    n, T, _ = X.shape; H = U.shape[1]
    dX, dW, dU, db = np.zeros_like(X), np.zeros_like(W), np.zeros_like(U), np.zeros(4 * H)
    dh, dc = np.zeros((n, H)), np.zeros((n, H))
    for t in reversed(range(T)):
        i, f, g, o = P[:, t, 0], P[:, t, 1], P[:, t, 2], P[:, t, 3]
        tc = np.tanh(Cs[:, t + 1])
        dh = dh + dH[:, t]
        dc = dc + dh * o * (1 - tc ** 2) + (0 if dC is None else dC[:, t])
        dz = np.concatenate([dc * g * i * (1 - i),                  # puerta de entrada
                             dc * Cs[:, t] * f * (1 - f),           # puerta de olvido
                             dc * i * (1 - g ** 2),                 # candidato
                             dh * tc * o * (1 - o)], 1)             # puerta de salida
        dW += dz.T @ X[:, t]; dU += dz.T @ Hs[:, t]; db += dz.sum(0)
        dX[:, t] = dz @ W
        dh = dz @ U
        dc = dc * f                                                 # la cinta: solo se multiplica por f
    return dX, dW, dU, db, dh, dc
