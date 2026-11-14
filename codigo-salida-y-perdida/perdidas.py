"""Activaciones de salida, pérdidas y sus gradientes, desde cero con NumPy (media sobre el lote)."""
import numpy as np


def logsumexp(z, axis=-1):
    """log Σ e^z sin desbordar: se resta el máximo antes de exponenciar."""
    z = np.asarray(z, float)
    m = np.max(z, axis=axis, keepdims=True)
    return np.squeeze(m, axis) + np.log(np.sum(np.exp(z - m), axis=axis))


def softmax(z, T=1.0, axis=-1):
    """e^(z/T) normalizado; restar el máximo no cambia el resultado y evita el desbordamiento."""
    z = np.asarray(z, float) / T
    e = np.exp(z - np.max(z, axis=axis, keepdims=True))
    return e / np.sum(e, axis=axis, keepdims=True)


def log_softmax(z, axis=-1):
    z = np.asarray(z, float)
    return z - np.expand_dims(logsumexp(z, axis), axis)


def sigmoide(z):
    z = np.asarray(z, float); e = np.exp(-np.abs(z))
    return np.where(z >= 0, 1 / (1 + e), e / (1 + e))


# ---------- regresión: ŷ son números sin activación ----------
def mse(yhat, y):
    return np.mean((yhat - y) ** 2)

def d_mse(yhat, y):
    return 2 * (yhat - y) / np.size(y)

def mae(yhat, y):
    return np.mean(np.abs(yhat - y))

def d_mae(yhat, y):
    return np.sign(yhat - y) / np.size(y)

def huber(yhat, y, delta=1.0):
    """Cuadrática si |e| ≤ δ, lineal después: e²/2 o δ(|e| − δ/2)."""
    e = np.abs(yhat - y)
    return np.mean(np.where(e <= delta, 0.5 * e ** 2, delta * (e - 0.5 * delta)))

def d_huber(yhat, y, delta=1.0):
    return np.clip(yhat - y, -delta, delta) / np.size(y)


# ---------- clasificación: siempre desde los logits ----------
def bce_logits(z, y, pos_weight=1.0):
    """−[w·y·log σ(z) + (1 − y)·log(1 − σ(z))] con log σ(z) = −softplus(−z), sin formar σ(z)."""
    z = np.asarray(z, float)
    sp = lambda t: np.maximum(t, 0) + np.log1p(np.exp(-np.abs(t)))      # softplus estable
    return np.mean(pos_weight * y * sp(-z) + (1 - y) * sp(z))

def d_bce_logits(z, y, pos_weight=1.0):
    p = sigmoide(z)
    return (pos_weight * y * (p - 1) + (1 - y) * p) / np.size(y)        # con w = 1: (p − y)/n

def ce_logits(Z, c):
    """Entropía cruzada media: −log softmax(Z)[i, c_i] = logsumexp(Z_i) − Z_i[c_i]."""
    Z = np.asarray(Z, float); c = np.asarray(c)
    return np.mean(logsumexp(Z, axis=1) - Z[np.arange(len(c)), c])

def d_ce_logits(Z, c):
    P = softmax(Z, axis=1); P[np.arange(len(c)), c] -= 1                # p − y (y one-hot)
    return P / len(c)


def mse_sobre_p(z, y):
    """Error cuadrático aplicado a la probabilidad (lo que NO conviene hacer) y su gradiente en z."""
    p = sigmoide(z)
    return np.mean((p - y) ** 2), 2 * (p - y) * p * (1 - p) / np.size(y)
