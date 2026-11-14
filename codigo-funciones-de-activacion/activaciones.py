"""Funciones de activación y sus derivadas, desde cero con NumPy."""
import numpy as np
from scipy.special import erf

def escalon(z):                       # el perceptrón: 1 si z > 0 (derivada 0 en casi todas partes)
    return (np.asarray(z) > 0).astype(float)

def sigmoide(z):
    """1 / (1 + e^-z) sin desbordar: para z < 0 se usa e^z / (1 + e^z)."""
    z = np.asarray(z, float); e = np.exp(-np.abs(z))
    return np.where(z >= 0, 1 / (1 + e), e / (1 + e))

def d_sigmoide(z):
    s = sigmoide(z); return s * (1 - s)              # como mucho 0,25, en z = 0

def tanh(z):
    return np.tanh(z)                                # = 2·sigmoide(2z) − 1, centrada en 0

def d_tanh(z):
    return 1 - np.tanh(z) ** 2                       # como mucho 1, en z = 0

def relu(z):
    return np.maximum(np.asarray(z, float), 0)

def d_relu(z):
    return (np.asarray(z) > 0).astype(float)         # en z = 0 se toma 0, como PyTorch

def leaky_relu(z, a=0.01):
    z = np.asarray(z, float); return np.where(z > 0, z, a * z)

def d_leaky_relu(z, a=0.01):
    return np.where(np.asarray(z) > 0, 1.0, a)

def gelu(z):
    """z · Φ(z), con Φ la función de distribución de la normal estándar."""
    z = np.asarray(z, float); return z * 0.5 * (1 + erf(z / np.sqrt(2)))

def d_gelu(z):
    z = np.asarray(z, float)
    return 0.5 * (1 + erf(z / np.sqrt(2))) + z * np.exp(-z * z / 2) / np.sqrt(2 * np.pi)

def gelu_tanh(z):
    """La aproximación de Hendrycks y Gimpel (2016) que usan GPT-2 y BERT."""
    z = np.asarray(z, float); return 0.5 * z * (1 + np.tanh(np.sqrt(2 / np.pi) * (z + 0.044715 * z ** 3)))

ACTIVACIONES = {"sigmoide": (sigmoide, d_sigmoide), "tanh": (tanh, d_tanh), "relu": (relu, d_relu),
                "leaky_relu": (leaky_relu, d_leaky_relu), "gelu": (gelu, d_gelu)}
