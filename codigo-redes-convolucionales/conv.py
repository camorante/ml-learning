"""Convolución 1D y 2D (con paso, relleno y canales) y max-pooling 1D desde cero en NumPy, hacia delante y hacia
atrás. Como PyTorch, es una correlación cruzada: el filtro NO se da la vuelta."""
import numpy as np


def ventanas1d(L, k, s):
    """Índices de las ventanas: fila i = posiciones s·i, s·i+1, …, s·i+k−1 de la entrada (ya rellenada)."""
    n_out = (L - k) // s + 1
    return s * np.arange(n_out)[:, None] + np.arange(k)              # (n_out, k)


def conv1d(x, W, b, s=1, p=0):
    """x: (n, C, L) · W: (F, C, k) · b: (F,) → y: (n, F, ⌊(L + 2p − k)/s⌋ + 1)."""
    xp = np.pad(x, ((0, 0), (0, 0), (p, p)))
    idx = ventanas1d(xp.shape[2], W.shape[2], s)
    trozos = xp[:, :, idx]                                            # (n, C, n_out, k)
    y = np.einsum("ncok,fck->nfo", trozos, W) + b[None, :, None]
    return y, (x.shape, xp.shape, idx, trozos, W, p)


def conv1d_atras(dy, cache):
    """dy = ∂pérdida/∂y. Devuelve ∂/∂x, ∂/∂W, ∂/∂b. Cada peso se usa en todas las posiciones: sus gradientes se suman."""
    xshape, xpshape, idx, trozos, W, p = cache
    dW = np.einsum("nfo,ncok->fck", dy, trozos)                       # correlación de la entrada con dy
    db = dy.sum((0, 2))
    dtrozos = np.einsum("nfo,fck->ncok", dy, W)
    dxp = np.zeros(xpshape)
    np.add.at(dxp, (slice(None), slice(None), idx), dtrozos)          # cada lectura recibe de todas sus ventanas
    return dxp[:, :, p:xpshape[2] - p], dW, db


def conv2d(x, W, b, s=1, p=0):
    """x: (n, C, H, A) · W: (F, C, kh, ka) → y: (n, F, H', A'), la misma fórmula de tamaño en cada eje."""
    xp = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)))
    ih, ia = ventanas1d(xp.shape[2], W.shape[2], s), ventanas1d(xp.shape[3], W.shape[3], s)
    trozos = xp[:, :, ih[:, None, :, None], ia[None, :, None, :]]     # (n, C, H', A', kh, ka)
    y = np.einsum("ncijab,fcab->nfij", trozos, W) + b[None, :, None, None]
    return y, (xp.shape, ih, ia, trozos, W, p)


def conv2d_atras(dy, cache):
    xpshape, ih, ia, trozos, W, p = cache
    dW = np.einsum("nfij,ncijab->fcab", dy, trozos)
    db = dy.sum((0, 2, 3))
    dxp = np.zeros(xpshape)
    np.add.at(dxp, (slice(None), slice(None), ih[:, None, :, None], ia[None, :, None, :]),
              np.einsum("nfij,fcab->ncijab", dy, W))
    return dxp[:, :, p:xpshape[2] - p, p:xpshape[3] - p], dW, db


def maxpool1d(x, k=2, s=None):
    """Máximo de cada ventana de k lecturas (por defecto, sin solaparse: paso s = k)."""
    s = s or k
    idx = ventanas1d(x.shape[2], k, s)
    trozos = x[:, :, idx]                                             # (n, C, n_out, k)
    arg = trozos.argmax(-1)
    return np.take_along_axis(trozos, arg[..., None], -1)[..., 0], (x.shape, idx, arg)


def maxpool1d_atras(dy, cache):
    """El gradiente solo pasa por la lectura que ganó en cada ventana; las demás reciben 0."""
    xshape, idx, arg = cache
    dx = np.zeros(xshape)
    n, C, n_out = dy.shape
    gan = idx[np.arange(n_out)[None, None, :], arg]                   # posición ganadora de cada ventana
    np.add.at(dx, (np.arange(n)[:, None, None], np.arange(C)[None, :, None], gan), dy)
    return dx
