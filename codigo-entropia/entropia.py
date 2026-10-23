"""Entropía, entropía cruzada, KL, códigos de Huffman, pérdidas de clasificación estables,
ganancia de información e información mutua, desde cero (numpy)."""
import heapq
import math
import numpy as np


def _log(x, base):
    return np.log(x) / (1.0 if base is None or base == math.e else math.log(base))


def _plogq(p, q, base):
    """p · log q con el convenio 0 · log 0 = 0 (y p > 0, q = 0 → −inf)."""
    p, q = np.broadcast_arrays(np.asarray(p, float), np.asarray(q, float))
    out = np.zeros(p.shape)
    m = p > 0
    with np.errstate(divide="ignore"):
        out[m] = p[m] * _log(q[m], base)
    return out


def sorpresa(p, base=2):
    """Información de un suceso de probabilidad p: −log p (bits si base = 2)."""
    with np.errstate(divide="ignore"):
        return -_log(np.asarray(p, float), base)


def entropia(p, base=2):
    """H(p) = −Σ p log p. Normaliza p si no suma 1 (como scipy.stats.entropy)."""
    p = np.asarray(p, float); p = p / p.sum()
    return float(-_plogq(p, p, base).sum()) + 0.0                      # + 0.0: sin «−0.0»


def entropia_cruzada(p, q, base=2):
    """H(p, q) = −Σ p log q: bits por mensaje si los datos vienen de p y el código es el de q."""
    p = np.asarray(p, float); q = np.asarray(q, float)
    return float(-_plogq(p / p.sum(), q / q.sum(), base).sum())


def kl(p, q, base=2):
    """D(p ‖ q) = Σ p log(p / q) = H(p, q) − H(p) ≥ 0. Infinita si q = 0 donde p > 0."""
    p = np.asarray(p, float); q = np.asarray(q, float); p, q = p / p.sum(), q / q.sum()
    return float((_plogq(p, p, base) - _plogq(p, q, base)).sum())


def huffman(p):
    """Código de Huffman binario: lista de palabras ('0'/'1') en el orden de p.
    Junta siempre los dos nodos menos probables; desempata por orden de llegada."""
    if len(p) == 1:
        return ["0"]
    cola = [(float(pi), i, (i,)) for i, pi in enumerate(p)]
    heapq.heapify(cola)
    codigo = [""] * len(p); n = len(p)
    while len(cola) > 1:
        p0, _, a = heapq.heappop(cola); p1, _, b = heapq.heappop(cola)
        for i in a: codigo[i] = "0" + codigo[i]
        for i in b: codigo[i] = "1" + codigo[i]
        heapq.heappush(cola, (p0 + p1, n, a + b)); n += 1
    return codigo


def longitud_media(p, codigo):
    p = np.asarray(p, float)
    return float(np.dot(p / p.sum(), [len(c) for c in codigo]))


def kraft(codigo):
    """Σ 2^(−longitud): ≤ 1 para cualquier código prefijo; = 1 si el árbol está lleno."""
    return float(sum(2.0 ** -len(c) for c in codigo))


# ---------------- pérdidas de clasificación ----------------
def log_loss_binaria(y, p, eps=np.finfo(float).eps):
    """Media de −[y log p + (1 − y) log(1 − p)], en nats, recortando p a [eps, 1 − eps]."""
    y = np.asarray(y, float); p = np.clip(np.asarray(p, float), eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log1p(-p)))


def bce_con_logits(y, z):
    """La misma pérdida a partir del logit z (p = σ(z)), sin calcular p:
    max(z, 0) − z·y + log(1 + e^(−|z|)). No se desborda ni toma log 0."""
    y = np.asarray(y, float); z = np.asarray(z, float)
    return float(np.mean(np.maximum(z, 0) - z * y + np.log1p(np.exp(-np.abs(z)))))


def logsumexp(z, axis=-1):
    z = np.asarray(z, float); m = np.max(z, axis=axis, keepdims=True)
    return np.squeeze(m, axis) + np.log(np.sum(np.exp(z - m), axis=axis))


def log_softmax(z, axis=-1):
    z = np.asarray(z, float)
    return z - np.expand_dims(logsumexp(z, axis), axis)


def softmax(z, axis=-1):
    return np.exp(log_softmax(z, axis))


def entropia_cruzada_logits(y, z):
    """Entropía cruzada categórica media (nats). y: índices de clase o matriz de
    probabilidades objetivo (one-hot o suavizadas); z: logits (n × K)."""
    lp = log_softmax(z)
    y = np.asarray(y)
    if y.ndim == 1:
        return float(-np.mean(lp[np.arange(len(y)), y.astype(int)]))
    return float(-np.mean(np.sum(y * lp, axis=1)))


def gradiente_logits(y, z):
    """Derivada de la entropía cruzada media respecto a los logits: (softmax(z) − y) / n."""
    z = np.asarray(z, float); y = np.asarray(y)
    if y.ndim == 1:
        y = np.eye(z.shape[1])[y.astype(int)]
    return (softmax(z) - y) / len(z)


def suavizar_etiquetas(y, K, eps):
    """Label smoothing: (1 − ε)·one-hot + ε/K."""
    return (1 - eps) * np.eye(K)[np.asarray(y, int)] + eps / K


# ---------------- árboles e información mutua ----------------
def entropia_etiquetas(y, base=2):
    _, c = np.unique(np.asarray(y), return_counts=True)
    return entropia(c, base)


def ganancia_informacion(y, izquierda, base=2):
    """H(padre) − media ponderada de H(hijos) al partir con la máscara booleana 'izquierda'."""
    y = np.asarray(y); m = np.asarray(izquierda, bool); n = len(y)
    h = lambda v: entropia_etiquetas(v, base) if len(v) else 0.0
    return h(y) - m.sum() / n * h(y[m]) - (~m).sum() / n * h(y[~m])


def informacion_mutua(x, y, base=None):
    """I(X; Y) = Σ p(x, y) log p(x, y) / (p(x) p(y)) con frecuencias (nats por defecto, como sklearn)."""
    _, xi = np.unique(np.asarray(x), return_inverse=True); _, yi = np.unique(np.asarray(y), return_inverse=True)
    t = np.zeros((xi.max() + 1, yi.max() + 1)); np.add.at(t, (xi, yi), 1); t /= t.sum()
    px, py = t.sum(1, keepdims=True), t.sum(0, keepdims=True)
    return float(_plogq(t, t / (px * py) + (t == 0), base).sum())


def entropia_normal(sigma, base=None):
    """Entropía diferencial de N(μ, σ²): ½ log(2πe σ²). Puede ser negativa."""
    return float(0.5 * _log(2 * math.pi * math.e * np.asarray(sigma, float) ** 2, base))
