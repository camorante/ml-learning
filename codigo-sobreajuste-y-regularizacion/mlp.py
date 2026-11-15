"""Una red densa desde cero en NumPy: capas tanh, dropout invertido tras cada capa oculta, salida lineal y error
cuadrático medio. Pesos con la forma de nn.Linear (salidas × entradas). Optimizadores Adam con L2 acoplado
(torch.optim.Adam(weight_decay=λ)) y AdamW con decaimiento desacoplado (torch.optim.AdamW)."""
import numpy as np


def xavier_uniform(n_out, n_in, rng, gain=1.0):
    """Glorot y Bengio (2010): U(−a, a) con a = gain·√(6/(n_in + n_out)), varianza 2/(n_in + n_out)."""
    a = gain * np.sqrt(6 / (n_in + n_out))
    return rng.uniform(-a, a, (n_out, n_in))


def he_normal(n_out, n_in, rng):
    """He et al. (2015), modo fan_in: N(0, 2/n_in). Es nn.init.kaiming_normal_ con nonlinearity='relu'."""
    return rng.normal(0, np.sqrt(2 / n_in), (n_out, n_in))


def iniciar(capas, seed):
    """capas = [1, 64, 64, 1] → [W1, b1, W2, b2, W3, b3]. Xavier (ganancia 5/3, la de tanh) y sesgos a 0."""
    rng = np.random.default_rng(seed)
    P = []
    for i, (a, b) in enumerate(zip(capas[:-1], capas[1:])):
        P += [xavier_uniform(b, a, rng, 5 / 3 if i < len(capas) - 2 else 1.0), np.zeros(b)]
    return P


def adelante(P, X, p=0.0, rng=None):
    """Pasada hacia delante. Con p > 0 y un rng, modo entrenamiento: cada neurona oculta se apaga con
    probabilidad p y las que quedan se multiplican por 1/(1 − p) (dropout invertido). Sin rng, modo evaluación."""
    H, cache = X, []
    for i in range(0, len(P) - 2, 2):
        A = np.tanh(H @ P[i].T + P[i + 1])
        M = (rng.random(A.shape) >= p) / (1 - p) if (p > 0 and rng is not None) else None
        cache.append((H, A, M))
        H = A * M if M is not None else A
    return (H @ P[-2].T + P[-1])[:, 0], cache + [(H, None, None)]


def gradiente(P, X, y, p=0.0, rng=None):
    """Error cuadrático medio del lote y su gradiente (la retropropagación del tema 5)."""
    z, cache = adelante(P, X, p, rng)
    L = float(np.mean((z - y) ** 2))
    d = (2 * (z - y) / len(y))[:, None]                   # ∂L/∂z
    G = [None] * len(P)
    Hn = cache[-1][0]
    G[-2], G[-1] = d.T @ Hn, d.sum(0)
    for k in range(len(cache) - 2, -1, -1):
        Hin, A, M = cache[k]
        d = d @ P[2 * k + 2]                               # hacia atrás por la capa siguiente
        if M is not None:
            d = d * M                                      # dropout: solo pasa por las neuronas encendidas
        d = d * (1 - A ** 2)                               # tanh' = 1 − tanh²
        G[2 * k], G[2 * k + 1] = d.T @ Hin, d.sum(0)
    return L, G


class Adam:
    """Adam de torch.optim. desacoplado=False: torch.optim.Adam(weight_decay=λ), suma λ·θ al gradiente (L2).
    desacoplado=True: torch.optim.AdamW(weight_decay=λ), θ ← θ·(1 − η·λ) antes del paso de Adam."""

    def __init__(self, lr=1e-3, wd=0.0, desacoplado=True, betas=(0.9, 0.999), eps=1e-8):
        self.lr, self.wd, self.dec, (self.b1, self.b2), self.eps = lr, wd, desacoplado, betas, eps
        self.t, self.m, self.v = 0, None, None

    def step(self, P, G):
        if self.m is None:
            self.m, self.v = [np.zeros_like(p) for p in P], [np.zeros_like(p) for p in P]
        self.t += 1
        c1, c2 = 1 - self.b1 ** self.t, 1 - self.b2 ** self.t
        for p, g, m, v in zip(P, G, self.m, self.v):
            if self.wd and self.dec:
                p *= 1 - self.lr * self.wd                 # decaimiento desacoplado (AdamW)
            elif self.wd:
                g = g + self.wd * p                        # L2 dentro del gradiente (Adam con weight_decay)
            m *= self.b1
            m += (1 - self.b1) * g
            v *= self.b2
            v += (1 - self.b2) * g * g
            p -= self.lr * (m / c1) / (np.sqrt(v / c2) + self.eps)


def entrenar(X, y, Xv, yv, capas=(1, 64, 64, 1), epocas=3000, lr=0.01, wd=0.0, desacoplado=True, p=0.0,
             lote=None, seed_init=0, seed_drop=1, rejilla=None, fotos=()):
    """Entrena con lote completo (o lotes barajados de `lote`). Tras cada época mide el error cuadrático medio de
    entrenamiento y de validación EN MODO EVALUACIÓN. Guarda la mejor época de validación y sus pesos."""
    P = iniciar(list(capas), seed_init)
    o = Adam(lr, wd, desacoplado)
    rng = np.random.default_rng(seed_drop)
    ev = lambda Q, A, b: float(np.mean((adelante(Q, A)[0] - b) ** 2))
    tr, va = [ev(P, X, y)], [ev(P, Xv, yv)]
    mejor, Pm, pred = 0, [q.copy() for q in P], {}
    if 0 in fotos and rejilla is not None:
        pred[0] = adelante(P, rejilla)[0]
    for ep in range(1, epocas + 1):
        orden = rng.permutation(len(y)) if lote else [np.arange(len(y))]
        for j in ([orden[k:k + lote] for k in range(0, len(y), lote)] if lote else orden):
            _, G = gradiente(P, X[j], y[j], p, rng if p > 0 else None)
            o.step(P, G)
        tr.append(ev(P, X, y)); va.append(ev(P, Xv, yv))
        if va[-1] < va[mejor]:
            mejor, Pm = ep, [q.copy() for q in P]
        if ep in fotos and rejilla is not None:
            pred[ep] = adelante(P, rejilla)[0]
    return dict(tr=tr, va=va, mejor=mejor, P=P, Pm=Pm, fotos=pred)
