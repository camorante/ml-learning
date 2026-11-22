"""Atención, atención multicabeza y un bloque Transformer desde cero en NumPy: hacia delante y hacia atrás.
Convención: una secuencia es una matriz X de forma (n, d), una fila por ficha; las capas son X @ W + b."""
import numpy as np


def softmax(Z):
    Z = Z - Z.max(-1, keepdims=True)                   # restar el máximo no cambia nada y evita desbordes
    E = np.exp(Z)
    return E / E.sum(-1, keepdims=True)


def mascara_causal(n):
    return np.triu(np.ones((n, n), dtype=bool), 1)      # True encima de la diagonal: el futuro


def atencion(Q, K, V, causal=False):
    """Q, K: (n, dk) · V: (n, dv) → salida (n, dv) y pesos A (n, n); cada fila de A suma 1."""
    S = Q @ K.T / np.sqrt(Q.shape[-1])
    if causal:
        S = np.where(mascara_causal(len(S)), -np.inf, S)   # e^−∞ = 0: nadie mira al futuro
    A = softmax(S)
    return A @ V, A


def atencion_atras(dO, Q, K, V, A):
    dV = A.T @ dO
    dA = dO @ V.T
    dS = A * (dA - (dA * A).sum(-1, keepdims=True))     # jacobiano del softmax, fila a fila
    dS = dS / np.sqrt(Q.shape[-1])                      # (donde la máscara puso 0, dS también es 0)
    return dS @ K, dS.T @ Q, dV


def mha(X, P, h, causal=False):
    """Atención multicabeza: h cabezas de tamaño d/h sobre trozos de Q, K y V; se concatenan y se mezclan con Wo."""
    Q, K, V = X @ P["Wq"] + P["bq"], X @ P["Wk"] + P["bk"], X @ P["Wv"] + P["bv"]
    dh = Q.shape[1] // h
    sl = [slice(i * dh, (i + 1) * dh) for i in range(h)]
    sal = [atencion(Q[:, s], K[:, s], V[:, s], causal) for s in sl]
    C = np.concatenate([o for o, _ in sal], axis=1)
    cache = (X, Q, K, V, C, [a for _, a in sal], sl)
    return C @ P["Wo"] + P["bo"], cache


def mha_atras(dY, cache, P):
    X, Q, K, V, C, As, sl = cache
    g = {"Wo": C.T @ dY, "bo": dY.sum(0)}
    dC = dY @ P["Wo"].T
    dQ, dK, dV = np.zeros_like(Q), np.zeros_like(K), np.zeros_like(V)
    for s, A in zip(sl, As):
        dQ[:, s], dK[:, s], dV[:, s] = atencion_atras(dC[:, s], Q[:, s], K[:, s], V[:, s], A)
    for n, dZ in (("q", dQ), ("k", dK), ("v", dV)):
        g["W" + n], g["b" + n] = X.T @ dZ, dZ.sum(0)
    return dQ @ P["Wq"].T + dK @ P["Wk"].T + dV @ P["Wv"].T, g


def capa_norma(X, g, b, eps=1e-5):
    mu, var = X.mean(-1, keepdims=True), X.var(-1, keepdims=True)
    Xh = (X - mu) / np.sqrt(var + eps)
    return g * Xh + b, (Xh, np.sqrt(var + eps), g)


def capa_norma_atras(dY, cache):
    Xh, s, g = cache
    dXh = dY * g
    dX = (dXh - dXh.mean(-1, keepdims=True) - Xh * (dXh * Xh).mean(-1, keepdims=True)) / s
    return dX, (dY * Xh).sum(0), dY.sum(0)


def bloque(X, P, h, causal=False):
    """Bloque Transformer con la norma delante (pre-LN), como GPT-2:
    H = X + MHA(LN1(X)) · Y = H + FFN(LN2(H)), con FFN(z) = ReLU(z W1 + b1) W2 + b2 en cada posición."""
    A, c1 = capa_norma(X, P["g1"], P["be1"])
    M, cm = mha(A, P, h, causal)
    H = X + M                                           # conexión residual
    B, c2 = capa_norma(H, P["g2"], P["be2"])
    Z = B @ P["W1"] + P["b1"]
    R = np.maximum(Z, 0)
    Y = H + R @ P["W2"] + P["b2"]                       # otra conexión residual
    return Y, (c1, cm, c2, B, Z, R)


def bloque_atras(dY, cache, P):
    c1, cm, c2, B, Z, R = cache
    g = {"W2": R.T @ dY, "b2": dY.sum(0)}
    dZ = (dY @ P["W2"].T) * (Z > 0)
    g["W1"], g["b1"] = B.T @ dZ, dZ.sum(0)
    dH, g["g2"], g["be2"] = capa_norma_atras(dZ @ P["W1"].T, c2)
    dH = dH + dY                                        # el atajo deja pasar el gradiente tal cual
    dA, gm = mha_atras(dH, cm, P)
    g.update(gm)
    dX, g["g1"], g["be1"] = capa_norma_atras(dA, c1)
    return dX + dH, g


def posiciones(n, d):
    """Codificación posicional sinusoidal (Vaswani et al., 2017): seno en las columnas pares, coseno en las impares."""
    p = np.arange(n)[:, None]
    w = 1.0 / 10000 ** (np.arange(0, d, 2) / d)
    PE = np.zeros((n, d))
    PE[:, 0::2], PE[:, 1::2] = np.sin(p * w), np.cos(p * w)
    return PE


def modelo(ids, P, h, causal=True):
    """El modelo del tema: embedding + posición → un bloque → norma final → lineal al vocabulario (logits)."""
    X = P["E"][ids] + posiciones(len(ids), P["E"].shape[1])
    Y, cache = bloque(X, P, h, causal)
    N, cf = capa_norma(Y, P["gf"], P["bef"])
    return N @ P["Wout"] + P["bout"], (ids, cache, cf, N)


def modelo_atras(dZ, cache, P):
    ids, cb, cf, N = cache
    g = {"Wout": N.T @ dZ, "bout": dZ.sum(0)}
    dY, g["gf"], g["bef"] = capa_norma_atras(dZ @ P["Wout"].T, cf)
    dX, gb = bloque_atras(dY, cb, P)
    g.update(gb)
    g["E"] = np.zeros_like(P["E"])
    np.add.at(g["E"], ids, dX)                          # la tabla de embeddings del tema 10: gradiente disperso
    return g


def entropia_cruzada(Z, y):
    """Pérdida media −ln p(y) y su gradiente respecto a los logits (softmax − one-hot) / n."""
    P = softmax(Z)
    n = len(y)
    dZ = P.copy()
    dZ[np.arange(n), y] -= 1
    return -np.log(P[np.arange(n), y]).mean(), dZ / n
