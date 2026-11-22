"""Skip-gram con muestreo negativo (word2vec) desde cero en NumPy, sobre los mismos pares de códigos.
En lugar de un softmax sobre los 20 códigos, cada par (a, b) real se compara con k códigos «negativos» sorteados:
pérdida = −ln σ(e_a·u_b) − Σ ln σ(−e_a·u_n). Con un vocabulario de 100 000 palabras, eso ahorra calcular
100 000 logits por ejemplo."""
import numpy as np
from datos import V, NOMBRES, SISTEMA, P_TR
from modelos import coseno, vecinos, pureza

sig = lambda x: 1 / (1 + np.exp(-x))


def sgns(d, k=5, epocas=40, lr=0.05, seed=0):
    rng = np.random.default_rng(seed)
    E = rng.uniform(-0.5 / d, 0.5 / d, size=(V, d))           # vectores de entrada (los embeddings)
    U = np.zeros((V, d))                                       # vectores de salida, a cero como en word2vec
    f = np.bincount(P_TR[:, 1], minlength=V) ** 0.75; f /= f.sum()   # negativos ∝ frecuencia^0,75
    T, t = epocas * len(P_TR), 0
    for ep in range(epocas):
        tot = 0.0
        for a, b in P_TR[rng.permutation(len(P_TR))]:
            eta = lr * max(1e-4, 1 - t / T); t += 1             # la tasa baja en línea recta, como en word2vec
            neg = rng.choice(V, size=k, p=f)
            ids = np.r_[b, neg]; et = np.r_[1.0, np.zeros(k)]  # 1 = par real, 0 = negativo
            s = sig(U[ids] @ E[a])
            tot -= np.log(s[0]) + np.log(1 - s[1:]).sum()
            g = s - et                                         # ∂pérdida/∂(e_a·u)
            ga = g @ U[ids]
            np.add.at(U, ids, -eta * g[:, None] * E[a])         # solo se tocan k + 1 filas de U
            E[a] -= eta * ga                                    # y una fila de E
    return E, tot / len(P_TR)


for d in (2, 8):
    E, L = sgns(d)
    ok, n = pureza(E, SISTEMA)
    print(f"d = {d}: pérdida media de la última época {L:.3f} · vecinos por coseno del mismo sistema: {ok} de {n}")
    S = coseno(E)
    for c in ("P0217", "P0401", "C0035", "U0121"):
        i = NOMBRES.index(c)
        print(f"   {c} → " + ", ".join(f"{NOMBRES[j]} {S[i, j]:.3f}" for j in vecinos(S, i)))
