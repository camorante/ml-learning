"""Ejemplo resuelto: los tres primeros pasos de Adam (tasa 0.1) sobre UN peso de la red del juguete, el que lleva
la neurona oculta 1 a la salida, con los gradientes reales de los tres primeros lotes. Y lo que haría SGD."""
import json
import numpy as np
import red
from optim import Adam
from datos import X_TR, Y_TR, escalar

X = escalar(X_TR)
rng = np.random.default_rng(1); orden = rng.permutation(600)              # el mismo barajado que el entrenamiento
P = red.iniciar(8, seed=0); o = Adam(0.1)
w0 = P[2][0, 0]; m = v = 0.0; w = w0; ws = w0
print(f"peso de partida w0 = {w0:.6f}   (β1 = 0.9, β2 = 0.999, ε = 1e-8, tasa η = 0.1)")
for t in (1, 2, 3):
    j = orden[(t - 1) * 32:t * 32]
    _, G = red.gradiente(P, X[j], Y_TR[j]); g = G[2][0, 0]
    m = 0.9 * m + 0.1 * g; v = 0.999 * v + 0.001 * g * g
    mh, vh = m / (1 - 0.9 ** t), v / (1 - 0.999 ** t)
    paso = 0.1 * mh / (np.sqrt(vh) + 1e-8); w -= paso
    print(f"paso {t}: g = {g:.6f}   m = {m:.6f}   v = {v:.4e}   1 − 0.9^{t} = {1 - 0.9 ** t:.3f}   1 − 0.999^{t} = {1 - 0.999 ** t:.6f}")
    print(f"        m̂ = {mh:.6f}   v̂ = {vh:.4e}   √v̂ = {np.sqrt(vh):.6f}   m̂/√v̂ = {mh / np.sqrt(vh):.4f}"
          f"   paso = {paso:.6f}   w = {w:.6f}   (SGD con la misma tasa movería {0.1 * g:.6f})")
    ws -= 0.1 * g
    o.step(P, G)
print(f"tras 3 pasos: Adam w = {w:.6f} (se ha movido {w - w0:+.6f});  SGD con tasa 0.1 w = {ws:.6f} ({ws - w0:+.6f})")
print(f"el mismo peso dentro de optim.Adam tras 3 pasos: {P[2][0, 0]:.6f}  (diferencia {abs(P[2][0, 0] - w):.1e})")
f = json.load(open("bucle.json", encoding="utf-8"))["fotos"][2][2][0][0]
print(f"y en el entrenamiento guardado (redondeado a 4 decimales): {f}")
print(f"sin corregir el sesgo, el primer paso sería 0.1·m/(√v + ε) = {0.1 * (0.1 * 0.141585) / (np.sqrt(0.001 * 0.141585 ** 2)):.4f}")
