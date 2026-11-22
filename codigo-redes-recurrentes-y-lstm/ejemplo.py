"""El ejemplo resuelto: la LSTM de una sola unidad (hueco de 25 lecturas) con sus pesos redondeados a 2 decimales,
paso a paso sobre el primer tramo con frenada brusca de la prueba. Imprime cada puerta con 4 decimales."""
import math
import numpy as np
from datos import datos

W = [-3.72, -1.02, 1.87, 5.49]          # pesos de la entrada, bloques i, f, g, o (orden de PyTorch)
U = [2.74, -1.59, -0.14, 0.07]          # pesos del estado anterior h_{t-1}
b = [-2.82, 8.53, 1.06, 4.11]           # b_ih + b_hh: PyTorch guarda dos sesgos y solo cuenta su suma
wo, bo = -5.09, 0.05                    # la salida: z = wo·h_T + bo, probabilidad σ(z)
sg = lambda z: 1 / (1 + math.exp(-z))

_, _, (X, Y, _) = datos(25)
x = [round(float(v), 2) for v in X[int(np.argmax(Y == 1))]]
print("lecturas 1 a 9:", x[:9], "· lecturas en total:", len(x))
h = c = 0.0
for t, v in enumerate(x, 1):
    xs = v / 4                                          # la lectura entra dividida por 4
    zi, zf = W[0] * xs + U[0] * h + b[0], W[1] * xs + U[1] * h + b[1]
    zg, zo = W[2] * xs + U[2] * h + b[2], W[3] * xs + U[3] * h + b[3]
    i, f, g, o = sg(zi), sg(zf), math.tanh(zg), sg(zo)
    c_ant, c = c, f * c + i * g
    h_ant, h = h, o * math.tanh(c)
    if t in (1, 2, 3, 9, len(x)):
        print(f"paso {t:2d}: x = {v:5.2f} (x/4 = {xs:.4f}), h_ant = {h_ant:.4f}, c_ant = {c_ant:.4f}")
        print("   términos W·x/4 y U·h_ant:", "  ".join(f"{n}: {W[k] * xs:.4f} {U[k] * h_ant:+.4f}" for k, n in enumerate("ifgo")))
        print(f"   z_i = {zi:.4f} → i = {i:.4f} · z_f = {zf:.4f} → f = {f:.5f} · z_g = {zg:.4f} → g = {g:.4f} · z_o = {zo:.4f} → o = {o:.4f}")
        print(f"   c = {f:.5f}·({c_ant:.4f}) + {i:.4f}·({g:.4f}) = {c:.4f} ;  h = {o:.4f}·tanh({c:.4f}) = {h:.4f}")
z = wo * h + bo
print(f"respuesta al final: z = {wo}·({h:.4f}) + {bo} = {z:.4f} → probabilidad de frenada brusca al salir {sg(z):.4f}")
print(f"si la puerta de olvido fuera 0,9 en los 32 pasos que siguen a la lectura 9: 0,9^32 = {0.9 ** 32:.4f}")
