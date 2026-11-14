import numpy as np
from datos import XOR, carga_consumo
from red import MLP, Densa

escalon = lambda z: (np.asarray(z) > 0).astype(float)
X, y = XOR
A = escalon(X @ [1, 1] - 0.5)                  # neurona A: x1 + x2 > 0.5  (al menos uno)
B = escalon(X @ [1, 1] - 1.5)                  # neurona B: x1 + x2 > 1.5  (los dos)
puertas = {"A o B   (1, 1, −0.5)": (1, 1, -0.5), "A y B   (1, 1, −1.5)": (1, 1, -1.5),
           "A y no B (1, −1, −0.5)": (1, -1, -0.5)}
print("x       A  B   objetivo")
for x, a, b, t in zip(X.astype(int), A, B, y):
    print(x, " ", int(a), int(b), "   ", t)
for nombre, (u, v, c) in puertas.items():
    s = escalon(u * A + v * B + c)
    print(f"salida {nombre}: {s.astype(int)}  errores {int(np.sum(s != y))}")

# con sigmoides empinadas (×10) la misma red da probabilidades casi 0/1
red = MLP([Densa([[10, 10], [10, 10]], [-5, -15]), Densa([[10, -10]], [-5])])
print("XOR con sigmoides:", red.forward(X)[:, 0].round(4))

# la franja de los viajes: «gasta de más» o «gasta de menos», con escalones
C, yc = carga_consumo()
arriba = escalon(C[:, 1] - (26.5 + 0.55 * C[:, 0]))
abajo = escalon((21.5 + 0.55 * C[:, 0]) - C[:, 1])
for nombre, s in (("solo A", arriba), ("solo B", abajo), ("A o B", escalon(arriba + abajo - 0.5))):
    print(f"franja, {nombre:6s}: avisa en {int(s.sum()):2d} viajes, errores {int(np.sum(s != yc))} de 30")
