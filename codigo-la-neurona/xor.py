import itertools
import numpy as np
from neurona import Perceptron
from datos import XOR, carga_consumo

# 1) XOR: ninguna recta separa (0,1),(1,0) de (0,0),(1,1); el perceptrón no para nunca
X, y = XOR
p = Perceptron(eta=1.0, epocas=12).fit(X, y)
print("XOR, fallos por pasada:", p.fallos_por_pasada, "  pesos tras 12 pasadas:", p.w, p.b)
mejor = min(np.sum(((X @ w + b) > 0) != y) for w in itertools.product(np.linspace(-2, 2, 41), repeat=2)
            for b in np.linspace(-2, 2, 41))
print("mejor recta en XOR (rejilla de 41³ pesos):", mejor, "error de 4")
# con una variable más, x1·x2, sí es separable (en 3D basta un plano)
X3 = np.c_[X, X[:, 0] * X[:, 1]]
q = Perceptron(eta=1.0).fit(X3, y)
print("XOR con x1·x2 añadida:", q.correcciones, "correcciones; w =", q.w, " b =", q.b, " fallos:", np.sum(q.predict(X3) != y))

# 2) carga y consumo: lo normal es una franja; una recta deja escapar un lado
C, yc = carga_consumo()
pc = Perceptron(eta=0.01, epocas=200).fit(C, yc)
print(f"carga-consumo: perceptrón tras 200 pasadas, fallos en las 10 últimas {pc.fallos_por_pasada[-10:]}")
# variable nueva: distancia a la recta de consumo normal (24 + 0,55·carga)
d = np.abs(C[:, 1] - (24 + 0.55 * C[:, 0]))
Xd = np.c_[C, d]
r = Perceptron(eta=0.01).fit(Xd, yc)
print(f"con |consumo − (24 + 0,55·carga)| como tercera entrada: {r.correcciones} correcciones,"
      f" {len(r.fallos_por_pasada)} pasadas, fallos {np.sum(r.predict(Xd) != yc)}")
