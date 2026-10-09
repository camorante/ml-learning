import numpy as np
from producto import punto, norma, coseno, angulo_grados, puntuacion, distancia_frontera

km, precio = [40, 120, 200], [0.60, 0.40, 0.50]
print(punto(km, precio))
A, B, C = [12, 28], [24, 56], [22, 18]
print(punto(A, C), round(norma(A), 2), round(norma(C), 2))
print(round(coseno(A, C), 3), round(angulo_grados(A, C), 1))
print(round(norma(np.subtract(A, B)), 1), round(norma(np.subtract(A, C)), 1), round(coseno(A, B), 3))
ua, uc = np.array(A) / norma(A), np.array(C) / norma(C)
print(ua.round(3), uc.round(3), round(norma(ua - uc) ** 2, 3), round(2 * (1 - coseno(A, C)), 3))
w, b, x = [1.0, 0.8], -9.0, [6.2, 5.5]
print(round(puntuacion(w, x, b), 2), round(norma(w), 3), round(distancia_frontera(w, x, b), 3))
x_frontera = np.array(x) - puntuacion(w, x, b) * np.array(w) / norma(w) ** 2
print(x_frontera.round(3), round(puntuacion(w, x_frontera, b), 6))
print(round(puntuacion([2.0, 1.6], x, -18.0), 2), round(distancia_frontera([2.0, 1.6], x, -18.0), 3))
