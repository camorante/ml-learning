import numpy as np
from neurona import puntuacion, escalon, sigmoide

w, b = np.array([0.83, 1.28]), -6.0            # pesos aprendidos (frenadas/100 km, horas de ralentí) y sesgo
x = np.array([4.1, 2.2])                       # una jornada nueva
z = w @ x + b                                  # producto punto + sesgo
print(f"z = {w[0]}·{x[0]} + {w[1]}·{x[1]} + ({b}) = {z:.3f}   escalón → {escalon(z)}   σ(z) = {sigmoide(z):.4f}")
print(f"distancia con signo a la frontera: z/‖w‖ = {z:.3f}/{np.linalg.norm(w):.4f} = {z / np.linalg.norm(w):.4f}")
print(f"dirección normal w/‖w‖ = {np.round(w / np.linalg.norm(w), 4)}")

# el sesgo como un peso más con una entrada fija 1
wa, xa = np.r_[w, b], np.r_[x, 1.0]
print(f"con la entrada aumentada (x, 1): w̃·x̃ = {wa @ xa:.3f}")

# la misma cuenta para un lote de jornadas: un producto matriz-vector
X = np.array([[6.8, 1.9], [1.2, 0.8], [4.1, 2.2], [6.6, 0.4]])
Z = puntuacion(X, w, b)
print("lote:", np.round(Z, 3), escalon(Z), np.round(sigmoide(Z), 3))

# multiplicar w y b por k > 0 no mueve la frontera, pero cambia la probabilidad
for k in (0.25, 1, 4):
    print(f"k = {k:<4}  σ(k·z) = {sigmoide(k * Z).round(3)}   decisiones {escalon(k * Z)}")
