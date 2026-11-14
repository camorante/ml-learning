import math
from neurona import Perceptron
from datos import X, y

eta = 0.1
# 1) y 2) la primera idea con la jornada 5: 4,6 frenadas/100 km, 2,6 h de ralentí, marcada «revisar»
w1, w2, b = 1.0, 0.2, -6.0
x1, x2, t = 4.6, 2.6, 1
z = w1 * x1 + w2 * x2 + b
print(f"2) z = {w1}·{x1} + {w2}·{x2} + ({b}) = {w1 * x1:.2f} + {w2 * x2:.2f} − 6 = {z:.2f}  → dice {int(z > 0)}, era {t}")
# 3) la frontera de la primera idea: frenadas + 0,2·horas = 6
print(f"3) corta el eje de frenadas en {-b / w1:.1f} (0 h) y llega a {(-b - w2 * 5) / w1:.1f} frenadas con 5 h")
# 4) una corrección: s = +1 porque debía decir «revisar»
s = 1
n1, n2, nb = w1 + eta * s * x1, w2 + eta * s * x2, b + eta * s
z2 = n1 * x1 + n2 * x2 + nb
print(f"4) w = ({n1:.2f}, {n2:.2f}), b = {nb:.1f}   z nueva = {z2:.3f}"
      f"   subida = {z2 - z:.3f} = η·(x1² + x2² + 1) = {eta * (x1**2 + x2**2 + 1):.3f}")
# 5) tras entrenar con las 20 jornadas
p = Perceptron(eta=eta, w0=(1.0, 0.2), b0=-6.0).fit(X, y)
W1, W2, B = round(p.w[0], 2), round(p.w[1], 2), round(p.b, 2)
x1, x2 = 4.1, 2.2
z = W1 * x1 + W2 * x2 + B
print(f"5) {p.correcciones} correcciones → w = ({W1}, {W2}), b = {B}   nueva jornada: {W1}·{x1} + {W2}·{x2} − 6"
      f" = {W1 * x1:.3f} + {W2 * x2:.3f} − 6 = {z:.3f}")
print(f"   1 h de ralentí equivale a {W2 / W1:.2f} frenadas; cortes de la frontera: {-B / W1:.2f} frenadas, {-B / W2:.2f} h")
# 6) la misma neurona con sigmoide, y un paso de gradiente de la entropía cruzada (y = 1)
q = 1 / (1 + math.exp(-z))
print(f"6) e^(−z) = {math.exp(-z):.4f}   p = σ(z) = {q:.4f}   pérdida −ln p = {-math.log(q):.4f}")
g = q - 1
print(f"   ∂L/∂z = p − y = {g:.4f}   ∂L/∂w = ({g * x1:.4f}, {g * x2:.4f})   ∂L/∂b = {g:.4f}")
v1, v2, vb = W1 - eta * g * x1, W2 - eta * g * x2, B - eta * g
zz = v1 * x1 + v2 * x2 + vb
print(f"   tras un paso (η = 0.1): w = ({v1:.4f}, {v2:.4f}), b = {vb:.4f}   z = {zz:.4f}   p = {1 / (1 + math.exp(-zz)):.4f}")
