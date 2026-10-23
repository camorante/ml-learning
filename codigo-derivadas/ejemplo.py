import math
from valor import Valor

x, y, w, b, eta = 2.0, 1, 0.5, -1.5, 0.5             # 2 atascos, llegó tarde; pesos iniciales; paso
# 1) hacia delante
u = w * x; z = u + b; p = 1 / (1 + math.exp(-z)); L = -math.log(p)
print(f"1) u = {u}  z = {z}  e^0,5 = {math.exp(0.5):.4f}  p = {p:.4f}  L = −log p = {L:.4f}")
# 2) y 3) hacia atrás
dLp = -1 / p; dpz = p * (1 - p); dLz = dLp * dpz
print(f"2) ∂L/∂p = {dLp:.4f}  ∂p/∂z = {dpz:.4f}  ∂L/∂z = {dLz:.4f}  (p − y = {p - y:.4f})")
print(f"3) ∂L/∂w = {dLz * x:.4f}  ∂L/∂b = {dLz:.4f}  ∂L/∂x = {dLz * w:.4f}")
# el mismo resultado con el autograd de valor.py
W, B = Valor(w), Valor(b); Lv = -(W * x + B).sigmoide().log(); Lv.atras()
print(f"   Valor: L = {Lv.dato:.4f}  ∂L/∂w = {W.grad:.4f}  ∂L/∂b = {B.grad:.4f}")
# 4) comprobación con diferencias centrales
Lf = lambda w_, b_: math.log(1 + math.exp(-(w_ * x + b_)))   # −log σ(z) = log(1 + e^(−z))
h = 1e-4
for nombre, mas, menos in (("w", Lf(w + h, b), Lf(w - h, b)), ("b", Lf(w, b + h), Lf(w, b - h))):
    print(f"4) L({nombre} + h) = {mas:.10f}  L({nombre} − h) = {menos:.10f}"
          f"  (dif)/2h = {(mas - menos) / (2 * h):.8f}")
# 5) un paso de descenso de gradiente
w2, b2 = w - eta * dLz * x, b - eta * dLz
z2 = w2 * x + b2; p2 = 1 / (1 + math.exp(-z2))
print(f"5) w = {w2:.4f}  b = {b2:.4f}  z = {z2:.4f}  p = {p2:.4f}  L = {-math.log(p2):.4f}")
# 6) regla de la cadena km/h → L/h → €/h a 80 km/h
v, precio = 80, 1.6
Lh = (20 + 0.0016 * v ** 2) * v / 100; dLh = 0.2 + 3 * 0.0016 * v ** 2 / 100
print(f"6) L/h = {Lh:.4f}  d(L/h)/dv = {dLh:.4f}  €/h = {precio * Lh:.4f}"
      f"  d(€/h)/dv = {precio * dLh:.4f}")
print(f"   +1 km/h de verdad: {precio * ((20 + 0.0016 * 81 ** 2) * 81 / 100 - Lh):.4f} €/h")
# 7) velocidad más barata: coste(v) = 1,6·(20 + 0,0016 v²) + 3000/v
c = lambda v: 1.6 * (20 + 0.0016 * v * v) + 3000 / v
vs = (3000 / 0.00512) ** (1 / 3)
print(f"7) v* = {vs:.4f}  coste'(v*) = {0.00512 * vs - 3000 / vs ** 2:.1e}"
      f"  coste''(v*) = {0.00512 + 6000 / vs ** 3:.5f}"
      f"  coste(v*) = {c(vs):.4f}  coste(90) = {c(90):.4f}  coste(60) = {c(60):.4f}")
