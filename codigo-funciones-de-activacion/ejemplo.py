import json
import numpy as np
from datos import tramos, verdad
from codos import ajustar

v, c = tramos()
relu = lambda z: max(z, 0.0)
# 0) la red ReLU del juguete (1 capa oculta, 3 neuronas): dónde pone cada neurona su codo
(W1, b1), (W2, b2) = json.load(open("red_toy.json"))["1-relu"]["capas"]
print("0) codos de la red:", [round(-b / w[0] * 100, 1) for w, b in zip(W1, b1)], "km/h")
# 1) la misma red escrita como recta + 2 codos, con los números de la salida por mínimos cuadrados
a, b, s1, s2 = np.round(ajustar(v, c, [40, 73]), 2)
print(f"1) a = {a}  b = {b}  s1 = {s1}  s2 = {s2}")
red = lambda vel: a + b * vel / 100 + s1 * relu(vel / 100 - 0.40) + s2 * relu(vel / 100 - 0.73)
# 2) hacia delante en tres tramos
for vel in (25, 60, 90):
    u = vel / 100
    print(f"2) v = {vel}: u = {u}  h2 = {relu(u - 0.40):.2f}  h3 = {relu(u - 0.73):.2f}"
          f"  consumo = {red(vel):.2f}   (curva real {verdad(vel):.2f})")
# 3) sin activación las capas se funden: una recta
a0, b0 = a - 0.40 * s1 - 0.73 * s2, b + s1 + s2
print(f"3) sin ReLU: consumo = {a0:.2f} + {b0:.2f}·u  → a 25 km/h {a0 + b0 * 0.25:.2f}")
rb, ra = np.polyfit(v / 100, c, 1)
print(f"   mejor recta: {ra:.2f} {rb:+.2f}·u  → a 25/60/90 km/h:", [round(float(ra + rb * x), 2) for x in (0.25, 0.6, 0.9)])
# 4) pendientes (derivadas) de cada tramo, en L/100 km por km/h
print(f"4) pendientes: {b / 100:.4f}  {(b + s1) / 100:.4f}  {(b + s1 + s2) / 100:.4f};  mínimo en 73 km/h: {red(73):.2f}")
# 5) gasóleo de dos rutas: reparto urbano (100 km a 25 km/h) y autovía (400 km a 105 km/h), a 1,60 €/L
recta = lambda x: ra + rb * x / 100
for km, vel in ((100, 25), (400, 105)):
    real, r, l = (km * f(vel) / 100 for f in (verdad, red, recta))
    print(f"5) {km} km a {vel} km/h: curva real {real:.1f} L  red {r:.1f} L ({r - real:+.1f})"
          f"  recta {l:.1f} L ({l - real:+.1f})  → error de la recta {(l - real) * 1.60:+.2f} €/ruta,"
          f" {(l - real) * 1.60 * 250:+,.0f} € en 250 rutas")
# 6) corrección que llega a la primera de 6 capas
print(f"6) sigmoide: 0.25^5 = {0.25 ** 5:.6f}   ReLU activa: 1^5 = 1")
