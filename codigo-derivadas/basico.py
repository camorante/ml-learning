import numpy as np
from datos import posicion, velocidad

t = 19.25                                        # minuto 19,25: empezando a frenar
v = float(velocidad(t)); print("velocímetro (derivada exacta):", v, "km/h")
for h in [10, 1, 0.1, 0.01, 1e-3, 1e-4]:
    adelante = (posicion(t + h) - posicion(t)) / h * 60               # km/min → km/h
    central = (posicion(t + h) - posicion(t - h)) / (2 * h) * 60
    print(f"h = {h:<7g} min   adelante {adelante:9.4f}   error {adelante - v:+.2e}"
          f"   central {central:9.4f}   error {central - v:+.2e}")
