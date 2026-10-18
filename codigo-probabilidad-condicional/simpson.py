import numpy as np
# entregas a tiempo por depósito y tipo de ruta: (a tiempo, total)
D = {"Norte": {"ciudad": (600, 800), "autopista": (190, 200)},
     "Sur":   {"ciudad": (140, 200), "autopista": (720, 800)}}
for dep, r in D.items():
    ok = sum(a for a, n in r.values()); n = sum(n for a, n in r.values())
    print(dep, {k: round(a / m, 3) for k, (a, m) in r.items()}, "total", ok, "/", n, "=", ok / n, " tarde × 30 € =", (n - ok) * 30)
# la tasa global es una media ponderada por la mezcla de rutas
for w in (0.8, 0.5, 0.2):
    print(f"Norte con {w:.0%} ciudad: {w * 0.75 + (1 - w) * 0.95:.3f}   Sur con {w:.0%} ciudad: {w * 0.70 + (1 - w) * 0.90:.3f}")
# comparación justa: misma mezcla para los dos (la de la flota entera: 1 000 ciudad, 1 000 autopista)
print("estandarizada 50/50:", 0.5 * 0.75 + 0.5 * 0.95, 0.5 * 0.70 + 0.5 * 0.90)
