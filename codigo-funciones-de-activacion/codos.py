"""Idea 3: una recta más unos pocos 'codos' ReLU, ajustada por mínimos cuadrados (lo mismo que hace el JS)."""
import numpy as np
from datos import tramos, verdad

relu = lambda z: np.maximum(z, 0)

def base(v, codos):
    u = np.asarray(v, float) / 100
    return np.column_stack([np.ones_like(u), u] + [relu(u - c / 100) for c in codos])

def ajustar(v, c, codos):
    coef, *_ = np.linalg.lstsq(base(v, codos), c, rcond=None)
    return coef

if __name__ == "__main__":
    v, c = tramos()
    for codos in ([], [70], [40], [30, 70], [35, 75], [25, 50, 80], [20, 40, 70]):
        k = ajustar(v, c, codos); pred = base(v, codos) @ k
        print(f"codos {str(codos):13s} error medio {np.sqrt(np.mean((pred - c) ** 2)):.3f} L/100 km   coef {np.round(k, 2)}")
    # mejores posiciones en una rejilla de 1 km/h
    g = range(20, 101)
    best1 = min(g, key=lambda a: np.mean((base(v, [a]) @ ajustar(v, c, [a]) - c) ** 2))
    best2 = min(((a, b) for a in g for b in g if b > a + 4), key=lambda p: np.mean((base(v, p) @ ajustar(v, c, p) - c) ** 2))
    for p in ([best1], list(best2)):
        pred = base(v, p) @ ajustar(v, c, p); print("mejor", p, round(float(np.sqrt(np.mean((pred - c) ** 2))), 3))
    print("ruido de los datos (curva verdadera):", round(float(np.sqrt(np.mean((verdad(v) - c) ** 2))), 3))
