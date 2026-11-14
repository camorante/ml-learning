import math
import numpy as np
from datos import carga_consumo
from red import n_param

C, y = carga_consumo()
sig = lambda z: 1 / (1 + math.exp(-z))
k = 2.0                                             # cuánto sube la puntuación por cada litro fuera de la franja
wA, bA = (-0.55 * k, k), -26.5 * k                  # A: consumo − 0.55·carga − 26.5 > 0  («gasta de más»)
wB, bB = (0.55 * k, -k), 21.5 * k                   # B: 21.5 + 0.55·carga − consumo > 0  («gasta de menos»)
print(f"2-3) wA = {wA}, bA = {bA}   wB = {wB}, bB = {bB}   σ(k·1 L) = {sig(k):.4f}")
for i in (5, 3, 6):                                 # viaje 5 (por encima), 3 (por debajo), 6 (normal)
    c, q = C[i - 1]
    zA = wA[0] * c + wA[1] * q + bA; zB = wB[0] * c + wB[1] * q + bB
    hA, hB = sig(zA), sig(zB)
    z2 = 10 * hA + 10 * hB - 5; p = sig(z2)
    print(f"viaje {i} ({c} t, {q} L, marca {y[i - 1]}): 0.55·carga = {0.55 * c:.3f}, desvío = {q - 24 - 0.55 * c:.3f} L")
    print(f"   zA = {wA[0]}·{c} + {wA[1]}·{q} {bA} = {wA[0] * c:.2f} + {wA[1] * q:.2f} {bA} = {zA:.2f}   hA = {hA:.6f}")
    print(f"   zB = {wB[0]}·{c} {wB[1]}·{q} + {bB} = {wB[0] * c:.2f} {wB[1] * q:.2f} + {bB} = {zB:.2f}   hB = {hB:.6f}")
    print(f"   z = 10·{hA:.4f} + 10·{hB:.4f} − 5 = {z2:.3f}   p = {p:.4f}")
H = np.array([[sig(wA[0] * c + wA[1] * q + bA), sig(wB[0] * c + wB[1] * q + bB)] for c, q in C])
print(f"7) hA + hB: normales ≤ {H[y == 0].sum(1).max():.3f}, raros ≥ {H[y == 1].sum(1).min():.3f};"
      f"  parámetros {n_param([2, 2, 1])} = (2 + 1)·2 + (2 + 1)·1")
