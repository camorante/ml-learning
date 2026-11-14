"""Ejemplo resuelto: el viaje 3 (4,8 t, 22,6 L/100 km, gasta de menos) por la red 2-2-1 mal colocada.
Pasada hacia delante, los 9 gradientes a mano, un paso con η = 0.03 y la pérdida nueva."""
import math
import numpy as np
from datos import carga_consumo, red_inicial, ETA

C, y = carga_consumo()
c, q = C[2]; t = y[2]                                # viaje 3
sig = lambda z: 1 / (1 + math.exp(-z))
wA, bA, wB, bB, v, b2 = (-1.1, 2.0), -53.0, (1.1, -2.0), 36.0, (10.0, 10.0), -5.0
zA = wA[0] * c + wA[1] * q + bA; zB = wB[0] * c + wB[1] * q + bB
hA, hB = sig(zA), sig(zB)
z = v[0] * hA + v[1] * hB + b2; p = sig(z); L = -math.log(p) if t else -math.log(1 - p)
print(f"1) viaje 3: carga {c} t, consumo {q} L, marca {t}")
print(f"   zA = {zA:.2f}  hA = {hA:.3e}   zB = {wB[0]}·{c} {wB[1]}·{q} + {bB} = {wB[0] * c:.2f} − {-wB[1] * q:.2f} + {bB} = {zB:.2f}  hB = {hB:.5f}")
print(f"   z = 10·{hA:.2e} + 10·{hB:.5f} − 5 = {z:.4f}   p = {p:.5f}   pérdida = −ln p = {L:.4f}")
dz = p - t
print(f"2) ∂L/∂z = p − y = {dz:.5f}")
gv = (hA * dz, hB * dz); gb2 = dz
print(f"3) ∂L/∂vA = hA·(p − y) = {gv[0]:.2e}   ∂L/∂vB = {hB:.5f}·({dz:.5f}) = {gv[1]:.5f}   ∂L/∂c = {gb2:.5f}")
sA, sB = hA * (1 - hA), hB * (1 - hB)
dA, dB = dz * v[0] * sA, dz * v[1] * sB
print(f"4) σ'(zA) = {sA:.3e}   σ'(zB) = {hB:.5f}·{1 - hB:.5f} = {sB:.5f}")
print(f"   δA = ({dz:.5f})·10·{sA:.3e} = {dA:.3e}   δB = ({dz:.5f})·10·{sB:.5f} = {dB:.5f}")
gA = (dA * c, dA * q, dA); gB = (dB * c, dB * q, dB)
print(f"5) ∂L/∂wB = δB·(carga, consumo) = ({gB[0]:.4f}, {gB[1]:.4f})   ∂L/∂bB = {gB[2]:.5f}")
print(f"   ∂L/∂wA = ({gA[0]:.2e}, {gA[1]:.2e})   ∂L/∂bA = {gA[2]:.2e}")
dx = (dA * wA[0] + dB * wB[0], dA * wA[1] + dB * wB[1])
print(f"   (por los dos caminos) ∂L/∂consumo = δA·2 + δB·(−2) = {dA * 2:.2e} + {dB * -2:.5f} = {dx[1]:.5f}")
# paso
nwB = (wB[0] - ETA * gB[0], wB[1] - ETA * gB[1]); nbB = bB - ETA * gB[2]
nwA = (wA[0] - ETA * gA[0], wA[1] - ETA * gA[1]); nbA = bA - ETA * gA[2]
nv = (v[0] - ETA * gv[0], v[1] - ETA * gv[1]); nb2 = b2 - ETA * gb2
print(f"6) η = {ETA}: wB = ({nwB[0]:.4f}, {nwB[1]:.4f})  bB = {nbB:.4f}  vB = {nv[1]:.4f}  c = {nb2:.4f}  (A casi no cambia)")
zB2 = nwB[0] * c + nwB[1] * q + nbB; zA2 = nwA[0] * c + nwA[1] * q + nbA
hB2, hA2 = sig(zB2), sig(zA2); z2 = nv[0] * hA2 + nv[1] * hB2 + nb2; p2 = sig(z2)
print(f"7) zB = {nwB[0]:.4f}·{c} {nwB[1]:.4f}·{q} + {nbB:.4f} = {zB2:.3f}  hB = {hB2:.4f}"
      f"   z = {z2:.3f}  p = {p2:.4f}   pérdida {L:.3f} → {-math.log(p2):.3f}")
# comprobación con retro.py: el mismo viaje solo (lote de 1)
red = red_inicial(); Lr, g = red.gradientes(C[2:3], y[2:3]); red.paso(C[2:3], y[2:3], ETA)
print(f"   retro.py (lote de 1): pérdida {Lr:.4f} → {red.perdida_media(C[2:3], y[2:3]):.4f}")
# y en los otros 29 viajes: este paso, ¿estropea algo?
r0 = red_inicial()
print(f"   pérdida media de los 30 viajes: {r0.perdida_media(C, y):.4f} → {red.perdida_media(C, y):.4f}")
