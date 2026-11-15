"""Ejemplo resuelto. (A) El primer paso de AdamW (tasa 0.01, λ = 1) sobre UN peso de la red del juguete, frente a Adam con
L2 y a SGD con L2. (B) Una pasada hacia delante con dropout p = 0.3 por la red de la idea 5, para un viaje a 85 km/h."""
import json
import numpy as np
from datos import V_TR, Y_TR, V_VA, Y_VA, ex, ey, SY, MY
from mlp import iniciar, gradiente, entrenar, Adam

# (A) el peso que une la neurona 1 de la primera capa oculta con la neurona 1 de la segunda
P = iniciar([1, 64, 64, 1], 0)
_, G = gradiente(P, ex(V_TR), ey(Y_TR))
w, g, lr, lam = P[2][0, 0], G[2][0, 0], 0.01, 1.0
print(f"(A) w0 = {w:.6f}   g1 = {g:.6f}   η = {lr}   λ = {lam}")
m, v = 0.1 * g, 0.001 * g * g
mh, vh = m / 0.1, v / 0.001
print(f"    m1 = {m:.7f}   v1 = {v:.4e}   m̂1 = {mh:.6f}   √v̂1 = {np.sqrt(vh):.6f}   m̂/(√v̂ + ε) = {mh / (np.sqrt(vh) + 1e-8):.6f}")
dec = lr * lam * w
w_w = w * (1 - lr * lam) - lr * mh / (np.sqrt(vh) + 1e-8)
print(f"    AdamW: w·(1 − ηλ) = {w:.6f}·{1 - lr * lam} = {w * (1 - lr * lam):.6f} (decaimiento {dec:.6f});  paso de Adam {lr * mh / (np.sqrt(vh) + 1e-8):.6f}  →  w1 = {w_w:.6f}")
g2 = g + lam * w
w_l = w - lr * np.sign(g2) * abs(g2) / (abs(g2) + 1e-8)
print(f"    con L2, el gradiente cambia de signo si λ > g/|w| = {g / abs(w):.5f}")
print(f"    Adam + L2: g + λw = {g:.6f} + {lam * w:.6f} = {g2:.6f}  →  paso {w - w_l:.6f}  →  w1 = {w_l:.6f}")
w_s = w - lr * (g + lam * w)
print(f"    SGD + L2: w − η(g + λw) = {w:.6f} − {lr}·{g2:.6f} = {w_s:.6f}   (= w·(1 − ηλ) − ηg = {w * (1 - lr * lam) - lr * g:.6f})")
Q = [p.copy() for p in P]; o = Adam(lr, lam, desacoplado=True); o.step(Q, G)
Q2 = [p.copy() for p in P]; o2 = Adam(lr, lam, desacoplado=False); o2.step(Q2, G)
print(f"    comprobación con mlp.Adam: AdamW {Q[2][0, 0]:.6f}, Adam + L2 {Q2[2][0, 0]:.6f}")

# (B) dropout invertido: la red de la idea 5 (p = 0.3) y la primera máscara de apagadas.json
A = json.load(open("apagadas.json", encoding="utf-8"))
r = entrenar(ex(V_TR), ey(Y_TR), ex(V_VA), ey(Y_VA), epocas=3000, lr=0.01, p=0.3)
P = r["P"]; x = ex([85.0])
h1 = np.tanh(x @ P[0].T + P[1])[0]                              # primera capa oculta, en evaluación
m1, m2 = np.array(A["m"][0][0]), np.array(A["m"][0][1])          # la primera máscara de la idea 5
print(f"(B) viaje a 85 km/h → entrada estandarizada x = {x[0, 0]:.4f};  1/(1 − p) = 1/0.7 = {1 / 0.7:.6f}")
for i in range(5):
    a = P[0][i, 0] * x[0, 0] + P[1][i]
    print(f"    neurona {i + 1}: w = {P[0][i, 0]:.4f}  b = {P[1][i]:.4f}  w·x + b = {a:.4f}  tanh = {h1[i]:.4f}"
          f"  máscara {int(m1[i])}  →  {h1[i] / 0.7 if m1[i] else 0.0:.4f}")
print(f"    encendidas en esta máscara: {int(m1.sum())} de 64 en la 1.ª capa y {int(m2.sum())} de 64 en la 2.ª")
h2 = np.tanh((h1 * m1 / 0.7) @ P[2].T + P[3])
zm = float((h2 * m2 / 0.7) @ P[4][0] + P[5][0])
ze = float((np.tanh(h1 @ P[2].T + P[3]) @ P[4].T + P[5])[0])
print(f"    salida con esta máscara {zm:.4f} → {zm:.4f}·{SY:.4f} + {MY:.3f} = {zm * SY + MY:.2f} L/100 km;"
      f"  en evaluación {ze:.4f} → {ze * SY + MY:.2f} L/100 km")
print(f"    (apagadas.json, a 85 km/h: máscara 1 → {A['f'][0][56]:.2f}; evaluación → {A['fe'][56]:.2f})")
