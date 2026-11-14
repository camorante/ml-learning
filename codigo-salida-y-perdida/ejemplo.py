"""Ejemplo resuelto: un viaje de 180 km por las tres cabezas de la red, con cada pérdida a mano."""
import numpy as np
from datos import VIAJE, SEGURO_Y_EQUIVOCADO, CLASES, COSTE_AVERIA, COSTE_REVISION
import perdidas as P

V = VIAJE
# 1) consumo: salida lineal, error cuadrático (y MAE y Huber para comparar)
e = V["consumo_pred"] - V["consumo_real"]
print(f"1) error = {V['consumo_pred']} − {V['consumo_real']} = {e:.1f}   MSE = {e**2:.2f}   MAE = {abs(e):.2f}"
      f"   Huber(δ=2) = {0.5 * e**2:.2f}   ∂MSE/∂ŷ = {2 * e:.1f}")
litros = 180 * abs(e) / 100
print(f"   en el viaje: 180 km × {abs(e):.1f}/100 = {litros:.2f} L  →  × 1,60 €/L = {litros * 1.60:.2f} €")
# 2) avería: sigmoide y entropía cruzada binaria
z = V["z_averia"]; p = 1 / (1 + np.exp(-z))
print(f"2) e^1.2 = {np.exp(1.2):.4f}   p = 1/(1 + e^1.2) = {p:.4f}   BCE = −ln p = {-np.log(p):.4f}"
      f"   (si no hubiera avería: −ln(1 − p) = {-np.log(1 - p):.4f})   ∂/∂z = p − 1 = {p - 1:.4f}")
print(f"   con BCEWithLogits: softplus(1.2) = ln(1 + e^1.2) = {np.log1p(np.exp(1.2)):.4f}")
# 3) tipo de viaje: softmax y entropía cruzada
zk = np.array(V["logits"]); ex = np.exp(zk); q = ex / ex.sum()
print(f"3) e^z = {ex.round(4)}   suma {ex.sum():.4f}   softmax = {q.round(4)}")
print(f"   CE = −ln {q[1]:.4f} = {-np.log(q[1]):.4f}   logsumexp = {np.log(ex.sum()):.4f} → {np.log(ex.sum()):.4f} − {zk[1]} = "
      f"{np.log(ex.sum()) - zk[1]:.4f}   ∂/∂z = p − y = {(q - np.eye(3)[1]).round(4)}")
print(f"   restando el máximo (1.6): e^(z − 1.6) = {np.exp(zk - 1.6).round(4)}, suma {np.exp(zk - 1.6).sum():.4f}")
# 4) la pérdida total de las tres cabezas
L = [e**2, -np.log(p), -np.log(q[1])]
print(f"4) total = {L[0]:.2f} + {L[1]:.3f} + {L[2]:.3f} = {sum(L):.3f};  la cabeza del consumo aporta {L[0] / sum(L):.0%}")
Ls = [(e / 6)**2, L[1], L[2]]
print(f"   con el consumo estandarizado (÷ 6 L/100 km): {Ls[0]:.3f} + {L[1]:.3f} + {L[2]:.3f} = {sum(Ls):.3f}")
# 5) seguro y equivocado
S = SEGURO_Y_EQUIVOCADO
p2 = P.sigmoide(S["z_averia"]); q2 = P.softmax(S["logits"])
print(f"5) z = {S['z_averia']}: p = {p2:.4f} → BCE {-np.log(p2):.3f} ({-np.log(p2) / -np.log(p):.1f} veces más);"
      f"  logits {S['logits']}: p(regional) = {q2[1]:.4f} → CE {-np.log(q2[1]):.3f};  empujones {p2 - 1:.3f} y {(q2 - np.eye(3)[1]).round(3)}")
# 6) pesos de clase y el umbral de aviso
w = COSTE_AVERIA / COSTE_REVISION
print(f"6) w = {COSTE_AVERIA}/{COSTE_REVISION} = {w:.2f}   BCE ponderada = {w * -np.log(p):.2f}   umbral 1/(1+w) = {1 / (1 + w):.4f}")
print(f"   p = {p:.4f} ≥ {1 / (1 + w):.4f} → avisar; coste esperado sin avisar {p * COSTE_AVERIA:.0f} € frente a revisión {COSTE_REVISION} €")
