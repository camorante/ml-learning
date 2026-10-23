import math
import numpy as np
from sim import P_ANUAL, P_INVIERNO
from entropia import sorpresa, entropia, huffman, longitud_media, entropia_cruzada, kl, softmax, log_loss_binaria

print("1) sorpresas (bits):", np.round(sorpresa(P_ANUAL), 4))
print("2) aportaciones p·s:", np.round(P_ANUAL * sorpresa(P_ANUAL), 4), " H =", round(entropia(P_ANUAL), 4))
c = huffman(P_ANUAL)
print("3) Huffman:", c, " L =", round(longitud_media(P_ANUAL, c), 4), " fijo = 2")
print("4) invierno con el código del año: −log2 q =", np.round(sorpresa(P_ANUAL), 4), " p·(−log2 q) =", np.round(P_INVIERNO * sorpresa(P_ANUAL), 4))
print("   H(p) =", round(entropia(P_INVIERNO), 4), " H(p,q) =", round(entropia_cruzada(P_INVIERNO, P_ANUAL), 4),
      " KL =", round(kl(P_INVIERNO, P_ANUAL), 4), " KL al revés =", round(kl(P_ANUAL, P_INVIERNO), 4))
print("   longitud con los códigos de Huffman del año:", round(longitud_media(P_INVIERNO, c), 4), "bits")
y = np.array([1, 0, 0, 1]); p = np.array([0.80, 0.30, 0.10, 0.01])
q = np.where(y == 1, p, 1 - p)
print("5) prob. dada a lo que pasó:", q, " −ln:", np.round(-np.log(q), 4), " media =", round(log_loss_binaria(y, p), 4))
print("   sin el cuarto camión:", round(log_loss_binaria(y[:3], p[:3]), 4), " parte del total que pone el cuarto:", round(-math.log(0.01) / -np.log(q).sum(), 3))
print("   error al cuadrado:", np.round((y - p) ** 2, 4), " media =", round(float(np.mean((y - p) ** 2)), 4))
z = np.array([2.0, 0.5, -0.5, 0.0])
print("6) e^z =", np.round(np.exp(z), 4), " suma =", round(np.exp(z).sum(), 4), " softmax =", np.round(softmax(z), 4))
print("   pérdida (frenos) = −ln", round(softmax(z)[1], 4), "=", round(-math.log(softmax(z)[1]), 4), " p − y =", np.round(softmax(z) - np.eye(4)[1], 4))
H2 = lambda n, a: entropia([a, n - a])
g = H2(1000, 251) - 0.202 * H2(202, 160) - 0.798 * H2(798, 91)
print(f"7) H(251/1000) = {H2(1000, 251):.4f}  H(160/202) = {H2(202, 160):.4f}  H(91/798) = {H2(798, 91):.4f}"
      f"  media = {0.202 * H2(202, 160) + 0.798 * H2(798, 91):.4f}  ganancia = {g:.4f}")
