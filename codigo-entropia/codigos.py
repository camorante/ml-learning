import itertools
import numpy as np
from sim import ESTADOS, P_ESTADOS, P_ANUAL
from entropia import entropia, huffman, longitud_media, kraft

c = huffman(P_ESTADOS)
shannon = np.ceil(-np.log2(P_ESTADOS)).astype(int)                 # longitudes de Shannon: ⌈log2 1/p⌉
for e, p, w, ls in zip(ESTADOS, P_ESTADOS, c, shannon):
    print(f"{e:18s} p = {p:.2f}  −log2 p = {-np.log2(p):.3f}  Huffman {w:6s} ({len(w)})  Shannon {ls}")
H = entropia(P_ESTADOS)
print(f"H = {H:.4f}  L_Huffman = {longitud_media(P_ESTADOS, c):.4f}  L_Shannon = {P_ESTADOS @ shannon:.4f}"
      f"  Kraft Huffman = {kraft(c)}  Kraft Shannon = {np.sum(2.0 ** -shannon):.4f}")

# codificar los partes en bloques de n: la media por parte se acerca a H
print(f"partes del año, H = {entropia(P_ANUAL):.4f} bits por parte")
for n in range(1, 6):
    pb = np.array([np.prod(t) for t in itertools.product(P_ANUAL, repeat=n)])   # 4^n bloques
    L = longitud_media(pb, huffman(pb)) / n
    print(f"  bloques de {n}: {len(pb):4d} símbolos  L/n = {L:.4f}  exceso {L - entropia(P_ANUAL):.4f}  (cota 1/n = {1 / n:.3f})")
