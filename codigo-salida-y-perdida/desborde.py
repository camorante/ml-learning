"""Softmax ingenuo frente al estable: desbordamiento, invariancia al desplazamiento y temperatura."""
import warnings
import numpy as np
from perdidas import softmax, logsumexp

warnings.simplefilter("always")

def softmax_ingenuo(z):
    e = np.exp(z); return e / e.sum()

z = np.array([0.4, 1.6, 0.9])                       # urbano, regional, autovía (el viaje del juguete)
print("logits", z, "→ softmax", softmax(z).round(4))
print("  + 100 a los tres:", softmax(z + 100).round(4), " (sumar lo mismo no cambia nada)")
for T in (0.5, 1, 2, 5):
    print(f"  temperatura T = {T}: {softmax(z, T).round(4)}  (el orden se mantiene)")
for desplaz in (700, 710, 1000):
    with warnings.catch_warnings(record=True) as w:
        r = softmax_ingenuo(z + desplaz)
    print(f"ingenuo con z + {desplaz}: {r}  avisos {[x.category.__name__ for x in w]}")
print("estable con z + 1000:", softmax(z + 1000).round(4))
print("ingenuo con z − 800:", end=" ")
with warnings.catch_warnings(record=True) as w:
    print(softmax_ingenuo(z - 800), [x.category.__name__ for x in w])
z32 = (z + 89).astype(np.float32)
with warnings.catch_warnings(record=True) as w:
    print("float32 con z + 89:", softmax_ingenuo(z32), [x.category.__name__ for x in w])
with warnings.catch_warnings(record=True) as w:
    ing = np.log(np.sum(np.exp(z + 1000)))
print(f"logsumexp(z + 1000) = {logsumexp(z + 1000):.4f}   ingenuo: {ing}  avisos {[x.category.__name__ for x in w]}")
