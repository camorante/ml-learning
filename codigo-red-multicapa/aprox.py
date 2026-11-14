import numpy as np
from datos import perfil, KM
from red import sigmoide

v = perfil(KM)                                       # velocidad media cada 500 m en una ruta de 100 km


def ajuste(n):
    """n neuronas ocultas σ((km − cᵢ)/w): escalones repartidos por la ruta (cᵢ y w fijos).
    Solo se ajustan los n + 1 pesos de la salida, por mínimos cuadrados."""
    c = (np.arange(n) + 0.5) * 100 / n; w = 0.35 * 100 / n
    B = np.c_[np.ones_like(KM), sigmoide((KM[:, None] - c) / w)]
    a, *_ = np.linalg.lstsq(B, v, rcond=None)
    r = B @ a - v
    return np.sqrt(np.mean(r ** 2)), np.max(np.abs(r))


print(" n    error medio   error máximo  (km/h)")
for n in (1, 2, 4, 8, 16, 32, 64, 128):
    e, m = ajuste(n)
    print(f"{n:3d}   {e:9.2f}     {m:9.2f}")
