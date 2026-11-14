"""Comprobación de gradiente: retropropagación frente a diferencias centrales, con el error relativo
e = ‖a − b‖ / max(‖a‖, ‖b‖). Y lo que da el mismo test si se olvida la pendiente φ'(z) de una capa."""
import numpy as np
from datos import carga_consumo, red_inicial
from retro import Densa, MLP, gradiente_vector, diferencias_centrales, error_relativo, ACT

X, y = carga_consumo()
red = red_inicial()
a, b = gradiente_vector(red, X, y), diferencias_centrales(red, X, y)
print(f"red 2-2-1 de los 30 viajes, 9 parámetros: e = {error_relativo(a, b):.1e}")

rng = np.random.default_rng(7)
tam = [2, 16, 16, 1]
r2 = MLP([Densa(rng.normal(0, 1 / np.sqrt(p), (q, p)), rng.normal(0, .1, q), "tanh" if i < 2 else "identidad")
          for i, (p, q) in enumerate(zip(tam, tam[1:]))], "bce")
Xs = (X - X.mean(0)) / X.std(0)                     # entradas estandarizadas (tema D7)
for h in (1e-2, 1e-4, 1e-6, 1e-8, 1e-10):
    print(f"red 2-16-16-1 ({len(r2.vector())} parámetros), h = {h:.0e}: e = "
          f"{error_relativo(gradiente_vector(r2, Xs, y), diferencias_centrales(r2, Xs, y, h)):.1e}")

ACT["tanh_mal"] = (np.tanh, lambda z, h: np.ones_like(z))          # error típico: sin φ'(z)
r2.capas[1].act = "tanh_mal"
g_mal = gradiente_vector(r2, Xs, y); r2.capas[1].act = "tanh"
print(f"con la pendiente de la capa 2 olvidada: e = {error_relativo(g_mal, diferencias_centrales(r2, Xs, y)):.1e}")
