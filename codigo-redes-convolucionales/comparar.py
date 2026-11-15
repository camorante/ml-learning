"""Entrenamiento real: red convolucional frente a red densa con las mismas ventanas del acelerómetro.
A: la frenada puede estar en cualquier sitio. B: en el entrenamiento siempre está al principio; se prueba al final.
Cada red se entrena con tres semillas (pesos iniciales y barajado); se guardan las de la semilla 0."""
import os
import numpy as np
import torch
from datos import X_TR, Y_TR, X_VA, Y_VA, X_TE, Y_TE, XB_TR, YB_TR, XB_VA, YB_VA, XB_FI, YB_FI
from modelos import CNN, Densa, UnFiltro, n_param, entrenar, acierto

os.chdir(os.path.dirname(os.path.abspath(__file__)))
print(f"{'red':21s}{'semilla':>8s}{'parámetros':>11s}{'A: prueba':>11s}{'B: principio':>14s}{'B: al final':>13s}")
for nombre, M in [("convolucional", CNN), ("densa 64-32-1", Densa), ("un filtro (juguete)", UnFiltro)]:
    for seed in (0, 1, 2):
        ma, _ = entrenar(M, X_TR, Y_TR, X_VA, Y_VA, seed=seed)
        mb, _ = entrenar(M, XB_TR, YB_TR, XB_VA, YB_VA, seed=seed)
        if seed == (1 if M is UnFiltro else 0):
            torch.save(ma.state_dict(), f"{M.__name__}_A.pt"); torch.save(mb.state_dict(), f"{M.__name__}_B.pt")
        print(f"{nombre:21s}{seed:8d}{n_param(ma):11d}{acierto(ma, X_TE, Y_TE):11.1%}{acierto(mb, XB_VA, YB_VA):14.1%}"
              f"{acierto(mb, XB_FI, YB_FI):13.1%}")
print(f"«decir siempre brusca» acierta {Y_TE.mean():.1%} en A y {YB_FI.mean():.1%} al final de B")
