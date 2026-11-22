"""Autoencoder de eliminación de ruido (denoising): la entrada lleva ruido extra (σ = 0,3 m/s²) y el objetivo es la
ventana sin ese ruido extra. Frente al detector normal, con un acelerómetro el doble de ruidoso."""
import json
import numpy as np
import torch
from datos import X_TR, X_TE, Y_TE, TIPOS, FS, t, normales
from modelos import entrenar, reconstruir, error, AE

X_VA, P = normales(1000, seed=122, parametros=True)
LIMPIA = np.array([A * np.sin(2 * np.pi * f * t / FS + fa) for A, f, fa in P])   # el rebote de verdad, sin ruido
rng = np.random.default_rng(1200)
X_SUCIA = X_VA + rng.normal(0, 0.3, X_VA.shape)                                  # la misma ventana con ruido extra

base = AE(4); base.load_state_dict(torch.load("detector.pt"))
dae, _ = entrenar(X_TR, k=4, seed=0, ruido=0.3)
torch.save(dae.state_dict(), "denoising.pt")
ancho, _ = entrenar(X_TR, k=64, m=128, seed=0)               # sin embudo: 64 → 128 → 64 → 128 → 64
dae_ancho, _ = entrenar(X_TR, k=64, m=128, seed=0, ruido=0.3)
mse = lambda A, B: float(((A - B) ** 2).mean())
print(f"entrada con ruido extra frente al rebote de verdad: {mse(X_SUCIA, LIMPIA):.4f}")
MODELOS = [("código de 4, normal   ", base), ("código de 4, denoising", dae),
           ("código de 64, normal  ", ancho), ("código de 64, denoising", dae_ancho)]
for nm, m in MODELOS:
    print(f"{nm}: salida frente al rebote de verdad {mse(reconstruir(m, X_SUCIA), LIMPIA):.4f}"
          f" (con la entrada original: {mse(reconstruir(m, X_VA), LIMPIA):.4f})")
for nm, m in MODELOS:
    e_va, e_te = error(m, X_VA), error(m, X_TE)
    u = np.quantile(e_va, 0.99)
    print(f"{nm}: umbral {u:.4f} · avisa " + " · ".join(f"{TIPOS[k]} {int((e_te[Y_TE == k] > u).sum())}" for k in range(5)))
i = 3
json.dump({"sucia": X_SUCIA[i].round(4).tolist(), "limpia": LIMPIA[i].round(4).tolist(), "orig": X_VA[i].round(4).tolist(),
           "base": reconstruir(base, X_SUCIA[i])[0].round(4).tolist(), "dae": reconstruir(dae, X_SUCIA[i])[0].round(4).tolist(),
           "ancho": reconstruir(ancho, X_SUCIA[i])[0].round(4).tolist(), "dae_ancho": reconstruir(dae_ancho, X_SUCIA[i])[0].round(4).tolist()},
          open("denoising.json", "w", encoding="utf-8"))
