"""El código como mapa: el mismo autoencoder con un código de 2 números (64 → 32 → 2 → 32 → 64), entrenado solo con
ventanas normales. ¿Dónde caen las normales y dónde las averías? Guarda los puntos y el decodificador para el dibujo."""
import json
import numpy as np
import torch
from datos import X_TR, X_TE, Y_TE, TIPOS, normales
from modelos import entrenar, codigo, error
from sklearn.neighbors import NearestNeighbors

net, _ = entrenar(X_TR, k=2, seed=0)
torch.save(net.state_dict(), "mapa.pt")
X_VA, P_VA = normales(1000, seed=122, parametros=True)          # las mismas 1 000 de validación, con sus parámetros
z_tr, z_va, z_te = codigo(net, X_TR), codigo(net, X_VA), codigo(net, X_TE)
A, f, fase = P_VA.T
fm = (2 * np.pi * f * 3.15 + fase) % (2 * np.pi)                 # fase del rebote en el centro de la ventana (t = 3,15 s)
r, ang = np.hypot(*z_va.T), np.arctan2(z_va[:, 1], z_va[:, 0])
print(f"normales: el código ocupa x de {z_va[:, 0].min():.2f} a {z_va[:, 0].max():.2f}, y de {z_va[:, 1].min():.2f} a {z_va[:, 1].max():.2f}")
print(f"correlación entre la distancia al centro y la amplitud del rebote: {np.corrcoef(r, A)[0, 1]:.3f}")
print(f"correlación circular entre el ángulo y la fase en el centro de la ventana: {abs(np.mean(np.exp(1j * (ang - fm)))):.3f}")
# ¿caen fuera? distancia media a los 5 vecinos normales de entrenamiento más cercanos, frente a su percentil 99 en validación
vec = NearestNeighbors(n_neighbors=5).fit(z_tr)
d_va, d_te = vec.kneighbors(z_va)[0].mean(1), vec.kneighbors(z_te)[0].mean(1)
u_d = np.quantile(d_va, 0.99)
e_va, e_te = error(net, X_VA), error(net, X_TE)
u_e = np.quantile(e_va, 0.99)
for k, nm in enumerate(TIPOS):
    s = Y_TE == k
    print(f"  {nm:10s} fuera de la nube normal en el mapa: {int((d_te[s] > u_d).sum()):4d} de {s.sum()}"
          f" · error por encima del percentil 99: {int((e_te[s] > u_e).sum()):4d} de {s.sum()}")
sd = {k: v.numpy().round(6).tolist() for k, v in net.state_dict().items() if k.startswith("dec")}
sel = np.r_[np.flatnonzero(Y_TE == 0)[:300], np.flatnonzero(Y_TE > 0)]
json.dump({"z": z_te[sel].round(4).tolist(), "y": Y_TE[sel].tolist(), "e": e_te[sel].round(5).tolist(), "idx": sel.tolist(),
           "u_e": float(u_e), "dec": sd}, open("mapa.json", "w", encoding="utf-8"))
