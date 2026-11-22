"""El juguete: embeddings de 2 números para los 20 códigos de avería, aprendidos con la tarea «visto un código,
¿qué otro código hay en la misma visita?». Guarda fotos de la tabla durante el entrenamiento (mapa.json)."""
import json, os
import numpy as np
import torch
from datos import V, NOMBRES, SISTEMAS, SISTEMA, P_TR, P_TE, VIS_TR, VIS_TE
from modelos import Emb, Tabla, entrenar, n_param, coseno, vecinos, pureza

os.chdir(os.path.dirname(os.path.abspath(__file__)))
FOTOS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 30, 40, 60]

torch.manual_seed(0)
m = Emb(V, 2)
hist, snaps = entrenar(m, P_TR, P_TE, epocas=60, lr=0.02, lote=64, seed=0, fotos=FOTOS)
snaps = {ep: np.round(E, 4) for ep, E in snaps.items()}    # lo que ve el juguete: 4 decimales
print(f"{len(VIS_TR)} visitas de entrenamiento ({len(P_TR)} pares) · {len(VIS_TE)} de prueba ({len(P_TE)} pares)")
print(f"modelo: embedding 20×2 + capa lineal 2→20 = {n_param(m)} parámetros · Adam 0,02, lotes de 64, 60 épocas, semilla 0")
print("época  pérdida entren.  pérdida prueba  top-3 prueba  vecinos del mismo sistema")
for ep in FOTOS:
    _, ctr, _, cte, t3 = hist[ep]
    ok, tot = pureza(snaps[ep], SISTEMA)
    print(f"{ep:5d}  {ctr:15.3f}  {cte:14.3f}  {t3:12.1%}  {ok:2d} de {tot}")

E = snaps[60]
S = coseno(E)
print("\nfinal: cada código, su ángulo y sus 3 vecinos por coseno")
for i in range(V):
    ang = np.degrees(np.arctan2(E[i, 1], E[i, 0]))
    vs = vecinos(S, i)
    print(f"  {NOMBRES[i]} ({SISTEMAS[SISTEMA[i]]:9s}) ({E[i, 0]:6.2f}, {E[i, 1]:6.2f}) ángulo {ang:7.1f}°  →  "
          + ", ".join(f"{NOMBRES[j]} {S[i, j]:.3f}" for j in vs))

json.dump({"fotos": FOTOS, "E": {ep: snaps[ep].tolist() for ep in FOTOS},
           "hist": [[round(v, 4) for v in h] for h in hist],
           "out_w": m.out.weight.detach().numpy().tolist(), "out_b": m.out.bias.detach().numpy().tolist()},
          open("mapa.json", "w", encoding="utf-8"))
torch.save(m.state_dict(), "mapa.pt")
