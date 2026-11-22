"""El ejemplo resuelto: un camión del examen pasa por la red entrenada y por la regla de decisión en euros."""
import json
import numpy as np
import torch
from comun import trozos
from datos import MODELOS, DEPOS, CANALES
from red import Red, logits
from metricas import C_REV, C_AVE, UMBRAL

D, F, E, esc = trozos()
te, P = F["te"], np.load("probs.npz")
p = P["red completa|0"]
# el primer camión del examen (por número) con avería avisada y probabilidad entre 0,2 y 0,4
i = int(np.flatnonzero((te["y"] == 1) & (te["vis"] == 1) & (p > 0.2) & (p < 0.4))[0])
c, mes = te["cam"][i], te["mes"][i]
print(f"camión {c}, mes {mes}: modelo {MODELOS[te['mod'][i]]}, depósito {DEPOS[te['dep'][i]]}")
t = te["tab"][i]
print(f"tabla: edad {t[0]:.2f} años, {t[1]:.1f} miles de km, carga {t[2]:.2f} t, {t[3]:.2f} miles de km en el mes, {t[4]:.0f} averías en los 12 meses anteriores")
print("tabla escalada:", np.round(E["te"]["tab"][i], 3), " (media del entrenamiento", np.round(esc["tab"].m[0], 2),
      ", desviación", np.round(esc["tab"].d[0], 2), ")")
X = te["X"][i]
for k, nm in enumerate(CANALES):
    print(f"{nm:17s} 21 primeros días {X[:21, k].mean():7.2f}   últimos 7 {X[21:, k].mean():7.2f}   máximo {X[:, k].max():7.2f}")
m = Red(); m.load_state_dict(torch.load("modelo.pt"))
b = {k: torch.as_tensor(v[i:i + 1]) for k, v in E["te"].items()}
m.eval()
with torch.no_grad():
    partes = m.partes(b)[0].numpy()
    zr = float(m(b)[0])
print("embedding del modelo:", np.round(partes[16:19], 3), " del depósito:", np.round(partes[19:22], 3))
print("estado de la GRU tras el día 28:", np.round(partes[:16], 2))
ytr = F["tr"]["y"]
npos, nneg = int(ytr.sum()), int(len(ytr) - ytr.sum())
w = json.load(open("comparar.json", encoding="utf-8"))["curvas"]["red completa|0"]["w"]
print(f"entrenamiento: {nneg} fotos sanas y {npos} con avería → w = {nneg}/{npos} = {w:.4f}, ln w = {np.log(w):.4f}")
z = zr - np.log(w)
q = 1 / (1 + np.exp(-z))
print(f"logit de la red {zr:.4f} → logit corregido {zr:.4f} − {np.log(w):.4f} = {z:.4f} → p = {q:.4f}")
print(f"sin corregir: p = {1 / (1 + np.exp(-zr)):.4f}")
print(f"umbral: {C_REV:.0f} / {C_AVE:.0f} = {UMBRAL:.2f}")
print(f"coste esperado si avisamos: {C_REV:.0f} €;  si no avisamos: {q:.4f} × {C_AVE:.0f} = {q * C_AVE:.1f} €")
print(f"decisión: {'AVISAR' if q > UMBRAL else 'no avisar'}; lo que pasó: {'avería' if te['y'][i] else 'sin avería'}"
      f" (tipo {['rodamiento', 'refrigeración', 'aceite'][te['tipo'][i]]})")
print(f"mismo camión con el boosting: p = {P['boosting'][i]:.4f}; con la regresión logística: p = {P['logistica'][i]:.4f}")
json.dump(dict(i=i, cam=int(c), mes=int(mes), X=np.round(X, 2).tolist()), open("ejemplo.json", "w", encoding="utf-8"))
