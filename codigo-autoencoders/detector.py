"""El detector: un autoencoder 64 → 32 → 4 → 32 → 64 entrenado solo con ventanas normales. El umbral es el
percentil 99 del error en 1 000 ventanas normales de validación. Guarda el modelo y los errores."""
import json
import numpy as np
import torch
from datos import X_TR, X_VA, X_TE, Y_TE, TIPOS, FALSA, PERDIDA
from modelos import entrenar, error, metricas

net, curva = entrenar(X_TR, k=4, seed=0)
torch.save(net.state_dict(), "detector.pt")
print(f"parámetros: {sum(p.numel() for p in net.parameters())}")
e_tr, e_va, e_te = error(net, X_TR), error(net, X_VA), error(net, X_TE)
print(f"error medio (m/s²)²: entrenamiento {e_tr.mean():.4f} · validación {e_va.mean():.4f}"
      f" · el ruido solo ya daría {0.15 ** 2:.4f}")
umbral = float(np.quantile(e_va, 0.99))
print(f"umbral = percentil 99 de validación = {umbral:.4f}")
print("prueba, mediana del error y avisos por tipo:")
for k, nm in enumerate(TIPOS):
    e = e_te[Y_TE == k]
    print(f"  {nm:10s} mediana {np.median(e):.4f} · avisa {int((e > umbral).sum()):4d} de {len(e)} ({(e > umbral).mean():.1%})")
malo = Y_TE > 0
for nombre, u in [("percentil 95", np.quantile(e_va, 0.95)), ("percentil 99", umbral), ("percentil 99,9", np.quantile(e_va, 0.999))]:
    m = metricas(e_te, malo, u)
    print(f"{nombre:15s} umbral {u:.4f}: falsas alarmas {m['fp']:3d} · averías cazadas {m['vp']} de 200 · se escapan {m['fn']:2d}"
          f" · precisión {m['precision']:.3f} · recall {m['recall']:.3f} · F1 {m['f1']:.3f}"
          f" · coste {m['fp'] * FALSA + m['fn'] * PERDIDA} €")
json.dump({"umbral": umbral, "e_va": e_va.tolist(), "e_te": e_te.tolist(), "curva": curva},
          open("detector.json", "w", encoding="utf-8"))
