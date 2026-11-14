"""Genera ../embed.json y ../tembed.json (datos de los juguetes) e imprime las cifras que muestran."""
import json, os
import numpy as np
from sklearn.linear_model import LogisticRegression
from datos import *
import perdidas as P

os.chdir(os.path.dirname(os.path.abspath(__file__)))
V = VIAJE
print("juguete, valores del ejemplo:")
print(f"  consumo {V['consumo_pred']} vs {V['consumo_real']}: error² = {(V['consumo_pred'] - V['consumo_real'])**2:.2f}")
p = P.sigmoide(V["z_averia"])
print(f"  avería z = {V['z_averia']}: p = {p:.4f}, BCE si hubo avería = {-np.log(p):.3f}, si no = {-np.log(1 - p):.3f}")
q = P.softmax(V["logits"])
print(f"  tipo: softmax = {q.round(4)}, CE si era regional = {-np.log(q[1]):.4f}, urbano {-np.log(q[0]):.3f}, autovía {-np.log(q[2]):.3f}")
S = SEGURO_Y_EQUIVOCADO
p2, q2 = P.sigmoide(S["z_averia"]), P.softmax(S["logits"])
print(f"seguro y equivocado: error² {(S['consumo_pred'] - V['consumo_real'])**2:.1f}, p = {p2:.4f} → BCE {-np.log(p2):.3f};"
      f" softmax {q2.round(4)} → CE (regional) {-np.log(q2[1]):.3f}")

pl, yl = np.array(P_LOTE), np.array(Y_LOTE)
per = -(yl * np.log(pl) + (1 - yl) * np.log(1 - pl))
print("lote de 10:", per.round(3), f"media {per.mean():.3f}; sin el décimo {per[:9].mean():.3f}; el décimo pesa {per[9] / per.sum():.0%}")

x40, y40 = camiones(40, seed=4)
u40 = (x40 - 30) / 30
m = LogisticRegression(C=1e6).fit(u40[:, None], y40)
b40, w40 = float(m.intercept_[0]), float(m.coef_[0, 0])
ws = np.linspace(-8, 10, 1801)
ce = [P.bce_logits(b40 + w * u40, y40) for w in ws]
ms = [P.mse_sobre_p(b40 + w * u40, y40)[0] for w in ws]
print(f"paisaje (b fijo = {b40:.3f}): CE mínima {min(ce):.4f} en w = {ws[np.argmin(ce)]:.2f};"
      f" MSE sobre p mínima {min(ms):.4f} en w = {ws[np.argmin(ms)]:.2f}")
for w in (-6, -3, 0, 2.45, 6):
    g_ce = P.d_bce_logits(b40 + w * u40, y40) @ u40
    g_ms = P.mse_sobre_p(b40 + w * u40, y40)[1] @ u40
    print(f"  w = {w:5}: CE {P.bce_logits(b40 + w * u40, y40):.4f} (pendiente {g_ce:+.4f})"
          f"   MSE {P.mse_sobre_p(b40 + w * u40, y40)[0]:.4f} (pendiente {g_ms:+.4f})")

xe, ye = camiones(2000, seed=100); xp, yp = camiones(2000, seed=101)
m1 = LogisticRegression(C=1e6).fit(((xe - 30) / 30)[:, None], ye)
bc, wc = float(m1.intercept_[0]), float(m1.coef_[0, 0])

E = dict(viaje=V, seguro=S, clases=CLASES, ruta=RUTA, atipico=ATIPICO, p_lote=P_LOTE, y_lote=Y_LOTE,
         x40=x40.tolist(), y40=y40.tolist(), b40=round(b40, 4), w40=round(w40, 4),
         xp=xp.tolist(), yp=yp.tolist(), bc=round(bc, 4), wc=round(wc, 4),
         c_averia=COSTE_AVERIA, c_revision=COSTE_REVISION)
json.dump(E, open("../embed.json", "w", encoding="utf-8"), ensure_ascii=False, separators=(",", ":"))
json.dump(dict(x40=E["x40"], y40=E["y40"], b40=E["b40"], w40=E["w40"]),
          open("../tembed.json", "w", encoding="utf-8"), separators=(",", ":"))
print("embed.json", os.path.getsize("../embed.json"), "bytes")
