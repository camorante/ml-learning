import numpy as np, pandas as pd, json
from sim import descargas
from miss import imputar_media, imputar_grupo_sorteo, imputar_regresion
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

d = descargas(); y = d.espera.values; g = d.cliente.values
Z = np.column_stack([d.palets, (g == "tienda pequeña"), (g == "obra")]).astype(float)
verdad = (y.mean(), (y > 60).mean(), y.std(ddof=1))
print("verdad: media %.1f  >60 %.3f  desv %.1f" % verdad)
tabla = {}
for mec in ["mcar", "mar", "mnar"]:
    m = d["falta_" + mec].values; x = np.where(m, np.nan, y)
    met = {"quitar": x[~m], "media": imputar_media(x), "grupo": imputar_grupo_sorteo(x, g, 1),
           "regresion": imputar_regresion(x, Z), "regresion_ruido": imputar_regresion(x, Z, ruido=True, seed=1)}
    tabla[mec] = {k: (round(float(v.mean()), 1), round(float((v > 60).mean()), 3), round(float(v.std(ddof=1)), 1),
                      round(float(np.corrcoef(d.palets.values[~m] if k == "quitar" else d.palets.values, v)[0, 1]), 3)) for k, v in met.items()}
    tabla[mec]["falta"] = round(float(m.mean()), 3)
    print(mec, tabla[mec])
print("r verdad", round(np.corrcoef(d.palets, y)[0, 1], 3))
# --- avería: el hueco como información ---
t = np.where(d.falta_temp, np.nan, d.temp_motor.values); a = d.averia_30d.values; ind = d.falta_temp.values.astype(float)
tr, te = train_test_split(np.arange(len(d)), test_size=0.4, random_state=0, stratify=a)
def auc_log(X):
    mu = np.nanmean(X[tr, 0]); Xf = X.copy(); Xf[np.isnan(Xf[:, 0]), 0] = mu         # media aprendida solo con entrenamiento
    mdl = LogisticRegression(max_iter=1000).fit(Xf[tr], a[tr]); return roc_auc_score(a[te], mdl.predict_proba(Xf[te])[:, 1])
res = {"media": (round(auc_log(t.reshape(-1, 1)), 3),), "media_indicador": (round(auc_log(np.column_stack([t, ind])), 3),)}
h = HistGradientBoostingClassifier(max_iter=50, learning_rate=0.05, max_depth=2, random_state=0).fit(t[tr].reshape(-1, 1), a[tr])
res["arbol_nativo"] = (round(roc_auc_score(a[te], h.predict_proba(t[te].reshape(-1, 1))[:, 1]), 3),)
print(res)
print("tasa avería: falta %.3f  hay dato %.3f" % (a[d.falta_temp].mean(), a[~d.falta_temp].mean()), "falta temp %.3f" % d.falta_temp.mean())
json.dump(dict(tabla=tabla, verdad=[round(v, 3) for v in verdad], averia=res,
               tasas=[round(float(a[d.falta_temp].mean()), 3), round(float(a[~d.falta_temp].mean()), 3)]), open("../res.json", "w"))
