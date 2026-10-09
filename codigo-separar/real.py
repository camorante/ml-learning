import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.model_selection import KFold, GroupKFold, cross_val_score, cross_val_predict
from sklearn.metrics import mean_absolute_error, roc_auc_score
from sim import flota
from splits import date_split, mae_interval

df = flota(); X_COLS = ["km", "carga", "ralenti", "pendiente", "temp", "odometro"]
dev, prod = df[df.dia < 330].reset_index(drop=True), df[df.dia >= 330].reset_index(drop=True)
X, y = dev[X_COLS], dev.consumo
modelo = HistGradientBoostingRegressor(max_iter=300, random_state=0)

esquemas = {
    "KFold mezclado": KFold(5, shuffle=True, random_state=0).split(X),
    "GroupKFold (camión)": GroupKFold(5).split(X, groups=dev.camion),
    "por fecha, hueco 7 d": date_split(dev.dia, k=5, gap=7),
}
for nombre, cv in esquemas.items():
    s = -cross_val_score(modelo, X, y, cv=list(cv), scoring="neg_mean_absolute_error")
    print(f"{nombre:22s} MAE {s.mean():.2f} ± {s.std():.2f}")

final = modelo.fit(X, y)
err = np.abs(prod.consumo - final.predict(prod[X_COLS]))
print(f"{'producción (real)':22s} MAE {err.mean():.2f}")
lo, hi = mae_interval(err); lo2, hi2 = mae_interval(err, bloques=prod.camion)
print(f"IC 95 % ingenuo [{lo:.2f}, {hi:.2f}]   por camión [{lo2:.2f}, {hi2:.2f}]")

# validación adversaria: ¿se distingue el desarrollo de la producción?
both = np.r_[np.zeros(len(dev)), np.ones(len(prod))]; A = np.vstack([dev[X_COLS], prod[X_COLS]])
for cols in (X_COLS, [c for c in X_COLS if c != "odometro"]):
    Ac = np.vstack([dev[cols], prod[cols]])
    p = cross_val_predict(HistGradientBoostingClassifier(random_state=0), Ac, both,
                          cv=KFold(5, shuffle=True, random_state=0), method="predict_proba")[:, 1]
    print(f"AUC adversaria con {len(cols)} columnas: {roc_auc_score(both, p):.3f}")
