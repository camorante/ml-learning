import numpy as np, pandas as pd, json
from sklearn.ensemble import HistGradientBoostingRegressor as HGB
from sklearn.metrics import mean_absolute_error as mae
from sklearn.model_selection import GroupShuffleSplit
from sim import flota
df = flota().sort_values(["camion", "dia"]).reset_index(drop=True)
df["consumo_ayer"] = df.groupby("camion").consumo.shift(1)
BASE = ["km", "carga", "ralenti", "pendiente", "temp", "odometro"]
out = {}
# --- grupos: camiones nuevos ---
nuevos = df.camion >= 48; D = df[~nuevos & (df.dia < 330)]; N = df[nuevos & (df.dia < 330)]
rng = np.random.default_rng(1); m = rng.random(len(D)) < 0.8
mod = HGB(max_iter=300, random_state=0).fit(D[m][BASE], D[m].consumo)
e_rand = mae(D[~m].consumo, mod.predict(D[~m][BASE]))
tr_i, te_i = next(GroupShuffleSplit(1, test_size=0.2, random_state=0).split(D, groups=D.camion))
mod = HGB(max_iter=300, random_state=0).fit(D.iloc[tr_i][BASE], D.iloc[tr_i].consumo)
e_grp = mae(D.iloc[te_i].consumo, mod.predict(D.iloc[te_i][BASE]))
modall = HGB(max_iter=300, random_state=0).fit(D[BASE], D.consumo)
e_new = mae(N.consumo, modall.predict(N[BASE]))
out["grupos"] = dict(azar=round(e_rand, 2), camion=round(e_grp, 2), nuevos=round(e_new, 2))
# sin odómetro (la variable que delata al camión)
B2 = [f for f in BASE if f != "odometro"]
mod = HGB(max_iter=300, random_state=0).fit(D[m][B2], D[m].consumo); e_rand2 = mae(D[~m].consumo, mod.predict(D[~m][B2]))
modall2 = HGB(max_iter=300, random_state=0).fit(D[B2], D.consumo); e_new2 = mae(N.consumo, modall2.predict(N[B2]))
out["grupos_sin_odo"] = dict(azar=round(e_rand2, 2), nuevos=round(e_new2, 2))
# --- candidatas de variables (división temporal) ---
dev = df[df.dia < 330].copy(); prod = df[df.dia >= 330].copy()
dev["media_camion"] = dev.groupby("camion").consumo.transform("mean")     # calculada con TODO el periodo (incluye la prueba)
prod["media_camion"] = prod.camion.map(dev.groupby("camion").consumo.mean())
cands = {"ninguna": [], "consumo_ayer": ["consumo_ayer"], "media_camion": ["media_camion"], "litros_manana": ["litros_manana"]}
res = {}
for k, extra in cands.items():
    f = BASE + extra; tr, te = dev[dev.dia < 270], dev[dev.dia >= 270]
    mt = HGB(max_iter=300, random_state=0).fit(tr[f], tr.consumo); et = mae(te.consumo, mt.predict(te[f]))
    mp = HGB(max_iter=300, random_state=0).fit(dev[f], dev.consumo); P = prod[f].copy()
    if k == "litros_manana": P["litros_manana"] = dev.litros_manana.mean()
    ep = mae(prod.consumo, mp.predict(P)); res[k] = (round(et, 2), round(ep, 2))
out["candidatas"] = res
# --- fuga en la preparación: elegir variables con todos los datos ---
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import SelectKBest, f_classif
bad, good = [], []
for s in range(40):
    r = np.random.default_rng(100 + s); X = r.normal(size=(100, 2000)); y = r.integers(0, 2, 100)
    cv = StratifiedKFold(5, shuffle=True, random_state=s)
    Xs = SelectKBest(f_classif, k=20).fit_transform(X, y)                          # ¡con todos los datos!
    bad.append(cross_val_score(LogisticRegression(max_iter=2000), Xs, y, cv=cv).mean())
    pipe = make_pipeline(SelectKBest(f_classif, k=20), LogisticRegression(max_iter=2000))
    good.append(cross_val_score(pipe, X, y, cv=cv).mean())
out["seleccion"] = dict(fuera=[round(v, 3) for v in bad], dentro=[round(v, 3) for v in good])
print({k: v for k, v in out.items() if k != "seleccion"}, "sel", np.mean(bad), np.mean(good))
json.dump(out, open("ideas.json", "w"))
