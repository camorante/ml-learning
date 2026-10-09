import numpy as np, pandas as pd, json
from sklearn.ensemble import HistGradientBoostingRegressor as HGB
from sklearn.metrics import mean_absolute_error as mae
from sim import flota
df = flota().sort_values(["camion","dia"]).reset_index(drop=True)
g = df.groupby("camion").consumo
df["consumo_ayer"] = g.shift(1)
df["media_semana"] = g.transform(lambda s: s.rolling(7, center=True, min_periods=1).mean())  # incluye 3 días futuros y el propio
df["media_7_pasado"] = g.transform(lambda s: s.shift(1).rolling(7, min_periods=1).mean())
BASE = ["km","carga","ralenti","pendiente","temp","odometro"]
dev = df[df.dia<330]; prod = df[df.dia>=330]
res={}
for k in ["ninguna","consumo_ayer","media_7_pasado","media_semana","litros_manana"]:
    f = BASE + ([] if k=="ninguna" else [k]); tr, te = dev[dev.dia<270], dev[dev.dia>=270]
    et = mae(te.consumo, HGB(max_iter=300,random_state=0).fit(tr[f],tr.consumo).predict(te[f]))
    mp = HGB(max_iter=300,random_state=0).fit(dev[f],dev.consumo); P = prod[f].copy()
    # en producción, lo que no se sabe todavía: la media centrada solo con los días pasados disponibles; litros de mañana -> media
    if k=="media_semana": P["media_semana"] = prod.media_7_pasado
    if k=="litros_manana": P["litros_manana"] = dev.litros_manana.mean()
    res[k]=(round(et,2), round(mae(prod.consumo, mp.predict(P)),2))
print(res)
d=json.load(open("ideas.json")); d["candidatas"]=res; json.dump(d,open("ideas.json","w"))
