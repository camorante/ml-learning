import numpy as np, pandas as pd, json
from sklearn.ensemble import HistGradientBoostingRegressor as HGB
from sklearn.metrics import mean_absolute_error as mae
from sim import flota
df = flota()
BASE = ["km", "carga", "ralenti", "pendiente", "temp", "odometro"]
dev = df[df.dia < 330]; prod = df[df.dia >= 330]
def run(split, leak, seed=0):
    feats = BASE + (["litros_manana"] if leak else [])
    rng = np.random.default_rng(seed)
    if split == "azar":
        m = rng.random(len(dev)) < 0.8; tr, te = dev[m], dev[~m]
    else:
        tr, te = dev[dev.dia < 270], dev[dev.dia >= 270]
    mod = HGB(max_iter=300, random_state=0).fit(tr[feats], tr.consumo)
    e_test = mae(te.consumo, mod.predict(te[feats]))
    P = prod[feats].copy()
    if leak: P["litros_manana"] = tr.litros_manana.mean()        # en producción no existe todavía
    # para producción se reentrena con todo el periodo de desarrollo, como se haría de verdad
    mod2 = HGB(max_iter=300, random_state=0).fit(dev[feats], dev.consumo)
    e_prod = mae(prod.consumo, mod2.predict(P))
    return round(e_test, 2), round(e_prod, 2)
res = {f"{s}|{l}": run(s, l) for s in ("azar", "tiempo") for l in (False, True)}
print(res)
json.dump(res, open("hero.json", "w"))
