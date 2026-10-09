import numpy as np, pandas as pd, json
from sim import viajes
from prep import ciclo, media_pasada
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder, TargetEncoder, FunctionTransformer, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error as MAE

d = viajes()
d["hist30"] = media_pasada(d.camion, d.dia, d.consumo, 30)
tr = d.dia < 160; te = ~tr
d["hist30"] = d.hist30.fillna(d.consumo[tr].mean())
# ventana centrada de ±15 días que incluye el propio viaje (lo que NO se puede hacer en producción)
cen = np.empty(len(d))
for c, g in d.groupby("camion"):
    t = g.dia.values; v = g.consumo.values
    cen[g.index] = [v[np.abs(t - ti) <= 15].mean() for ti in t]
d["centrada"] = cen
y = d.consumo.values; Dtr, Dte = d[tr], d[te]; ytr, yte = y[tr], y[te]
R = {}
def ev(nombre, modelo, cols):
    modelo.fit(Dtr[cols], ytr); R[nombre] = dict(train=round(MAE(ytr, modelo.predict(Dtr[cols])), 3), test=round(MAE(yte, modelo.predict(Dte[cols])), 3))
    print(f"{nombre:42s}", R[nombre])

print("base: predecir la media", round(MAE(yte, np.full(te.sum(), ytr.mean())), 3)); R["media"] = dict(test=round(MAE(yte, np.full(te.sum(), ytr.mean())), 3))
# A. escalar con k vecinos
kc = ["tipo", "carga_t", "km", "hist30"]
ohk = lambda: ColumnTransformer([("t", OneHotEncoder(), ["tipo"]), ("n", "passthrough", ["carga_t", "km", "hist30"])])
ev("knn sin escalar", make_pipeline(ohk(), KNeighborsRegressor(20)), kc)
ev("knn escalado", make_pipeline(ohk(), StandardScaler(), KNeighborsRegressor(20)), kc)
# B. tipo de camión como número o como columnas
num = ColumnTransformer([("t", OrdinalEncoder(), ["tipo"]), ("c", "passthrough", ["carga_t"])])
oh = ColumnTransformer([("t", OneHotEncoder(), ["tipo"]), ("c", "passthrough", ["carga_t"])])
ev("lineal, tipo como número", make_pipeline(num, Ridge(1e-3)), ["tipo", "carga_t"])
ev("lineal, tipo one-hot", make_pipeline(oh, Ridge(1e-3)), ["tipo", "carga_t"])
# C. cliente (300 categorías) con boosting
base = ["tipo", "carga_t", "km"]
def ct(extra):
    return ColumnTransformer([("t", OneHotEncoder(), ["tipo"]), ("n", "passthrough", ["carga_t", "km"])] + extra)
hgb = lambda: HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=0)
ev("cliente: sin usarlo", make_pipeline(ct([]), hgb()), base)
ev("cliente: one-hot (una columna por cliente)", make_pipeline(ct([("k", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["cliente"])]), hgb()), base + ["cliente"])
class MediaIngenua(TargetEncoder):            # media del cliente calculada con TODO el entrenamiento, sin ajuste cruzado ni suavizado
    def fit_transform(self, X, y=None): return self.fit(X, y).transform(X)
ev("cliente: media sin ajuste cruzado (m=0)", make_pipeline(ct([("k", MediaIngenua(smooth=1e-9, target_type="continuous"), ["cliente"])]), hgb()), base + ["cliente"])
ev("cliente: media con ajuste cruzado (m=10)", make_pipeline(ct([("k", TargetEncoder(smooth=10.0, target_type="continuous", cv=KFold(5, shuffle=True, random_state=0)), ["cliente"])]), hgb()), base + ["cliente"])
# D. hora como número o como círculo (modelo lineal)
corto = FunctionTransformer(lambda X: np.exp(-np.asarray(X, float) / 40))      # trayectos cortos
lin = lambda h: make_pipeline(ColumnTransformer([("t", OneHotEncoder(), ["tipo"]), ("c", "passthrough", ["carga_t"]), ("km", corto, ["km"])] + h), Ridge(1e-3))
hc = FunctionTransformer(lambda X: ciclo(np.asarray(X).ravel(), 24))
ev("lineal, sin la hora", lin([]), ["tipo", "carga_t", "km"])
ev("lineal, hora como número", lin([("h", "passthrough", ["hora"])]), ["tipo", "carga_t", "km", "hora"])
ev("lineal, hora en círculo", lin([("h", hc, ["hora"])]), ["tipo", "carga_t", "km", "hora"])
# E. historial del camión
full = lambda extra: make_pipeline(ColumnTransformer([("t", OneHotEncoder(), ["tipo"]), ("n", "passthrough", ["carga_t", "km"] + extra),
                                                      ("h", FunctionTransformer(lambda X: ciclo(np.asarray(X).ravel(), 24)), ["hora"]),
                                                      ("k", TargetEncoder(smooth=10.0, target_type="continuous", cv=KFold(5, shuffle=True, random_state=0)), ["cliente"])]), hgb())
cols = ["tipo", "carga_t", "km", "hora", "cliente"]
ev("todo, sin historial", full([]), cols)
ev("todo + media de los 30 días anteriores", full(["hist30"]), cols + ["hist30"])
ev("todo + ventana centrada (fuga)", full(["centrada"]), cols + ["centrada"])
json.dump(R, open("../resultados.json", "w"), ensure_ascii=False, indent=1)

# ---------- datos para los widgets ----------
E = {"R": R}
rng = np.random.default_rng(11)                                   # juguete: 150 viajes de rígidos de una misma ruta
km = np.clip(np.exp(rng.normal(np.log(60), 1.0, 150)), 5, 600); cg = rng.uniform(0, 24, 150)
E["toy"] = dict(km=km.round(1).tolist(), carga_kg=(cg * 1000).round(0).tolist(), consumo=(22 + 0.32 * cg + 9 * np.exp(-km / 40) + rng.normal(0, 0.8, 150)).round(2).tolist())
E["kmcol"] = Dtr.km.sample(40, random_state=3).round(1).tolist()
m_num = make_pipeline(num, Ridge(1e-3)).fit(Dtr[["tipo", "carga_t"]], ytr); m_oh = make_pipeline(oh, Ridge(1e-3)).fit(Dtr[["tipo", "carga_t"]], ytr)
T = sorted(d.tipo.unique()); q = pd.DataFrame(dict(tipo=T, carga_t=Dtr.carga_t.mean()))
E["tipos"] = dict(nombres=T, real=[round(float(Dtr[Dtr.tipo == t].consumo.mean() - 0.32 * (Dtr[Dtr.tipo == t].carga_t.mean() - Dtr.carga_t.mean())), 2) for t in T],
                  num=m_num.predict(q).round(2).tolist(), oh=m_oh.predict(q).round(2).tolist(), n=[int((Dtr.tipo == t).sum()) for t in T])
vc = Dtr.cliente.value_counts(); g = float(ytr.mean())
pick = [vc.index[i] for i in [0, 5, 30, 120, 300, 600, 800, 1000]]
E["clientes"] = dict(global_=round(g, 2), lista=[dict(id=c, n=int(vc[c]), media=round(float(Dtr[Dtr.cliente == c].consumo.mean()), 2)) for c in pick])
mh = lin([("h", hc, ["hora"])]).fit(Dtr[["tipo", "carga_t", "km", "hora"]], ytr)
co = mh[-1].coef_[-2:]; hh = ciclo(Dtr.hora, 24) @ co
res = ytr - mh.predict(Dtr[["tipo", "carga_t", "km", "hora"]]) + hh
bins = np.floor(Dtr.hora.values).astype(int)
E["hora"] = dict(media=[round(float(res[bins == b].mean()), 3) if (bins == b).sum() > 5 else None for b in range(24)],
                 n=[int((bins == b).sum()) for b in range(24)], sin=round(float(co[0]), 4), cos=round(float(co[1]), 4))
mn = lin([("h", "passthrough", ["hora"])]).fit(Dtr[["tipo", "carga_t", "km", "hora"]], ytr)
E["hora"]["pend"] = round(float(mn[-1].coef_[-1]), 4); E["hora"]["hmean"] = round(float(Dtr.hora.mean()), 3); E["hora"]["hmed_ef"] = round(float(hh.mean()), 4)
cam = d[d.camion == d.camion.value_counts().index[3]]
E["camion"] = dict(dia=cam.dia.round(2).tolist(), consumo=cam.consumo.round(2).tolist(), corte=160)
json.dump(E, open("../embed.json", "w"), ensure_ascii=False)
print("embed ok", {k: (len(v) if hasattr(v, "__len__") else v) for k, v in E.items()}, E["tipos"], E["clientes"], E["hora"]["sin"], E["hora"]["cos"], E["hora"]["pend"])
