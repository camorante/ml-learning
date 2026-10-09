import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder, TargetEncoder
from sklearn.model_selection import KFold
from prep import *
rng = np.random.default_rng(0); e = dict(std=0, minmax=0, robust=0, onehot=0, target=0, target_fit=0, ventana=0)
for k in range(100):
    n = int(rng.integers(20, 2000)); X = rng.lognormal(0, 1, (n, 3)) * rng.uniform(1, 500, 3)
    e["std"] = max(e["std"], np.abs(Estandarizar().fit(X).transform(X) - StandardScaler().fit_transform(X)).max())
    e["minmax"] = max(e["minmax"], np.abs(MinMax().fit(X).transform(X) - MinMaxScaler().fit_transform(X)).max())
    e["robust"] = max(e["robust"], np.abs(Robusto().fit(X).transform(X) - RobustScaler().fit_transform(X)).max())
    c = rng.choice([f"c{i}" for i in range(int(rng.integers(2, 40)))], n)
    nuevo = np.r_[c[:5], ["desconocida"]]
    mine = UnaColumnaPorCategoria().fit(c).transform(nuevo)
    ref = OneHotEncoder(handle_unknown="ignore").fit(c.reshape(-1, 1)).transform(nuevo.reshape(-1, 1)).toarray()
    e["onehot"] = max(e["onehot"], np.abs(mine - ref).max())
    y = rng.normal(0, 1, n) + (np.char.str_len(c.astype(str)) == 2)
    m = float(rng.uniform(0.5, 30)); seed = int(rng.integers(0, 1000))
    te = TargetEncoder(smooth=m, target_type="continuous", cv=KFold(5, shuffle=True, random_state=seed))
    ref = te.fit_transform(c.reshape(-1, 1), y).ravel()
    folds = list(KFold(5, shuffle=True, random_state=seed).split(c))
    enc = MediaPorCategoria(m); mine = enc.fit_transform(c, y, folds)
    e["target"] = max(e["target"], np.abs(mine - ref).max())
    e["target_fit"] = max(e["target_fit"], np.abs(enc.transform(nuevo) - te.transform(nuevo.reshape(-1, 1)).ravel()).max())
    g = rng.choice(list("ABC"), n); t = np.round(rng.uniform(0, 100, n), 1); v = rng.normal(0, 1, n); w = float(rng.uniform(1, 30))
    mine = media_pasada(g, t, v, w)
    df = pd.DataFrame(dict(g=g, t=pd.to_datetime(t, unit="D"), v=v)).sort_values("t", kind="mergesort")
    ref = np.full(n, np.nan)
    for _, d in df.groupby("g"):                       # pandas: ventana de tiempo que excluye la propia fila y el borde
        r = d.rolling(pd.Timedelta(days=w), on="t", closed="neither").v.mean()
        ref[d.index.values] = r.values
    ok = ~pd.Series(t).duplicated(keep=False).values & ~np.isnan(mine)   # sin empates de t (pandas los trata en orden)
    e["ventana"] = max(e["ventana"], np.nanmax(np.abs(mine[ok] - ref[ok])) if ok.any() else 0)
print({k: float(f"{v:.1e}") for k, v in e.items()})
