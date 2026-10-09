import numpy as np
from sim import semanas
from imb import matriz, curva_roc, area, precision_media, corregir_prior, brier, umbral_coste
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

d = semanas(); X = d.drop(columns="averia").values; y = d.averia.values
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.5, stratify=y, random_state=0)
sc = StandardScaler().fit(Xtr); Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
pi = ytr.mean(); w = (1 - pi) / pi
print(round(w, 2))
for cw in (None, "balanced"):
    p = LogisticRegression(max_iter=3000, class_weight=cw).fit(Xtr, ytr).predict_proba(Xte)[:, 1]
    fx, ty, _ = curva_roc(yte, p)
    pc = p if cw is None else corregir_prior(p, w)
    print(cw, round(area(fx, ty), 4), round(precision_media(yte, p), 4), round(p.mean(), 4),
          round(brier(yte, p), 4), round(brier(yte, pc), 4))
t = umbral_coste(150, 4000); c = matriz(yte, p, t)
print(round(t, 4), c, c["fn"] * 4000 + c["fp"] * 150)
c = matriz(yte, pc, t); print(c, c["fn"] * 4000 + c["fp"] * 150)
