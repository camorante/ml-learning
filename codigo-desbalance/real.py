import numpy as np, json
from sim import semanas
from imb import matriz, metricas, curva_roc, area, precision_media, corregir_prior, smote, brier, umbral_coste
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

d = semanas(); X = d.drop(columns="averia").values; y = d.averia.values
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.5, stratify=y, random_state=0)
sc = StandardScaler().fit(Xtr); Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
rng = np.random.default_rng(0); pos, neg = np.where(ytr)[0], np.where(~ytr)[0]; pi = ytr.mean()
variantes = {
    "sin tocar": (Xtr, ytr, None, 1.0),
    "pesos equilibrados": (Xtr, ytr, "balanced", (1 - pi) / pi),
    "sobremuestreo": (np.vstack([Xtr, Xtr[rng.choice(pos, len(neg) - len(pos))]]), np.r_[ytr, np.ones(len(neg) - len(pos), bool)], None, (1 - pi) / pi),
    "submuestreo": (Xtr[np.r_[pos, rng.choice(neg, len(pos), replace=False)]], ytr[np.r_[pos, rng.choice(neg, len(pos), replace=False)]], None, (1 - pi) / pi),
}
Xs, ys = smote(Xtr, ytr, 5, seed=0); variantes["SMOTE"] = (Xs, ys, None, (1 - pi) / pi)
out = {}
for k, (A, b, w, f) in variantes.items():
    m = LogisticRegression(max_iter=3000, class_weight=w).fit(A, b); p = m.predict_proba(Xte)[:, 1]
    fx, ty, _ = curva_roc(yte, p); mt = metricas(**matriz(yte, p, 0.5)); pc = corregir_prior(p, f)
    out[k] = dict(auc=round(area(fx, ty), 4), ap=round(precision_media(yte, p), 4), prec05=round(mt["precision"], 3), rec05=round(mt["exhaustividad"], 3),
                  p_media=round(float(p.mean()), 4), brier=round(brier(yte, p), 4), brier_corr=round(brier(yte, pc), 4), p_media_corr=round(float(pc.mean()), 4))
    print(f"{k:20s}", out[k])
print("tasa real en la prueba", round(yte.mean(), 4))
# costes
m = LogisticRegression(max_iter=3000).fit(Xtr, ytr); p = m.predict_proba(Xte)[:, 1]
CFN, CFP = 4000, 150; t_opt = umbral_coste(CFP, CFN)
def coste(t): c = matriz(yte, p, t); return c["fn"] * CFN + c["fp"] * CFP, c
for t in [0.5, t_opt, 0.1, 0.02]:
    print(round(t, 4), coste(t))
ts = np.linspace(0.001, 0.6, 600); cs = [coste(t)[0] for t in ts]; print("mejor umbral empírico", round(ts[int(np.argmin(cs))], 3), min(cs), "nunca avisar", int(yte.sum()) * CFN)
# exportar para la página sencilla: puntuaciones del modelo sin tocar y etiquetas de la prueba
json.dump(dict(p=np.round(p, 5).tolist(), y=yte.astype(int).tolist(), variantes=out, cfn=CFN, cfp=CFP), open("../embed.json", "w"), separators=(",", ":"))
mb = LogisticRegression(max_iter=3000, class_weight="balanced").fit(Xtr, ytr); pb = mb.predict_proba(Xte)[:, 1]
e = json.load(open("../embed.json")); e["pb"] = np.round(pb, 5).tolist(); e["factor"] = round(float((1 - pi) / pi), 3)
json.dump(e, open("../embed.json", "w"), separators=(",", ":"))
print("factor", (1 - pi) / pi)
