import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import mutual_info_score
from sim import camiones
from entropia import entropia_etiquetas, ganancia_informacion, informacion_mutua

d = camiones(); y = d["averia"]
print("H(avería) =", round(entropia_etiquetas(y), 4), "bits  (", y.sum(), "de", len(y), ")")
X = np.c_[d["km"], d["urbana"], d["par"], d["temp"]]; nombres = ["km", "urbana", "par", "temp"]
for prof in (1, 2):
    t = DecisionTreeClassifier(criterion="entropy", max_depth=prof, random_state=0).fit(X, y)
    print(f"árbol de profundidad {prof}:\n" + export_text(t, feature_names=nombres, decimals=1, show_weights=True))
t1 = DecisionTreeClassifier(criterion="entropy", max_depth=1).fit(X, y).tree_
print("ganancia de la raíz (sklearn):", round(t1.impurity[0] - (t1.weighted_n_node_samples[1:] @ t1.impurity[1:]) / 1000, 4),
      "  desde cero:", round(ganancia_informacion(y, X[:, 3] <= t1.threshold[0]), 4))
print("«fuera de 80–100 °C» (una sola pregunta):", round(ganancia_informacion(y, (d["temp"] < 80) | (d["temp"] > 100)), 4))

print("correlación con la avería:", {n: round(float(np.corrcoef(X[:, j], y)[0, 1]), 4) for j, n in enumerate(nombres)})
tramos = np.digitize(d["temp"], np.arange(62, 122, 4))
print("IM temp (tramos de 4 °C): desde cero", round(informacion_mutua(tramos, y), 4), "nats  sklearn", round(mutual_info_score(tramos, y), 4))
rng = np.random.default_rng(0)
nulo = [mutual_info_score(tramos, rng.permutation(y)) for _ in range(200)]
print("   con la avería barajada (sin relación):", round(np.mean(nulo), 4), "de media, máx", round(max(nulo), 4), "← sesgo hacia arriba")
mi = mutual_info_classif(X, y, discrete_features=[False, True, True, False], random_state=0)
print("mutual_info_classif (k vecinos, nats):", {n: round(float(v), 4) for n, v in zip(nombres, mi)})
