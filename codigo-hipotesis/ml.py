import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

rng = np.random.default_rng(3); n = 6000
X = np.c_[rng.normal(0, 1, n), rng.normal(0, 1, n), rng.normal(0, 1, n)]       # vibración, temperatura, ruido (inútil)
y = (rng.random(n) < 1 / (1 + np.exp(-(-1.0 + 1.2 * X[:, 0] + 0.5 * X[:, 1])))).astype(int)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=2000, random_state=0)

# 1) ¿acierta más el modelo nuevo (vibración + temperatura) que el actual (solo vibración)?
a = LogisticRegression().fit(Xtr[:, :1], ytr).predict(Xte[:, :1]) == yte
b = LogisticRegression().fit(Xtr[:, :2], ytr).predict(Xte[:, :2]) == yte
n01, n10 = int(np.sum(a & ~b)), int(np.sum(~a & b))
print(f"acierto actual {a.mean():.4f}  nuevo {b.mean():.4f}   discordantes: solo acierta el actual {n01}, solo el nuevo {n10}")
print(f"McNemar exacto (binomial con p = 1/2 sobre {n01 + n10} discordantes): p = {stats.binomtest(n10, n01 + n10).pvalue:.4f}")
T = [[a.sum(), 2000 - a.sum()], [b.sum(), 2000 - b.sum()]]
print(f"χ² como si fueran muestras independientes (incorrecto: son los mismos camiones): p = {stats.chi2_contingency(T, correction=False).pvalue:.4f}")

# 2) ¿sirve la variable «ruido»? razón de verosimilitudes (tema 14) y test de Wald de un coeficiente
def loglik(Xm):
    m = LogisticRegression(C=np.inf, max_iter=2000).fit(Xm, y); p = m.predict_proba(Xm)[:, 1]
    return np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)), m, p
l1, m1, p1 = loglik(X); l0, _, _ = loglik(X[:, :2])
G = 2 * (l1 - l0); print(f"razón de verosimilitudes: G = 2·(ℓ₁ − ℓ₀) = {G:.3f}  ~ χ²(1)  p = {stats.chi2.sf(G, 1):.4f}")
Z = np.c_[np.ones(n), X]; H = Z.T @ (Z * (p1 * (1 - p1))[:, None]); ee = np.sqrt(np.diag(np.linalg.inv(H)))
w = m1.coef_[0][2] / ee[3]; print(f"Wald: β_ruido = {m1.coef_[0][2]:.4f}  ee = {ee[3]:.4f}  z = {w:.3f}  p = {2 * stats.norm.sf(abs(w)):.4f}")
