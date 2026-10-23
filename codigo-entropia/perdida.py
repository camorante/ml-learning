import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, brier_score_loss, accuracy_score
from sim import semanas
from entropia import bce_con_logits, entropia_cruzada_logits, softmax, gradiente_logits

X, averia, tipo = semanas()
Xtr, Xte, ytr, yte = train_test_split(X, averia, test_size=0.5, random_state=0, stratify=averia)
m = LogisticRegression(C=np.inf, max_iter=1000).fit(Xtr, ytr)
z = m.decision_function(Xte); p = m.predict_proba(Xte)[:, 1]
ll = np.sum(yte * np.log(p) + (1 - yte) * np.log1p(-p))                # log-verosimilitud de Bernoulli
print(f"binaria: log_loss = {log_loss(yte, p):.5f}  BCE con logits = {bce_con_logits(yte, z):.5f}  −ℓ/n = {-ll / len(yte):.5f}")
print(f"         Brier = {brier_score_loss(yte, p):.5f}  acierto = {accuracy_score(yte, p > 0.5):.4f}  (siempre «no»: {1 - yte.mean():.4f})")
for T in (1, 3, 10):                                                   # misma ordenación, más «seguridad»
    pT = 1 / (1 + np.exp(-T * z))
    print(f"   logits × {T:2d}: acierto {accuracy_score(yte, pT > 0.5):.4f}  log_loss {log_loss(yte, pT):.4f}  Brier {brier_score_loss(yte, pT):.4f}")
# calibración: media de p frente a la frecuencia real, por tramos
b = np.digitize(p, [0.05, 0.1, 0.2, 0.4])
print("calibración (media de p, frecuencia real, n):", [(round(float(p[b == k].mean()), 3), round(float(yte[b == k].mean()), 3), int((b == k).sum())) for k in range(5)])

# multiclase: tipo de avería con softmax (solo semanas con avería)
s = averia == 1
Xa, ya = X[s], tipo[s]
mm = LogisticRegression(C=np.inf, max_iter=5000, tol=1e-12).fit(Xa, ya)
Z = mm.decision_function(Xa)
print(f"multiclase ({s.sum()} averías, 4 tipos): log_loss = {log_loss(ya, mm.predict_proba(Xa)):.5f}"
      f"  desde cero = {entropia_cruzada_logits(ya, Z):.5f}  softmax = predict_proba: {np.abs(softmax(Z) - mm.predict_proba(Xa)).max():.1e}")
G = gradiente_logits(ya, Z)
print("gradiente respecto a los pesos, X^T (p − y): máx", f"{np.abs(np.c_[np.ones(len(Xa)), Xa].T @ G).max():.1e}", "← en el óptimo es 0")
print("entropía de las clases (modelo sin variables):", round(log_loss(ya, np.tile(np.bincount(ya) / len(ya), (len(ya), 1))), 4), "nats")
