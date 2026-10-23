import numpy as np
from scipy import special
from sklearn.metrics import log_loss
from entropia import logsumexp, softmax, bce_con_logits

np.seterr(all="ignore")
z = np.array([[1000.0, 999.0, 990.0]])                       # logits grandes: exp desborda
ingenuo = np.exp(z) / np.exp(z).sum()
print("softmax ingenuo:", ingenuo, "  estable:", softmax(z).round(6), "  logsumexp:", logsumexp(z))
z = np.array([40.0, -40.0]); y = np.array([0.0, 1.0])      # dos casos seguros y equivocados
p = 1 / (1 + np.exp(-z))
print("p =", p, "  1 − p =", 1 - p)
print("BCE ingenua :", -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
print("BCE logits  :", bce_con_logits(y, z), "  (exacta: 40 + log(1 + e^-40) =", 40 + np.log1p(np.exp(-40)), ")")
print("log_loss    :", log_loss(y, p), "  ← recorta p a [ε, 1 − ε] con ε =", np.finfo(float).eps, "→ −ln ε =", -np.log(np.finfo(float).eps))
print("float32     : 1 − σ(17) =", 1 - special.expit(np.float32(17)), "  log1p(−σ) se pierde antes que en float64")
