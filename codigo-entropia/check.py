"""Compara entropia.py con SciPy y scikit-learn en 200 problemas aleatorios: error máximo."""
import numpy as np
from scipy import stats, special
from sklearn.metrics import log_loss, mutual_info_score
from sklearn.tree import DecisionTreeClassifier
import entropia as E

rng = np.random.default_rng(2026); err = {}
def anota(k, a, b):
    err[k] = max(err.get(k, 0.0), float(np.max(np.abs(np.asarray(a, float) - np.asarray(b, float)))))

for _ in range(200):
    K = int(rng.integers(2, 9))
    p = rng.dirichlet(np.ones(K) * 0.7); q = rng.dirichlet(np.ones(K))
    cero = rng.random(K) < 0.2; cero[np.argmax(p)] = False
    p[cero] = 0; p /= p.sum()                                        # algunos ceros: 0 · log 0 = 0
    for base in (2, np.e):
        anota("entropía vs stats.entropy", E.entropia(p, base), stats.entropy(p, base=base))
        anota("KL vs stats.entropy(p, q)", E.kl(p, q, base), stats.entropy(p, q, base=base))
        anota("H(p,q) = H(p) + KL", E.entropia_cruzada(p, q, base), E.entropia(p, base) + E.kl(p, q, base))
    c = E.huffman(p + 1e-12); H, L = E.entropia(p + 1e-12), E.longitud_media(p + 1e-12, c)
    assert H - 1e-9 <= L < H + 1 and abs(E.kraft(c) - 1) < 1e-12           # cota de Shannon y Kraft
    prefijos = all(not b.startswith(a) for a in c for b in c if a is not b)
    assert prefijos
    n = int(rng.integers(20, 400)); z = rng.normal(0, 3, (n, K)); y = rng.integers(0, K, n)
    anota("logsumexp vs special.logsumexp", E.logsumexp(z), special.logsumexp(z, axis=1))
    anota("softmax vs special.softmax", E.softmax(z), special.softmax(z, axis=1))
    lab = np.arange(K)
    anota("entropía cruzada (logits) vs log_loss", E.entropia_cruzada_logits(y, z), log_loss(y, special.softmax(z, axis=1), labels=lab))
    ys = E.suavizar_etiquetas(y, K, 0.1)
    anota("con label smoothing vs −Σ y' log p", E.entropia_cruzada_logits(ys, z), -np.mean(np.sum(ys * special.log_softmax(z, axis=1), 1)))
    h = 1e-6; i, j = int(rng.integers(n)), int(rng.integers(K)); zz = z.copy(); zz[i, j] += h
    num = (E.entropia_cruzada_logits(y, zz) - E.entropia_cruzada_logits(y, z)) / h
    anota("gradiente (p − y)/n vs diferencias finitas", E.gradiente_logits(y, z)[i, j], num)
    yb = rng.integers(0, 2, n); zb = rng.normal(0, 4, n); pb = special.expit(zb)
    anota("log loss binaria vs log_loss", E.log_loss_binaria(yb, pb), log_loss(yb, pb, labels=[0, 1]))
    anota("BCE con logits vs log_loss", E.bce_con_logits(yb, zb), log_loss(yb, pb, labels=[0, 1]))
    x1 = rng.integers(0, int(rng.integers(2, 6)), n); x2 = (x1 + rng.integers(0, 3, n)) % 4
    anota("información mutua vs mutual_info_score", E.informacion_mutua(x1, x2), mutual_info_score(x1, x2))
    xc = rng.normal(0, 1, (n, 1)); yc = (xc[:, 0] + rng.normal(0, 1, n) > 0).astype(int)
    if 0 < yc.sum() < n:
        t = DecisionTreeClassifier(criterion="entropy", max_depth=1).fit(xc, yc).tree_
        if t.node_count == 3:
            w = t.weighted_n_node_samples
            dec = t.impurity[0] - (w[1] * t.impurity[1] + w[2] * t.impurity[2]) / w[0]
            anota("ganancia de información vs árbol (entropy)", E.ganancia_informacion(yc, xc[:, 0] <= t.threshold[0]), dec)
    s = float(rng.uniform(0.1, 30))
    anota("entropía normal vs norm.entropy", E.entropia_normal(s), stats.norm(0, s).entropy())

for k, v in err.items():
    print(f"{k:45s} {v:.1e}")
print("Huffman: H ≤ L < H + 1, Kraft = 1 y código prefijo en los 200 casos")
