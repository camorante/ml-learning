import numpy as np
from scipy.spatial.distance import cosine as sp_cos
from scipy.stats import pearsonr
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances
from producto import *

rng = np.random.default_rng(0)
e = dict(punto=0, coseno=0, dcos=0, pearson=0, mat=0, matd=0, proj=0, gs=0, cs=0)
topk_ok = 0
for t in range(200):
    d = int(rng.integers(2, 40))
    a, b = rng.normal(size=d) * rng.uniform(0.1, 50), rng.normal(size=d) * rng.uniform(0.1, 50)
    e["punto"] = max(e["punto"], abs(punto(a, b) - np.dot(a, b)) / max(1, abs(np.dot(a, b))))
    e["coseno"] = max(e["coseno"], abs(coseno(a, b) - cosine_similarity([a], [b])[0, 0]))
    e["dcos"] = max(e["dcos"], abs(distancia_coseno(a, b) - sp_cos(a, b)))
    e["pearson"] = max(e["pearson"], abs(pearson(a, b) - pearsonr(a, b).statistic))
    p, r = proyeccion(a, b)
    e["proj"] = max(e["proj"], abs(np.dot(r, b)) / (np.linalg.norm(r) * np.linalg.norm(b)))
    X, Y = rng.normal(size=(30, d)), rng.normal(size=(20, d))
    e["mat"] = max(e["mat"], np.abs(matriz_coseno(X, Y) - cosine_similarity(X, Y)).max())
    e["matd"] = max(e["matd"], np.abs(1 - matriz_coseno(X, Y) - cosine_distances(X, Y)).max())
    V = rng.normal(size=(min(d, 6), d)); Q = gram_schmidt(V)
    e["gs"] = max(e["gs"], np.abs(Q @ Q.T - np.eye(len(Q))).max())
    Qr, _ = np.linalg.qr(V.T)
    e["cs"] = max(e["cs"], np.abs(np.abs(Q) - np.abs(Qr.T)).max())
    idx, s = top_k(X, Y, 3)
    ref = np.argsort(-cosine_similarity(X, Y), 1, kind="stable")[:, :3]
    topk_ok += int((idx == ref).all())
# a·b = |a||b|cos θ y ||â - b̂||² = 2(1 - cos)
a, b = rng.normal(size=8), rng.normal(size=8)
ua, ub = a / norma(a), b / norma(b)
ident = abs(np.sum((ua - ub) ** 2) - 2 * (1 - coseno(a, b)))
print(f"punto vs np.dot (error relativo)                      {e['punto']:.1e}")
print(f"coseno vs sklearn cosine_similarity                   {e['coseno']:.1e}")
print(f"distancia_coseno vs scipy.spatial.distance.cosine     {e['dcos']:.1e}")
print(f"pearson (coseno centrado) vs scipy.stats.pearsonr     {e['pearson']:.1e}")
print(f"matriz_coseno vs cosine_similarity (30x20)            {e['mat']:.1e}")
print(f"1 - matriz_coseno vs cosine_distances                 {e['matd']:.1e}")
print(f"resto de la proyección ⟂ b (coseno residual)          {e['proj']:.1e}")
print(f"gram_schmidt: |QQᵀ - I|                               {e['gs']:.1e}")
print(f"gram_schmidt vs np.linalg.qr (salvo signo)            {e['cs']:.1e}")
print(f"top_k = argsort de sklearn en {topk_ok}/200 problemas")
print(f"||â - b̂||² - 2(1 - cos)                               {ident:.1e}")
print(coseno([0, 0, 0], [1, 2, 3]))
