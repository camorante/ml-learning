import numpy as np
from sklearn.decomposition import PCA, TruncatedSVD
from sim import flota
from svd import svd_jacobi

X, *_ = flota()                                         # 300 camiones × 28 días
Xc = X - X.mean(axis=0)                                 # PCA: centrar cada columna (cada día)
U, s, Vt = svd_jacobi(Xc)
sg = np.sign(Vt[np.arange(len(s)), np.abs(Vt).argmax(axis=1)])   # signos como svd_flip(u_based_decision=False)
U, Vt = U * sg, Vt * sg[:, None]
p = PCA(n_components=5, svd_solver="full").fit(X)
print("componentes vs PCA.components_      ", f"{np.abs(Vt[:5] - p.components_).max():.1e}")
print("σ²/(n−1) vs explained_variance_      ", f"{np.max(np.abs(s[:5]**2 / (len(X) - 1) - p.explained_variance_) / p.explained_variance_):.1e}")
print("U·Σ vs PCA.transform(X)              ", f"{np.abs(U[:, :5] * s[:5] - p.transform(X)).max():.1e}")
ts = TruncatedSVD(n_components=5, algorithm="arpack").fit(X)      # sin centrar
media = X.mean(axis=0) / np.linalg.norm(X.mean(axis=0))
print("sin centrar: |cos| entre la 1.ª componente y la dirección de la media", round(abs(ts.components_[0] @ media), 4))
print("% de varianza, PCA (centrado):", (100 * p.explained_variance_ratio_).round(1))
