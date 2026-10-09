import numpy as np
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics.pairwise import nan_euclidean_distances
from miss import *
rng = np.random.default_rng(0); e = dict(media=0, nan_eu=0, knn=0, mice=0)
for t in range(40):
    n, p = int(rng.integers(30, 200)), int(rng.integers(2, 6))
    C = rng.normal(size=(p, p)); X = rng.normal(size=(n, p)) @ C
    X[rng.random((n, p)) < 0.2] = np.nan
    X[np.isnan(X).all(1), 0] = 0.0
    a = np.column_stack([imputar_media(X[:, j]) for j in range(p)]); e["media"] = max(e["media"], np.abs(a - SimpleImputer().fit_transform(X)).max())
    e["nan_eu"] = max(e["nan_eu"], np.nanmax(np.abs(nan_euclidean(X[:20], X) - nan_euclidean_distances(X[:20], X))))
    k = int(rng.integers(1, 7)); e["knn"] = max(e["knn"], np.abs(imputar_knn(X, k) - KNNImputer(n_neighbors=k).fit_transform(X)).max())
    ref = IterativeImputer(estimator=LinearRegression(), max_iter=10, tol=1e-3, sample_posterior=False, random_state=0).fit_transform(X)
    e["mice"] = max(e["mice"], np.abs(imputar_encadenado(X, 10, 1e-3) - ref).max())
print({k_: float(f"{v:.1e}") for k_, v in e.items()})
q, T, W, B = rubin([10.2, 9.8, 10.6], [0.25, 0.20, 0.30]); print("Rubin", round(q, 3), round(T, 4), W, round(B, 4))
