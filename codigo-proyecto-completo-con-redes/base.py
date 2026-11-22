"""Modelos de referencia con los 18 rasgos hechos a mano + la tabla (5 columnas) + las categorías: regresión logística (one-hot)
y boosting de árboles (HistGradientBoosting, categorías nativas, nº de árboles elegido con la validación)."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import log_loss
from rasgos import a_mano, Escalador


def matriz_rasgos(F):
    return np.concatenate([a_mano(F["X"]), F["tab"]], 1)


def logistica(Ftr, Fva, Fte):
    A = [matriz_rasgos(F) for F in (Ftr, Fva, Fte)]
    esc = Escalador(A[0], 0)
    oh = lambda F: np.c_[np.eye(8)[F["mod"]], np.eye(10)[F["dep"]]]
    M = [np.c_[esc(a), oh(F)] for a, F in zip(A, (Ftr, Fva, Fte))]
    lr = LogisticRegression(C=1.0, max_iter=5000).fit(M[0], Ftr["y"])
    return [lr.predict_proba(m)[:, 1] for m in M[1:]]


def boosting(Ftr, Fva, Fte, seed=0):
    M = [np.c_[matriz_rasgos(F), F["mod"], F["dep"]] for F in (Ftr, Fva, Fte)]
    gb = HistGradientBoostingClassifier(max_iter=600, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=40,
                                        categorical_features=[23, 24], early_stopping=False, random_state=seed).fit(M[0], Ftr["y"])
    pva = list(gb.staged_predict_proba(M[1]))
    k = int(np.argmin([log_loss(Fva["y"], q[:, 1]) for q in pva]))
    pte = list(gb.staged_predict_proba(M[2]))[k]
    return pva[k][:, 1], pte[:, 1], k + 1
