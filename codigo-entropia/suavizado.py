import numpy as np
from scipy.optimize import minimize_scalar
from entropia import entropia, entropia_cruzada_logits, suavizar_etiquetas

K = 4
for eps in (0.0, 0.05, 0.1, 0.2):
    y = suavizar_etiquetas([1], K, eps)                         # la avería era de frenos (clase 1)
    loss = lambda m: entropia_cruzada_logits(y, np.array([[0, m, 0, 0]]))   # m = ventaja del logit verdadero
    r = minimize_scalar(loss, bounds=(0, 60), method="bounded")
    teo = np.inf if eps == 0 else np.log((1 - eps + eps / K) / (eps / K))
    print(f"ε = {eps:.2f}: margen óptimo {r.x:6.3f} (teoría {teo:6.3f})  pérdida mínima {abs(r.fun):.4f}"
          f" = H(y') = {entropia(y[0], np.e):.4f} nats  p(verdadera) = {1 - eps + eps / K:.3f}")
