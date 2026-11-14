"""Verificación de red.py: mismos pesos en torch (float64) y en sklearn.neural_network.MLPClassifier."""
import warnings
import numpy as np
import torch
from sklearn.neural_network import MLPClassifier
from datos import motor, RPM0, RPM_S, TMP0, TMP_S
from red import MLP, Densa, n_param

torch.set_default_dtype(torch.float64)
warnings.simplefilter("ignore")
rng = np.random.default_rng(0)
TORCH = {"sigmoide": torch.nn.Sigmoid, "tanh": torch.nn.Tanh, "relu": torch.nn.ReLU, "identidad": torch.nn.Identity}

# 1) redes al azar: misma arquitectura y mismos pesos en torch → mismas salidas
e_max, n_redes = 0.0, 0
for act in ("sigmoide", "tanh", "relu"):
    for tam in ([2, 2, 1], [2, 8, 1], [3, 16, 8, 1], [5, 32, 32, 16, 1]):
        for _ in range(25):
            capas, mods = [], []
            for l, (d, m) in enumerate(zip(tam, tam[1:])):
                W, b = rng.normal(0, 1.5, (m, d)), rng.normal(0, 1, m)
                a = act if l < len(tam) - 2 else "sigmoide"
                capas.append(Densa(W, b, a))
                lin = torch.nn.Linear(d, m)
                with torch.no_grad():
                    lin.weight.copy_(torch.tensor(W)); lin.bias.copy_(torch.tensor(b))
                mods += [lin, TORCH[a]()]
            red, net = MLP(capas), torch.nn.Sequential(*mods)
            X = rng.normal(0, 2, (500, tam[0]))
            with torch.no_grad():
                ref = net(torch.tensor(X)).numpy()
            e_max = max(e_max, np.max(np.abs(red.forward(X) - ref)))
            assert red.n_param == n_param(tam) == sum(p.numel() for p in net.parameters())
            n_redes += 1
print(f"{n_redes} redes al azar (3 activaciones × 4 arquitecturas), 500 entradas cada una:")
print(f"   salida vs torch.nn.Sequential: diferencia máxima {e_max:.1e};  parámetros iguales a los de torch")

# 2) MLPClassifier entrenado: cargar coefs_ e intercepts_ en nuestra clase
X, y = motor()
Xe = np.c_[(X[:, 0] - RPM0) / RPM_S, (X[:, 1] - TMP0) / TMP_S]
Xg = np.c_[rng.uniform(-3, 3, 5000), rng.uniform(-3, 3, 5000)]       # puntos nuevos por todo el plano
for act, capas in (("logistic", (6,)), ("tanh", (8, 4)), ("relu", (16, 16))):
    sk = MLPClassifier(hidden_layer_sizes=capas, activation=act, solver="lbfgs", max_iter=5000,
                       random_state=0).fit(Xe, y)
    red = MLP.desde_sklearn(sk)
    d = np.max(np.abs(red.predict_proba(Xg) - sk.predict_proba(Xg)))
    print(f"MLPClassifier {act:8s} {str(capas):8s}: forma {red.tamanos}, {red.n_param:3d} parámetros,"
          f" predict_proba vs la nuestra {d:.1e}, mismas clases: {bool(np.all(red.predict(Xg) == sk.predict(Xg)))},"
          f" errores {int(np.sum(red.predict(Xe) != y))} de {len(y)}")
