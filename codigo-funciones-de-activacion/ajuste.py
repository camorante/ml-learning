"""La misma U con scikit-learn: MLPRegressor de 16 neuronas ocultas con cada activación."""
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from datos import tramos

v, c = tramos(); vn, cn = tramos(200, seed=7)
X, Xn = (v[:, None] - 60) / 30, (vn[:, None] - 60) / 30         # velocidad escalada
rmse = lambda a, b: np.sqrt(np.mean((a - b) ** 2))
lin = LinearRegression().fit(X, c)
print(f"LinearRegression          error {rmse(lin.predict(X), c):.3f}   tramos nuevos {rmse(lin.predict(Xn), cn):.3f}")
for act in ("identity", "logistic", "tanh", "relu"):
    m = min((MLPRegressor(hidden_layer_sizes=(16,), activation=act, solver="lbfgs", alpha=1e-4, max_iter=20000,
                          random_state=s).fit(X, c) for s in range(5)), key=lambda m: rmse(m.predict(X), c))
    print(f"MLPRegressor {act:9s}    error {rmse(m.predict(X), c):.3f}   tramos nuevos {rmse(m.predict(Xn), cn):.3f}"
          f"   consumo a 20/75/105 km/h: {np.round(m.predict((np.array([[20], [75], [105]]) - 60) / 30), 1)}")
