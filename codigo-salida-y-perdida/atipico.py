"""El mejor número único para 10 viajes de una ruta según la pérdida: MSE → media, MAE → mediana,
Huber → algo entre medias. El décimo viaje lo leyó mal el sensor (95 L/100 km)."""
import numpy as np
from scipy.optimize import minimize_scalar
from datos import RUTA, ATIPICO
from perdidas import mse, mae, huber

buenos = np.array(RUTA)
for nombre, y in (("sin el atípico", buenos), ("con el atípico", np.r_[buenos, ATIPICO])):
    mejor = {"MSE": y.mean(), "MAE": np.median(y),
             "Huber δ=2": minimize_scalar(lambda c: huber(c, y, 2.0), bounds=(20, 100), method="bounded", options={"xatol": 1e-9}).x}
    print(f"{nombre:15s} " + "   ".join(f"{k}: {v:6.2f}" for k, v in mejor.items()))
ref = buenos.mean()
y = np.r_[buenos, ATIPICO]
e = ATIPICO - ref
print(f"error del atípico frente a {ref:.2f}: {e:.1f} L/100 km → castigo MSE {e**2:.0f}, MAE {abs(e):.1f}, "
      f"Huber {2 * (abs(e) - 1):.1f}")
print(f"peso del atípico en la pérdida total: MSE {e**2 / (len(y) * mse(ref, y)):.1%}"
      f"  MAE {abs(e) / (len(y) * mae(ref, y)):.1%}")
