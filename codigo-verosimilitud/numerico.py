import numpy as np
from scipy import stats, special
from sim import alertas_largo
from mv import log_sigmoid, logsumexp, ll_poisson

# producto frente a suma de logaritmos
x = alertas_largo()
print("producto de 1000 probabilidades:", np.prod(stats.poisson.pmf(x, x.mean())),
      "| suma de logaritmos:", round(ll_poisson(x.mean(), x), 3))
z = np.array([-800.0, -40.0, 0.0, 40.0])
with np.errstate(over="ignore", divide="ignore"):
    print("log(sigmoide) ingenuo:", np.log(1 / (1 + np.exp(-z))))
print("log_sigmoid estable  :", log_sigmoid(z))
print("special.log_expit    :", special.log_expit(z))
v = np.array([-1000.0, -1001.0, -1002.0])
with np.errstate(divide="ignore"):
    print("log(Σ e^v) ingenuo:", np.log(np.sum(np.exp(v))))
print("logsumexp:", round(logsumexp(v), 6), "| special.logsumexp:", round(special.logsumexp(v), 6))
with np.errstate(divide="ignore", invalid="ignore"):
    print("0·log(0) con xlogy:", special.xlogy(0, 0), "| ingenuo 0·log(0):", np.float64(0) * np.log(np.float64(0)))
