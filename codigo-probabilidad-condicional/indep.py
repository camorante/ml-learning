import numpy as np
from scipy import stats
from cond import esperada, chi2, cramer_v, condicional

T = np.array([[50, 150], [80, 720]])          # filas: lluvia sí / no; columnas: alerta sí / no
print("esperada si fueran independientes:\n", esperada(T))
est, gl, p = chi2(T); print("chi2 =", round(est, 2), "gl =", gl, "p =", f"{p:.2e}", "V =", round(cramer_v(T), 3))
r = stats.chi2_contingency(T, correction=False); print("scipy:", round(r.statistic, 2), f"{r.pvalue:.2e}")
print("crítico 5 %, 1 gl:", round(stats.chi2.ppf(0.95, 1), 3), "= 1.96² =", round(1.959964**2, 3))
# la misma proporción (25 % frente a 10 %) con solo 100 viajes: 20 de lluvia, 5 con alerta; 80 secos, 8 con alerta
t = np.array([[5, 15], [8, 72]]); e, g, pp = chi2(t)
print("con 100 viajes: chi2 =", round(e, 2), "p =", round(pp, 3), "esperada mínima", esperada(t).min())
print("Fisher exacta:", round(stats.fisher_exact(t).pvalue, 3))
# tabla independiente: 10 % de alertas con y sin lluvia
I = np.array([[20, 180], [80, 720]]); print("independiente:", condicional(I)[:, 0], chi2(I)[0], round((I / I.sum())[0, 0], 3), round(0.2 * 0.1, 3))
# excluyentes: conductor Ana / conductor Luis en el mismo viaje (30 % y 25 %)
pa, pb = 0.30, 0.25; print("excluyentes: P(A y B) = 0 frente a P(A)P(B) =", pa * pb, " P(B | A) = 0 frente a P(B) =", pb)
# independencia dos a dos sin independencia conjunta: lunes y martes a tiempo (50 %), C = «los dos días igual»
w = [(l, m) for l in (0, 1) for m in (0, 1)]
P = lambda f: sum(0.25 for x in w if f(*x))
A = lambda l, m: l == 1; Bv = lambda l, m: m == 1; C = lambda l, m: l == m
print("pares:", P(lambda l, m: A(l, m) and C(l, m)), P(A) * P(C), " tres:", P(lambda l, m: A(l, m) and Bv(l, m) and C(l, m)), P(A) * P(Bv) * P(C))
