from cond import prob_total, bayes

p_av = 0.02                     # P(avería esta semana)
sens, fa = 0.90, 0.05           # P(alerta | avería), P(alerta | sin avería)
p_alerta = prob_total([p_av, 1 - p_av], [sens, fa])
post = bayes([p_av, 1 - p_av], [sens, fa])[0]
print("P(alerta) =", round(p_alerta, 4), " P(avería | alerta) =", round(post, 4))
# forma de odds: odds a posteriori = odds a priori × razón de verosimilitudes
o0, rv = p_av / (1 - p_av), sens / fa
o1 = o0 * rv; print("odds:", round(o0, 4), "×", rv, "=", round(o1, 4), "->", round(o1 / (1 + o1), 4))
# coste de revisar cada alerta (150 €) por avería encontrada
print("€ por avería encontrada:", round(150 / post, 1))
# segunda señal (vibración), independiente de la primera DADO el estado del camión
s2, f2 = 0.80, 0.10
o2 = o0 * rv * (s2 / f2); print("con dos señales:", round(o2, 4), "->", round(o2 / (1 + o2), 4))
# en 1 000 camiones
print("de 1 000:", 1000 * p_av, 1000 * p_av * sens, 1000 * (1 - p_av) * fa)
