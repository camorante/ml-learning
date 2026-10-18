import numpy as np
from scipy import stats

T = np.array([[50, 150], [80, 720]])           # filas: lluvia sí / no; columnas: alerta sí / no
n = T.sum(); fil, col = T.sum(1), T.sum(0)
print("1. totales", fil, col, n, " P(lluvia) =", fil[0] / n, " P(alerta) =", col[0] / n)
print("2. 50 entre:", 50 / n, 50 / fil[0], round(50 / col[0], 4))
print("3. producto:", 0.20 * 0.25, round(0.13 * (50 / 130), 4))
print("4. invierno:", round(0.40 * 0.25 + 0.60 * 0.10, 4), " alertas de 1 000:", round(1000 * (0.40 * 0.25 + 0.60 * 0.10)),
      " × 6 € =", round(1000 * (0.40 * 0.25 + 0.60 * 0.10)) * 6, "frente a", 130 * 6)
print("5. Bayes:", round(0.20 * 0.25 / 0.13, 4), " odds", 0.2 / 0.8, "×", 0.25 / 0.10, "=", 0.25 * 2.5, "->", round(0.625 / 1.625, 4))
E = np.outer(fil, col) / n
c = (T - E) ** 2 / E
print("6. esperada", E.tolist(), " sumandos", c.round(2).tolist(), " chi2 =", round(c.sum(), 2),
      " crítico", round(stats.chi2.ppf(0.95, 1), 2), " p =", f"{stats.chi2.sf(c.sum(), 1):.1e}", " V =", round(np.sqrt(c.sum() / n), 3))
