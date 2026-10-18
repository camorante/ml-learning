import math
import numpy as np
from scipy import stats
from cor import rangos

x = np.array([6, 9, 11, 14, 16, 19, 21, 24.]); y = np.array([27, 30, 29, 33, 31, 35, 36, 35.]); n = 8
dx, dy = x - x.mean(), y - y.mean()
print("1. medias", x.sum(), "/ 8 =", x.mean(), "  ", y.sum(), "/ 8 =", y.mean())
print("2. dx", dx.tolist(), " dy", dy.tolist(), "\n   dx·dy", (dx * dy).tolist(), " dx²", (dx**2).tolist(), " dy²", (dy**2).tolist())
print("   sumas", dx.sum(), dy.sum(), (dx * dy).sum(), (dx**2).sum(), (dy**2).sum())
c = (dx * dy).sum() / (n - 1); sx, sy = math.sqrt((dx**2).sum() / 7), math.sqrt((dy**2).sum() / 7)
print("3. cov", round(c, 4), " con n en vez de n − 1:", (dx * dy).sum() / n)
r = c / (sx * sy); print("4. sx", round(sx, 4), " sy", round(sy, 4), " sx·sy", round(sx * sy, 4), " r", round(r, 4), " √(268·74)", round(math.sqrt(268 * 74), 2))
print("5. en kg: cov", round(c * 1000, 1), " sx", round(sx * 1000, 1), " r", round(c * 1000 / (sx * 1000 * sy), 4))
b = 130 / 268; a = y.mean() - b * x.mean()
print("6. pendiente", round(b, 4), " ordenada", round(a, 3), " r²", round(r * r, 4), " 100 000 km:", round(b * 1000, 1), "L", round(b * 1000 * 1.65, 1), "€",
      "  camión de 20 t:", round(a + b * 20, 2))
rx, ry = rangos(x), rangos(y); d = rx - ry
print("7. puestos x", rx.tolist(), " y", ry.tolist(), " d", d.tolist(), " Σd²", (d**2).sum(),
      " 1 − 6Σd²/(n(n²−1))", round(1 - 6 * (d**2).sum() / (n * (n * n - 1)), 4), " Pearson de los puestos", round(np.corrcoef(rx, ry)[0, 1], 4), " scipy", round(stats.spearmanr(x, y).statistic, 4))
t = r * math.sqrt((n - 2) / (1 - r * r)); z = math.atanh(r); h = 1.959964 / math.sqrt(n - 3)
print("8. t", round(t, 3), " p", round(2 * stats.t.sf(t, n - 2), 5), " atanh(r)", round(z, 4), " 1/√5", round(1 / math.sqrt(5), 4), " h", round(h, 4),
      " [", round(z - h, 4), round(z + h, 4), "] → [", round(math.tanh(z - h), 3), round(math.tanh(z + h), 3), "]")
