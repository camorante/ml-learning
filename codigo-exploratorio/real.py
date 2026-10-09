import numpy as np, pandas as pd
from sim import viajes
from eda import resumen, centinelas, fuera_de_rango, vif, spearman

df = viajes()                                                 # 2 400 viajes con errores de registro
num = df.select_dtypes("number").columns

# 1) resumen: media frente a mediana, asimetría
r = pd.DataFrame({c: resumen(df[c]) for c in ["distancia_km", "vel_media", "temp_motor", "carga_t", "consumo"]}).T
print(r[["media", "mediana", "min", "max", "asimetria"]].round(2))

# 2) valores imposibles y centinelas
reglas = {"vel_media": (5, 130), "temp_motor": (-10, 130), "carga_t": (0, 26)}
print("fuera de rango:", fuera_de_rango(df, reglas))
print("centinelas:", {c: centinelas(df[c]) for c in ["vel_media", "temp_motor"]})
malo = np.zeros(len(df), bool)
for c, (lo, hi) in reglas.items(): malo |= (df[c] < lo) | (df[c] > hi)
limpio = df[~malo]
print(f"filas marcadas: {malo.sum()} ({malo.mean():.1%})")
print("vel_media  media con errores %.1f  sin errores %.1f" % (df.vel_media.mean(), limpio.vel_media.mean()))
print("r(vel, consumo)  con errores %.3f  sin errores %.3f  Spearman con errores %.3f" % (
    df.vel_media.corr(df.consumo), limpio.vel_media.corr(limpio.consumo), spearman(df.vel_media, df.consumo)))

# 3) asimetría y logaritmo
print("asimetría distancia %.2f  log(distancia) %.2f" % (limpio.distancia_km.skew(), np.log(limpio.distancia_km).skew()))

# 4) pares redundantes y VIF
c = limpio[num].corr().abs().where(lambda m: np.triu(np.ones(m.shape, bool), 1)).stack().sort_values(ascending=False)
print(c.head(4).round(3))
X = ["distancia_km", "km_gps", "vel_media", "duracion_h", "carga_t", "ralenti_pct", "temp_motor"]
print("VIF:", {k: float(v) for k, v in zip(X, vif(limpio[X]).round(1))})
X2 = [x for x in X if x not in ("km_gps", "duracion_h")]
print("VIF sin km_gps ni duracion:", {k: float(v) for k, v in zip(X2, vif(limpio[X2]).round(2))})

# 5) Simpson: pendiente global frente a pendiente por tipo
pend = lambda g: np.polyfit(g.vel_media, g.consumo, 1)[0]
print("pendiente global %.3f" % pend(limpio), {str(t): round(float(pend(g)), 3) for t, g in limpio.groupby("tipo")})
print(limpio.groupby("tipo")[["vel_media", "consumo", "carga_t"]].mean().round(1))
