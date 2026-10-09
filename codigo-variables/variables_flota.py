import numpy as np
from sim import viajes                                       # simulador (incluido en la descarga)
from prep import ciclo, media_pasada
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, TargetEncoder, FunctionTransformer
from sklearn.model_selection import KFold
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error as MAE

d = viajes()
d["hist30"] = media_pasada(d.camion, d.dia, d.consumo, 30)   # solo viajes anteriores del mismo camión
tr = d.dia < 160                                               # separación temporal: 160 días / 80 días
d["hist30"] = d.hist30.fillna(d.consumo[tr].mean())          # sin historial: media del entrenamiento
prep = ColumnTransformer([
    ("tipo", OneHotEncoder(handle_unknown="ignore"), ["tipo"]),
    ("num", "passthrough", ["carga_t", "km", "hist30"]),
    ("hora", FunctionTransformer(lambda X: ciclo(np.asarray(X).ravel(), 24)), ["hora"]),
    ("cliente", TargetEncoder(smooth=10.0, target_type="continuous",
                              cv=KFold(5, shuffle=True, random_state=0)), ["cliente"]),
])
modelo = make_pipeline(prep, HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=0))
cols = ["tipo", "carga_t", "km", "hist30", "hora", "cliente"]
modelo.fit(d.loc[tr, cols], d.consumo[tr])                   # todo se ajusta SOLO con el entrenamiento
print(round(MAE(d.consumo[~tr], modelo.predict(d.loc[~tr, cols])), 3))
print(round(MAE(d.consumo[~tr], np.full((~tr).sum(), d.consumo[tr].mean())), 3))
