"""Datos de los widgets de la versión técnica (tembed.json)."""
import json
from sim import carga_consumo
x, y = carga_consumo()
json.dump(dict(reg=[x.tolist(), y.tolist()]), open("../tembed.json", "w"), separators=(",", ":"))
