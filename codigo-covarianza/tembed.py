"""Datos de los widgets de la versión técnica (tembed.json), sacados de embed.json."""
import json
E = json.load(open("../embed.json"))
json.dump(dict(vel=E["vel"], at=E["at"][0], at0=E["at0"][0]), open("../tembed.json", "w"), separators=(",", ":"))
