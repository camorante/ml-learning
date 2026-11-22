"""Carga de los tres trozos del reparto ya preparados (lo usan casi todos los scripts)."""
from datos import cargar, filas, reparto
from rasgos import preparar


def trozos():
    D = cargar()
    F = {k: filas(D, k) for k in ("tr", "va", "te")}
    Etr, esc = preparar(F["tr"])
    E = dict(tr=Etr, va=preparar(F["va"], esc)[0], te=preparar(F["te"], esc)[0])
    return D, F, E, esc
