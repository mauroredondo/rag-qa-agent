"""Carga de config.yaml."""
from pathlib import Path

import yaml

RAIZ = Path(__file__).resolve().parent.parent


def cargar_config(ruta=None):
    """Lee config.yaml y devuelve un diccionario."""
    ruta = Path(ruta) if ruta else RAIZ / "config.yaml"
    with open(ruta, encoding="utf-8") as f:
        return yaml.safe_load(f)
