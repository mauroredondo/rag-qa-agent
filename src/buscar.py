"""Uso desde terminal:

    python -m src.buscar --reconstruir          # (re)crea el indice
    python -m src.buscar "¿Que cubre el seguro de hogar en caso de robo?"
"""
import argparse
from pathlib import Path

from .config import RAIZ, cargar_config
from .embeddings import crear_embedder
from .indice import Indice, firma_indice


def obtener_indice(config, reconstruir=False):
    embedder = crear_embedder(config)
    firma = firma_indice(config)
    carpeta_indice = RAIZ / config["carpeta_indice"]
    config = {**config, "carpeta_documentos": RAIZ / config["carpeta_documentos"]}
    if not reconstruir and (carpeta_indice / "vectores.npy").exists():
        return Indice.cargar(carpeta_indice, embedder, firma)
    indice = Indice.construir(config, embedder)
    indice.guardar(carpeta_indice, firma)
    print(f"Indice creado: {len(indice.trozos)} fragmentos.")
    return indice


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pregunta", nargs="?")
    parser.add_argument("--reconstruir", action="store_true")
    args = parser.parse_args()
    config = cargar_config()
    indice = obtener_indice(config, args.reconstruir)
    if not args.pregunta:
        return
    for trozo, sim in indice.buscar(args.pregunta, config["k"]):
        print(f"\n[{sim:.3f}] {trozo.fuente}, pag. {trozo.pagina}")
        print(trozo.texto[:400] + ("..." if len(trozo.texto) > 400 else ""))


if __name__ == "__main__":
    main()
