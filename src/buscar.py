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
    parser.add_argument("--completo", action="store_true", help="mostrar el fragmento entero")
    args = parser.parse_args()
    config = cargar_config()
    indice = obtener_indice(config, args.reconstruir)
    if not args.pregunta:
        return
    modo = config.get("modo_busqueda", "denso")
    for trozo, puntos in indice.buscar(args.pregunta, config["k"], modo):
        print(f"\n[{puntos:.3f}] {trozo.fuente}, pag. {trozo.pagina}")
        corte = None if args.completo else 400
        print(trozo.texto[:corte] + ("..." if corte and len(trozo.texto) > corte else ""))


if __name__ == "__main__":
    main()
