"""Paso 3b: el indice de busqueda.

Es una matriz de vectores (una fila por fragmento) mas la lista de fragmentos.
Buscar = convertir la pregunta en vector y multiplicar por la matriz: cada
elemento del resultado es la similitud coseno con un fragmento.
"""
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .carga import cargar_documentos
from .troceado import Trozo, trocear_documentos


def firma_indice(config):
    """Lo que debe coincidir para poder reutilizar un indice guardado."""
    from .embeddings import modelo_activo
    return {
        "proveedor": config.get("proveedor_embeddings", "gemini"),
        "modelo": modelo_activo(config),
        "tamano_trozo": config["tamano_trozo"],
        "solapamiento": config["solapamiento"],
    }


class Indice:
    def __init__(self, trozos, vectores, embedder):
        """embedder(textos, tarea) -> matriz normalizada. Se inyecta para poder
        probar el indice sin llamar a ninguna API."""
        self.trozos = trozos
        self.vectores = vectores
        self.embedder = embedder

    @classmethod
    def construir(cls, config, embedder):
        paginas = cargar_documentos(config["carpeta_documentos"])
        trozos = trocear_documentos(paginas, config["tamano_trozo"], config["solapamiento"])
        if not trozos:
            raise ValueError(f"No hay documentos en {config['carpeta_documentos']}")
        vectores = embedder([t.texto for t in trozos], "RETRIEVAL_DOCUMENT")
        return cls(trozos, vectores, embedder)

    def buscar(self, pregunta, k=5):
        """Devuelve los k fragmentos mas parecidos como lista de (trozo, similitud)."""
        consulta = self.embedder([pregunta], "RETRIEVAL_QUERY")[0]
        similitudes = self.vectores @ consulta
        mejores = np.argsort(-similitudes)[:k]
        return [(self.trozos[i], float(similitudes[i])) for i in mejores]

    def guardar(self, carpeta, firma=None):
        """Guarda el indice. `firma` describe como se construyo (proveedor, modelo,
        troceado): un indice solo es valido con el mismo modelo que lo creo."""
        carpeta = Path(carpeta)
        carpeta.mkdir(parents=True, exist_ok=True)
        (carpeta / "firma.json").write_text(json.dumps(firma or {}), encoding="utf-8")
        np.save(carpeta / "vectores.npy", self.vectores)
        (carpeta / "trozos.json").write_text(
            json.dumps([asdict(t) for t in self.trozos], ensure_ascii=False),
            encoding="utf-8")

    @classmethod
    def cargar(cls, carpeta, embedder, firma=None):
        carpeta = Path(carpeta)
        if firma is not None:
            guardada = json.loads((carpeta / "firma.json").read_text(encoding="utf-8"))
            if guardada != firma:
                raise ValueError(
                    "El indice guardado se creo con otra configuracion "
                    f"({guardada}) y no coincide con la actual ({firma}). "
                    "Reconstruyelo con --reconstruir.")
        vectores = np.load(carpeta / "vectores.npy")
        trozos = [Trozo(**d) for d in json.loads(
            (carpeta / "trozos.json").read_text(encoding="utf-8"))]
        return cls(trozos, vectores, embedder)
