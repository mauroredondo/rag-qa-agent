"""Paso 3b: el indice de busqueda.

Es una matriz de vectores (una fila por fragmento) mas la lista de fragmentos.
Buscar = convertir la pregunta en vector y multiplicar por la matriz: cada
elemento del resultado es la similitud coseno con un fragmento.
"""
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .bm25 import BM25
from .carga import cargar_documentos
from .troceado import Trozo, trocear_documentos


def fusion_rrf(rankings, k_rrf=60):
    """Reciprocal Rank Fusion: mezcla varias listas ordenadas de fragmentos.

    Cada fragmento suma 1/(k_rrf + posicion) por cada lista en la que aparece.
    Usa solo las posiciones, asi no hay que comparar puntuaciones de escalas
    distintas (similitud coseno frente a BM25).
    """
    puntos = {}
    for ranking in rankings:
        for pos, i in enumerate(ranking, start=1):
            puntos[i] = puntos.get(i, 0.0) + 1.0 / (k_rrf + pos)
    return sorted(puntos.items(), key=lambda par: -par[1])


def firma_indice(config):
    """Lo que debe coincidir para poder reutilizar un indice guardado."""
    from .embeddings import modelo_activo
    return {
        "proveedor": config.get("proveedor_embeddings", "gemini"),
        "modelo": modelo_activo(config),
        "tamano_trozo": config["tamano_trozo"],
        "solapamiento": config["solapamiento"],
        "separador_estructura": config.get("separador_estructura"),
        "prefijo_fuente": bool(config.get("prefijo_fuente", False)),
    }


class Indice:
    def __init__(self, trozos, vectores, embedder, prefijo_fuente=False):
        """embedder(textos, tarea) -> matriz normalizada. Se inyecta para poder
        probar el indice sin llamar a ninguna API."""
        self.trozos = trozos
        self.vectores = vectores
        self.embedder = embedder
        self.prefijo_fuente = prefijo_fuente
        self._bm25 = None

    def texto_indexado(self, trozo):
        """Texto que ven los buscadores. Con `prefijo_fuente` se le antepone el
        nombre del documento: asi una pregunta que dice 'Caser' o 'ASISA' puede
        encontrar fragmentos que no repiten ese nombre en su texto."""
        if not self.prefijo_fuente:
            return trozo.texto
        return f"{trozo.fuente.rsplit('.', 1)[0].replace('_', ' ')}. {trozo.texto}"

    @classmethod
    def construir(cls, config, embedder):
        paginas = cargar_documentos(config["carpeta_documentos"])
        trozos = trocear_documentos(paginas, config["tamano_trozo"], config["solapamiento"],
                                  config.get("separador_estructura"))
        if not trozos:
            raise ValueError(f"No hay documentos en {config['carpeta_documentos']}")
        indice = cls(trozos, None, embedder, bool(config.get("prefijo_fuente", False)))
        indice.vectores = embedder([indice.texto_indexado(t) for t in trozos],
                                   "RETRIEVAL_DOCUMENT")
        return indice

    def buscar(self, pregunta, k=5, modo="denso"):
        """Devuelve los k fragmentos mejor valorados como lista de (trozo, puntuacion).

        modo: 'denso' (embeddings, similitud coseno), 'bm25' (palabras clave) o
        'hibrido' (los dos mezclados con RRF).
        """
        if modo == "denso":
            puntos = self._puntos_densos(pregunta)
            mejores = np.argsort(-puntos)[:k]
            return [(self.trozos[i], float(puntos[i])) for i in mejores]
        if modo == "bm25":
            puntos = self._puntos_bm25(pregunta)
            mejores = np.argsort(-puntos)[:k]
            return [(self.trozos[i], float(puntos[i])) for i in mejores]
        if modo == "hibrido":
            candidatos = 50  # cuantos fragmentos aporta cada buscador a la mezcla
            denso = np.argsort(-self._puntos_densos(pregunta))[:candidatos]
            palabras = np.argsort(-self._puntos_bm25(pregunta))[:candidatos]
            fusion = fusion_rrf([denso.tolist(), palabras.tolist()])[:k]
            return [(self.trozos[i], float(p)) for i, p in fusion]
        raise ValueError(f"Modo de busqueda desconocido: {modo!r}")

    def _puntos_densos(self, pregunta):
        consulta = self.embedder([pregunta], "RETRIEVAL_QUERY")[0]
        return self.vectores @ consulta

    def _puntos_bm25(self, pregunta):
        if self._bm25 is None:
            self._bm25 = BM25([self.texto_indexado(t) for t in self.trozos])
        return self._bm25.puntuaciones(pregunta)

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
        return cls(trozos, vectores, embedder, bool((firma or {}).get("prefijo_fuente", False)))
