"""BM25: busqueda por palabras clave (lo contrario de los embeddings).

Los embeddings entienden el significado pero pueden fallar con palabras exactas
(numeros de articulo, cifras, nombres propios). BM25 puntua cada fragmento segun
cuantas palabras de la pregunta contiene, dando mas peso a las palabras raras
(las que aparecen en pocos fragmentos) y penalizando los fragmentos muy largos.
"""
import math
import re
import unicodedata
from collections import Counter

import numpy as np

# Palabras muy frecuentes que no ayudan a distinguir fragmentos.
PARADAS = {
    "de", "la", "el", "los", "las", "que", "en", "y", "a", "un", "una", "por",
    "para", "con", "se", "del", "al", "es", "lo", "su", "sus", "o", "como", "si",
    "no", "qué", "cual", "cuál", "cuando", "cuándo", "cuanto", "cuánto", "cuantos",
    "cuántos", "hay", "puede", "ser", "son", "tiene",
}


def _sin_acentos(texto):
    return "".join(c for c in unicodedata.normalize("NFD", texto)
                   if unicodedata.category(c) != "Mn")


# Se comparan sin acentos, porque los textos tambien se normalizan asi. Por eso la
# lista no incluye palabras como "esta" (pronombre) que chocarian con "esta" (verbo).
_PARADAS_NORM = {_sin_acentos(p) for p in PARADAS}


def _raiz(palabra):
    """Quita el plural: 'joyas' y 'joya' cuentan como la misma palabra."""
    if len(palabra) > 4 and palabra.endswith("es"):
        return palabra[:-2]
    if len(palabra) > 3 and palabra.endswith("s"):
        return palabra[:-1]
    return palabra


def tokenizar(texto):
    texto = _sin_acentos(texto.lower())
    return [_raiz(p) for p in re.findall(r"\w+", texto) if p not in _PARADAS_NORM]


class BM25:
    def __init__(self, textos, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        documentos = [tokenizar(t) for t in textos]
        self.n = len(documentos)
        self.longitudes = np.array([len(d) for d in documentos], dtype=np.float32)
        self.media = float(self.longitudes.mean()) if self.n else 0.0
        self.postings = {}  # palabra -> lista de (fragmento, veces que aparece)
        for i, doc in enumerate(documentos):
            for palabra, veces in Counter(doc).items():
                self.postings.setdefault(palabra, []).append((i, veces))

    def _idf(self, palabra):
        df = len(self.postings.get(palabra, ()))
        return math.log(1 + (self.n - df + 0.5) / (df + 0.5))

    def puntuaciones(self, consulta):
        """Devuelve un array con la puntuacion BM25 de cada fragmento."""
        puntos = np.zeros(self.n, dtype=np.float32)
        for palabra in set(tokenizar(consulta)):
            idf = self._idf(palabra)
            for i, veces in self.postings.get(palabra, ()):
                norma = self.k1 * (1 - self.b + self.b * self.longitudes[i] / self.media)
                puntos[i] += idf * veces * (self.k1 + 1) / (veces + norma)
        return puntos
