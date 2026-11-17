"""Paso 2: cortar cada pagina en fragmentos (chunks) con solapamiento.

Un fragmento demasiado grande diluye su significado en el embedding; uno
demasiado pequeno pierde contexto. El solapamiento evita partir una idea por la
mitad: el final de un fragmento se repite al principio del siguiente.
"""
import re
from dataclasses import dataclass

from .carga import Pagina


@dataclass
class Trozo:
    id: int
    fuente: str
    pagina: int
    texto: str


def _normalizar(texto):
    # Los PDFs traen saltos de linea a mitad de frase y espacios repetidos.
    return re.sub(r"\s+", " ", texto).strip()


def _buscar_corte(texto, inicio, fin):
    """Intenta cortar al final de una frase; si no, en un espacio."""
    if fin >= len(texto):
        return len(texto)
    ventana = texto[inicio:fin]
    for separador in (". ", "; ", ": ", " "):
        pos = ventana.rfind(separador)
        if pos > len(ventana) * 0.5:  # no cortes demasiado pronto
            return inicio + pos + len(separador)
    return fin


def trocear_pagina(texto, tamano, solapamiento):
    """Devuelve la lista de fragmentos de un texto."""
    if solapamiento >= tamano:
        raise ValueError("El solapamiento debe ser menor que el tamano del trozo")
    texto = _normalizar(texto)
    trozos, inicio = [], 0
    while inicio < len(texto):
        fin = _buscar_corte(texto, inicio, inicio + tamano)
        trozo = texto[inicio:fin].strip()
        if trozo:
            trozos.append(trozo)
        if fin >= len(texto):
            break
        inicio = max(fin - solapamiento, inicio + 1)
    return trozos


def trocear_documentos(paginas: list[Pagina], tamano, solapamiento):
    """Trocea todas las paginas y numera los fragmentos."""
    resultado = []
    for pag in paginas:
        for texto in trocear_pagina(pag.texto, tamano, solapamiento):
            resultado.append(Trozo(len(resultado), pag.fuente, pag.pagina, texto))
    return resultado
