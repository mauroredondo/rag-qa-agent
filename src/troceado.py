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


def _ventana(texto, tamano, solapamiento):
    """Ventana deslizante: trozos de `tamano` caracteres que se solapan."""
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


def trocear_pagina(texto, tamano, solapamiento, separador=None, minimo=200):
    """Devuelve la lista de fragmentos de un texto.

    separador: expresion regular opcional que marca el INICIO de una seccion
    (por ejemplo 'Articulo quince.'). Si se da, se corta primero por secciones,
    para que un fragmento no mezcle el final de un articulo con el principio del
    siguiente; las secciones muy cortas (< minimo) se unen a la siguiente y las
    muy largas se parten con la ventana deslizante.
    """
    if solapamiento >= tamano:
        raise ValueError("El solapamiento debe ser menor que el tamano del trozo")
    texto = _normalizar(texto)
    if not separador:
        return _ventana(texto, tamano, solapamiento)
    secciones = [s for s in re.split(f"(?={separador})", texto) if s.strip()]
    trozos, buffer = [], ""
    for seccion in secciones + [None]:
        if seccion is not None and len(buffer) < minimo and len(buffer) + len(seccion) <= tamano:
            buffer += seccion  # seccion corta: se une para no perder contexto
            continue
        if buffer.strip():
            trozos += _ventana(buffer.strip(), tamano, solapamiento)
        buffer = seccion or ""
    return trozos


def trocear_documentos(paginas: list[Pagina], tamano, solapamiento, separador=None):
    """Trocea todas las paginas y numera los fragmentos."""
    resultado = []
    for pag in paginas:
        for texto in trocear_pagina(pag.texto, tamano, solapamiento, separador):
            resultado.append(Trozo(len(resultado), pag.fuente, pag.pagina, texto))
    return resultado
