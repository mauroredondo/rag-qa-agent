"""Paso 1: leer los documentos de una carpeta.

Devuelve una lista de paginas. Cada pagina recuerda de que archivo viene y su
numero, para poder citarla despues.
"""
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader


@dataclass
class Pagina:
    fuente: str   # nombre del archivo
    pagina: int   # empieza en 1
    texto: str


def leer_pdf(ruta):
    lector = PdfReader(str(ruta))
    paginas = []
    for i, p in enumerate(lector.pages, start=1):
        texto = (p.extract_text() or "").strip()
        if texto:
            paginas.append(Pagina(Path(ruta).name, i, texto))
    return paginas


def leer_texto(ruta):
    texto = Path(ruta).read_text(encoding="utf-8").strip()
    return [Pagina(Path(ruta).name, 1, texto)] if texto else []


def cargar_documentos(carpeta):
    """Lee todos los .pdf, .txt y .md de la carpeta (en orden alfabetico)."""
    paginas = []
    for ruta in sorted(Path(carpeta).iterdir()):
        sufijo = ruta.suffix.lower()
        if sufijo == ".pdf":
            paginas += leer_pdf(ruta)
        elif sufijo in (".txt", ".md"):
            paginas += leer_texto(ruta)
    return paginas
