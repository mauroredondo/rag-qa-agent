"""Paso 3a: convertir texto en vectores (embeddings) con Gemini.

Un embedding es un vector de cientos de numeros; textos con significado parecido
quedan cerca. Normalizamos cada vector a longitud 1: asi el producto escalar de
dos vectores ES su similitud coseno.
"""
import os
import time

import numpy as np
from dotenv import load_dotenv

LOTE = 50    # textos por peticion
PAUSA = 32   # segundos entre lotes (limite gratuito: ~100 textos por minuto)


def _cliente():
    from google import genai
    load_dotenv()
    return genai.Client(api_key=os.environ["GEMINI_API_KEY"])


def normalizar(matriz):
    matriz = np.asarray(matriz, dtype=np.float32)
    normas = np.linalg.norm(matriz, axis=1, keepdims=True)
    normas[normas == 0] = 1.0
    return matriz / normas


def embeber(textos, modelo, tarea="RETRIEVAL_DOCUMENT"):
    """Devuelve una matriz (n_textos, dimension) de vectores normalizados.

    tarea: RETRIEVAL_DOCUMENT para los fragmentos, RETRIEVAL_QUERY para la pregunta.
    """
    from google.genai import errors, types
    cliente = _cliente()
    vectores = []
    for i in range(0, len(textos), LOTE):
        lote = textos[i:i + LOTE]
        for intento in range(6):
            try:
                resp = cliente.models.embed_content(
                    model=modelo,
                    contents=lote,
                    config=types.EmbedContentConfig(task_type=tarea),
                )
                break
            except errors.ClientError as e:
                if e.code != 429 or intento == 5:
                    raise
                time.sleep(65)  # el plan gratuito limita las peticiones por minuto
        vectores += [e.values for e in resp.embeddings]
        if len(textos) > LOTE:
            print(f"  embeddings: {min(i + LOTE, len(textos))}/{len(textos)}", flush=True)
            time.sleep(PAUSA)  # evita pasarse del limite entre lotes
    return normalizar(vectores)


# --- Proveedor local (sin API, los documentos no salen del ordenador) ----------

_modelos_locales = {}


def embeber_local(textos, modelo, tarea="RETRIEVAL_DOCUMENT"):
    """Embeddings con un modelo que se ejecuta en tu ordenador.

    Requiere `pip install sentence-transformers`. Los modelos de la familia E5
    esperan un prefijo distinto para preguntas y para documentos.
    """
    from sentence_transformers import SentenceTransformer
    if modelo not in _modelos_locales:
        _modelos_locales[modelo] = SentenceTransformer(modelo)
    prefijo = "query: " if tarea == "RETRIEVAL_QUERY" else "passage: "
    if "e5" not in modelo.lower():
        prefijo = ""
    vectores = _modelos_locales[modelo].encode(
        [prefijo + t for t in textos], batch_size=32, show_progress_bar=len(textos) > 50)
    return normalizar(vectores)


def crear_embedder(config):
    """Devuelve embedder(textos, tarea) segun `proveedor_embeddings` del config."""
    proveedor = config.get("proveedor_embeddings", "gemini")
    if proveedor == "gemini":
        modelo = config["modelo_embeddings"]
        return lambda textos, tarea: embeber(textos, modelo, tarea)
    if proveedor == "local":
        modelo = config["modelo_embeddings_local"]
        return lambda textos, tarea: embeber_local(textos, modelo, tarea)
    raise ValueError(f"Proveedor de embeddings desconocido: {proveedor!r} (usa 'gemini' o 'local')")


def modelo_activo(config):
    """Nombre del modelo de embeddings que se usara con esta configuracion."""
    if config.get("proveedor_embeddings", "gemini") == "local":
        return config["modelo_embeddings_local"]
    return config["modelo_embeddings"]
