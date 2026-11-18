"""API HTTP con FastAPI: el buscador y el agente como servicio web.

Cualquier programa que sepa hacer peticiones HTTP (una web, una app, otro
servicio, curl) puede usarlo, sin saber Python.

Endpoints:
    GET  /             pagina web para preguntar (interfaz)
    GET  /salud        estado del servicio
    GET  /documentos   documentos indexados
    POST /buscar       fragmentos relevantes (sin LLM)
    POST /preguntar    respuesta con cita verificada o rechazo (necesita GEMINI_API_KEY)

Documentacion interactiva generada sola en http://127.0.0.1:8000/docs

Uso:
    python -m src.api
"""
from collections import Counter
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from .agente import preguntar

MAX_PREGUNTA = 1000
MAX_K = 10
PAGINA = Path(__file__).parent / "web" / "index.html"


# --- Formatos de entrada y salida ---------------------------------------------
# FastAPI usa estas clases para validar lo que llega (y responder 422 si algo no
# cumple) y para describir la API en /docs.

class PeticionBuscar(BaseModel):
    pregunta: str = Field(min_length=1, max_length=MAX_PREGUNTA, examples=["¿Está cubierto el robo de joyas?"])
    k: int = Field(default=5, ge=1, le=MAX_K, description="Cuántos fragmentos devolver")


class PeticionPreguntar(BaseModel):
    pregunta: str = Field(min_length=1, max_length=MAX_PREGUNTA,
                          examples=["¿Cuántos días tengo para desistir del seguro de ING?"])


class Fragmento(BaseModel):
    fuente: str
    pagina: int
    puntuacion: float
    texto: str


class Fuente(BaseModel):
    fuente: str
    pagina: int
    trozo_id: int


class Resultado(BaseModel):
    estado: Literal["respondida", "rechazada"]
    respuesta: str
    cita_textual: str | None = None
    motivo: str | None = None
    fuentes: list[Fuente]


class Documento(BaseModel):
    fuente: str
    fragmentos: int


class Salud(BaseModel):
    estado: Literal["ok"]
    documentos: int
    fragmentos: int
    agente_disponible: bool


def crear_app(indice, config, agente=None):
    """Monta la aplicacion. Se le pasan el indice y el agente (inyeccion de
    dependencias), igual que al servidor MCP: asi los tests usan versiones falsas."""
    app = FastAPI(title="rag-qa-agent",
                  description="Preguntas sobre documentos con citas verificadas.",
                  version="0.1.0")
    modo = config.get("modo_busqueda", "denso")

    def _limpiar(pregunta):
        pregunta = pregunta.strip()
        if not pregunta:
            raise HTTPException(status_code=422, detail="La pregunta no puede estar vacía.")
        return pregunta

    pagina = PAGINA.read_text(encoding="utf-8")

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def inicio():
        """La interfaz: una pagina que llama a /preguntar y /buscar desde el navegador."""
        return pagina

    @app.get("/salud", response_model=Salud)
    def salud():
        return Salud(estado="ok", documentos=len({t.fuente for t in indice.trozos}),
                     fragmentos=len(indice.trozos), agente_disponible=agente is not None)

    @app.get("/documentos", response_model=list[Documento])
    def documentos():
        cuenta = Counter(t.fuente for t in indice.trozos)
        return [Documento(fuente=f, fragmentos=n) for f, n in sorted(cuenta.items())]

    # Funciones normales (def, no async def): FastAPI las ejecuta en otro hilo, asi
    # una busqueda o una llamada a Gemini lenta no bloquea al resto de peticiones.
    @app.post("/buscar", response_model=list[Fragmento])
    def buscar(peticion: PeticionBuscar):
        pregunta = _limpiar(peticion.pregunta)
        return [Fragmento(fuente=t.fuente, pagina=t.pagina, puntuacion=round(p, 4), texto=t.texto)
                for t, p in indice.buscar(pregunta, peticion.k, modo)]

    @app.post("/preguntar", response_model=Resultado)
    def preguntar_endpoint(peticion: PeticionPreguntar):
        if agente is None:
            raise HTTPException(status_code=503,
                                detail="El agente no está disponible: falta GEMINI_API_KEY.")
        r = preguntar(agente, _limpiar(peticion.pregunta))
        return Resultado(estado=r["estado"], respuesta=r["respuesta"],
                         cita_textual=r.get("cita_textual"), motivo=r.get("motivo"),
                         fuentes=[Fuente(**f) for f in r["fuentes"]])

    return app


def crear_app_desde_config():
    """Carga el indice y, si hay clave, el agente. La usa uvicorn con --factory."""
    import os

    from dotenv import load_dotenv

    from .agente import construir_agente
    from .buscar import obtener_indice
    from .config import RAIZ, cargar_config
    from .llm import crear_llm

    load_dotenv(RAIZ / ".env")
    config = cargar_config()
    indice = obtener_indice(config)
    agente = None
    if os.environ.get("GEMINI_API_KEY"):
        agente = construir_agente(indice, crear_llm(config), config)
    return crear_app(indice, config, agente)


def main():
    import uvicorn
    # 127.0.0.1 = solo accesible desde tu ordenador, no desde la red.
    uvicorn.run("src.api:crear_app_desde_config", factory=True, host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
