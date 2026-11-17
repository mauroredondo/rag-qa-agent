"""El modelo de lenguaje que redacta la respuesta (Gemini).

El agente no habla con Gemini directamente: recibe una funcion `llm(prompt)`
que devuelve una `Respuesta`. Asi se puede probar con un modelo falso, o
cambiar de proveedor sin tocar el agente.
"""
import os
import time

from dotenv import load_dotenv
from pydantic import BaseModel, Field


class Respuesta(BaseModel):
    """Formato que debe devolver el modelo (salida estructurada)."""
    responde: bool = Field(description="False si los fragmentos no contienen la respuesta")
    respuesta: str = Field(description="Respuesta breve en español, solo con informacion de los fragmentos")
    citas: list[int] = Field(description="Numeros de los fragmentos usados, tal como aparecen entre corchetes")
    cita_textual: str = Field(description="Frase copiada LITERALMENTE de uno de los fragmentos citados que respalda la respuesta")


def crear_llm(config, espera=30, intentos=5):
    """Devuelve llm(prompt) -> Respuesta usando Gemini con salida estructurada."""
    from google import genai
    from google.genai import errors, types
    load_dotenv()
    cliente = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    modelo = config["modelo_generacion"]
    ajustes = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=Respuesta,
        temperature=0,  # mismas entradas, mismas respuestas (en lo posible)
    )

    def llm(prompt):
        for intento in range(intentos):
            try:
                resp = cliente.models.generate_content(
                    model=modelo, contents=prompt, config=ajustes)
                if resp.parsed is None:
                    raise ValueError("Gemini no devolvio una respuesta con el formato pedido")
                return resp.parsed
            except errors.APIError as e:
                # 429: limite de peticiones; 503: servidor saturado. Se reintenta.
                if e.code not in (429, 500, 503) or intento == intentos - 1:
                    raise
                time.sleep(espera * (intento + 1))
    return llm
