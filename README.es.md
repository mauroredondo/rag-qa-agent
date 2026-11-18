# rag-qa-agent

🇬🇧 [English](README.md) | 🇪🇸 Español

Haz preguntas sobre un conjunto de documentos y recibe respuestas respaldadas por una cita literal de ellos (RAG). La demo usa documentos de seguros: las condiciones de tres pólizas y la Ley de Contrato de Seguro.

Incluye embeddings locales, búsqueda híbrida (embeddings + BM25), un agente en LangGraph que verifica sus citas, un servidor MCP, una API HTTP con FastAPI, una página web sencilla y un conjunto de evaluación.

## Inicio rápido

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-local.txt
echo "GEMINI_API_KEY=tu_clave" > .env
# pon los PDFs en documentos/ (ver FUENTES.md)
python -m src.buscar --reconstruir
python -m src.api                     # y abre http://127.0.0.1:8000
```

## Úsalo con tus propios documentos

El código no sabe nada de seguros. Todo lo que depende del dominio está en `config.yaml` y en la carpeta `documentos/`, así que cambiar de tema son tres pasos:

1. Cambia los archivos de `documentos/` (PDF, `.txt` o `.md`) y edita `dominio` y `descripcion` en `config.yaml`.
2. Reconstruye el índice con `python -m src.buscar --reconstruir`.
3. Para medirlo, escribe preguntas nuevas en `evaluacion/preguntas.json`. Cada una necesita una `fuente` y una frase de `evidencia` que exista en ese documento; las actuales solo sirven para los documentos de seguros.

## Cómo funciona

1. El cargador lee los documentos y guarda el nombre del archivo y la página de cada texto, para que las respuestas puedan citarlos.
2. Cada página se corta en fragmentos de unos 900 caracteres con 150 de solapamiento.
3. Un modelo local (`intfloat/multilingual-e5-base`) convierte cada fragmento en un vector. También se puede usar la API de embeddings de Gemini.
4. La búsqueda combina embeddings (significado) y BM25 (palabras exactas) con Reciprocal Rank Fusion. Cada fragmento se indexa con el nombre de su documento delante, para que una pregunta que nombra "Caser" lo encuentre.
5. Un agente en LangGraph redacta la respuesta y tiene que citar los documentos literalmente. Un código normal comprueba la cita; si no está en el fragmento citado, el modelo tiene otro intento y después el agente se niega a responder.

El índice se guarda en `indice/` con una firma (modelo y ajustes de troceado) y no carga si cambia la configuración, así nunca se mezclan vectores de modelos distintos.

## Estructura

```
config.yaml          todo lo que depende del dominio y de los modelos
documentos/          los documentos que se indexan (los PDFs no están en el repo, ver FUENTES.md)
indice/              vectores y fragmentos guardados (se generan, no están en el repo)
evaluacion/          preguntas de evaluación, resultados y respuestas del agente
src/                 carga, troceado, embeddings, bm25, indice, buscar, llm, agente, mcp_server, api, evaluacion*
tests/               tests (no necesitan clave ni descargar el modelo)
```

## Instalación

Probado con Python 3.13 en macOS (Apple silicon). `pip install -r requirements-local.txt` instala todo, PyTorch incluido. La primera vez se descarga de Hugging Face el modelo de embeddings (unos 1,1 GB, versión fijada, formato `safetensors`); después funciona sin conexión. La clave de Gemini en `.env` solo hace falta para el agente.

Sin PyTorch, usa `requirements.txt` y pon `proveedor_embeddings: gemini`, aunque el plan gratuito de Gemini (unos 1.000 embeddings al día) no basta para indexar el corpus de la demo de una vez.

## Formas de usarlo

| Cómo | Comando | Qué obtienes |
|---|---|---|
| Búsqueda | `python -m src.buscar "pregunta"` | Los 5 fragmentos más relevantes, con documento y página. Añade `--completo` para ver el texto entero. |
| Agente | `python -m src.agente "pregunta"` | Una respuesta con cita verificada y su fuente, o un rechazo. |
| Página web | `python -m src.api` y abre <http://127.0.0.1:8000> | Una página para preguntar y ver la cita resaltada. |
| API HTTP | `python -m src.api` | `GET /salud`, `GET /documentos`, `POST /buscar`, `POST /preguntar`. Documentación interactiva en `/docs`. |
| Servidor MCP | `python -m src.mcp_server` | Tres herramientas de solo lectura para clientes de IA como Claude. |

```bash
curl -X POST http://127.0.0.1:8000/preguntar -H 'Content-Type: application/json' \
     -d '{"pregunta": "¿Cuántos días tengo para desistir del seguro de ING?"}'
```

La API valida la entrada con Pydantic (HTTP 422 si la pregunta está vacía o es demasiado larga) y devuelve 503 si no hay clave de Gemini. Solo acepta conexiones desde el mismo ordenador, porque todavía no tiene inicio de sesión.

### Servidor MCP en Claude Desktop

El servidor ofrece `buscar_en_documentos`, `listar_documentos` y `preguntar_a_documentos` (esta última solo con clave de Gemini). Añade esto a `claude_desktop_config.json` (cambia la ruta) y reinicia la app:

```json
{
  "mcpServers": {
    "rag-qa-agent": {
      "command": "sh",
      "args": ["-c", "cd /ruta/a/rag-qa-agent && HF_HUB_OFFLINE=1 exec .venv/bin/python -m src.mcp_server"]
    }
  }
}
```

Se ha usado desde Claude Desktop en macOS. Todavía no se ha probado en Claude Code.

## Evaluación

34 preguntas en `evaluacion/preguntas.json`: 28 con una frase de evidencia comprobada y 6 que los documentos no pueden responder.

```bash
python -m src.evaluacion --detalle      # solo la búsqueda: segundos
python -m src.evaluacion_agente         # el agente completo: unos 12 min con el plan gratuito de Gemini
```

| | |
|---|---|
| Búsqueda, solo embeddings | 71 % acierto@5, MRR 0,53 |
| Búsqueda, híbrida + prefijo del documento | 86 % acierto@5, MRR 0,67 |
| El agente responde las preguntas con respuesta | 26 / 28 |
| El agente rechaza las preguntas sin respuesta | 6 / 6 |
| Respuestas fieles a los documentos (revisión manual) | 26 / 26 |

Las preguntas las escribió el autor y la búsqueda se ajustó con ellas, así que las cifras son optimistas y los intervalos de confianza son anchos. Detalles en [`evaluacion/RESULTADOS.md`](evaluacion/RESULTADOS.md).

## Cómo ha evolucionado el proyecto

| Paso | Problema | Decisión |
|---|---|---|
| 1 | Primera versión con embeddings de Gemini | Código independiente del dominio y del proveedor |
| 2 | Plan gratuito de Gemini: 1.000 embeddings al día para ~1.050 fragmentos | Embeddings locales: sin límites, sin conexión, índice en ~20 s |
| 3 | No se pueden mezclar vectores de modelos distintos | Firma del índice, modelo fijado, dependencias revisadas |
| 4 | `FUENTES.md` se indexaba como un documento | Se sacó de `documentos/` |
| 5 | Línea base en 71 % de acierto@5; fallaba con palabras exactas y nombres de aseguradoras | Búsqueda híbrida y prefijo del documento: 86 % |
| 6 | El filtro de palabras vacías eliminaba "está" | Corregido y cubierto por un test |
| 7 | Un modelo de lenguaje puede inventar respuestas | Agente que cita literalmente, comprobado por código |
| 8 | La revisión manual encontró una respuesta esperada mal escrita | Corregida; la métrica no se cambió después |
| 9 | Solo se podía usar desde Python | Servidor MCP con errores claros ante entradas inválidas |
| 10 | Los programas normales necesitan HTTP | Servicio FastAPI |
| 11 | `/docs` es para programadores | Una interfaz web de una página |
| 12 | El MCP solo se había probado con un cliente de Python | Usado desde Claude Desktop |

### Qué falta

- Un segundo conjunto de evaluación escrito por otra persona.
- Dos preguntas con respuesta se rechazan por error porque la búsqueda no encuentra el fragmento; lo siguiente que conviene probar es un re-ranker cross-encoder.
- La API y el servidor MCP solo funcionan en el propio ordenador y no tienen inicio de sesión. Para una herramienta personal está bien; publicarlos en internet exigiría autenticación y límites de peticiones.

## Tests

```bash
python -m pytest
```

52 tests con un embedder falso, un modelo de lenguaje falso, un cliente MCP en memoria y el cliente de pruebas de FastAPI, así que no necesitan clave ni descargar el modelo.
