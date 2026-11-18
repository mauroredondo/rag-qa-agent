# rag-qa-agent

🇬🇧 English | 🇪🇸 [Español](README.es.md)

Ask questions about a set of documents and get answers backed by a literal quote from them (RAG). The demo uses insurance documents: the terms of three policies and the Spanish insurance contract law.

It includes local embeddings, hybrid search (embeddings + BM25), a LangGraph agent that verifies its citations, an MCP server, an HTTP API with FastAPI, a small web page and an evaluation set.

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-local.txt
echo "GEMINI_API_KEY=your_key" > .env
# put the PDFs in documentos/ (see FUENTES.md)
python -m src.buscar --reconstruir
python -m src.api                     # then open http://127.0.0.1:8000
```

## Use it with your own documents

The code knows nothing about insurance. Everything that depends on the domain lives in `config.yaml` and the `documentos/` folder, so switching topic takes three steps:

1. Replace the files in `documentos/` (PDF, `.txt` or `.md`) and edit `dominio` and `descripcion` in `config.yaml`.
2. Rebuild the index with `python -m src.buscar --reconstruir`.
3. To measure it, write new questions in `evaluacion/preguntas.json`. Each one needs a `fuente` and an `evidencia` phrase that exists in that document; the current ones only fit the insurance documents.

## How it works

1. The loader reads the documents and keeps the file name and page of every text, so answers can cite them.
2. Each page is split into chunks of about 900 characters with 150 of overlap.
3. A local model (`intfloat/multilingual-e5-base`) turns each chunk into a vector. The Gemini embeddings API can be used instead.
4. Search combines embeddings (meaning) and BM25 (exact words) with Reciprocal Rank Fusion. Each chunk is indexed with its document name in front, so a question that mentions "Caser" can find it.
5. A LangGraph agent writes the answer and must quote the documents literally. Plain code checks the quote; if it is not in the cited fragment, the model gets one more try, and then the agent refuses.

The index is saved in `indice/` with a signature (model and chunking settings) and refuses to load if the configuration changes, so vectors from different models are never mixed.

## Project layout

```
config.yaml          everything that depends on the domain and the models
documentos/          the documents to index (PDFs are not in the repo, see FUENTES.md)
indice/              saved vectors and chunks (generated, not in the repo)
evaluacion/          evaluation questions, results and the agent's answers
src/                 carga, troceado, embeddings, bm25, indice, buscar, llm, agente, mcp_server, api, evaluacion*
tests/               unit tests (no API key and no model download needed)
```

## Setup

Tested with Python 3.13 on macOS (Apple silicon). `pip install -r requirements-local.txt` installs everything, including PyTorch. The first run downloads the embedding model from Hugging Face (about 1.1 GB, pinned revision, `safetensors` format); after that it runs offline. The Gemini key in `.env` is only needed for the agent.

Without PyTorch, use `requirements.txt` and set `proveedor_embeddings: gemini`, but the Gemini free tier (about 1,000 embeddings per day) cannot index the demo corpus in one go.

## Ways to use it

| How | Command | What you get |
|---|---|---|
| Search | `python -m src.buscar "question"` | The 5 most relevant fragments, with document and page. Add `--completo` for the full text. |
| Agent | `python -m src.agente "question"` | An answer with a verified quote and its source, or a refusal. |
| Web page | `python -m src.api`, then open <http://127.0.0.1:8000> | A page to ask questions and see the highlighted quote. |
| HTTP API | `python -m src.api` | `GET /salud`, `GET /documentos`, `POST /buscar`, `POST /preguntar`. Interactive docs at `/docs`. |
| MCP server | `python -m src.mcp_server` | Three read-only tools for AI clients such as Claude. |

```bash
curl -X POST http://127.0.0.1:8000/preguntar -H 'Content-Type: application/json' \
     -d '{"pregunta": "¿Cuántos días tengo para desistir del seguro de ING?"}'
```

The API validates input with Pydantic (HTTP 422 for an empty or too long question) and returns 503 without a Gemini key. It only accepts connections from the same computer, because it has no login yet.

### MCP server in Claude Desktop

The server exposes `buscar_en_documentos`, `listar_documentos` and `preguntar_a_documentos` (the last one only with a Gemini key). Add this to `claude_desktop_config.json` (replace the path) and restart the app:

```json
{
  "mcpServers": {
    "rag-qa-agent": {
      "command": "sh",
      "args": ["-c", "cd /path/to/rag-qa-agent && HF_HUB_OFFLINE=1 exec .venv/bin/python -m src.mcp_server"]
    }
  }
}
```

It has been used from Claude Desktop on macOS. It has not been tried in Claude Code yet.

## Evaluation

34 questions in `evaluacion/preguntas.json`: 28 with a verified evidence phrase and 6 the documents cannot answer.

```bash
python -m src.evaluacion --detalle      # retrieval only: seconds
python -m src.evaluacion_agente         # full agent: ~12 min on the Gemini free tier
```

| | |
|---|---|
| Retrieval, embeddings only | 71 % hit@5, MRR 0.53 |
| Retrieval, hybrid + document-name prefix | 86 % hit@5, MRR 0.67 |
| Agent answers answerable questions | 26 / 28 |
| Agent refuses unanswerable questions | 6 / 6 |
| Answers matching the documents (manual review) | 26 / 26 |

The questions were written by the author and the search was tuned with them, so the figures are optimistic and the confidence intervals are wide. Details in [`evaluacion/RESULTADOS.md`](evaluacion/RESULTADOS.md).

## How the project evolved

| Step | Problem | Decision |
|---|---|---|
| 1 | First version with Gemini embeddings | Keep the code independent of domain and provider |
| 2 | Gemini free tier: 1,000 embeddings a day for ~1,050 chunks | Local embeddings: no limits, offline, index in ~20 s |
| 3 | Vectors from different models cannot be mixed | Index signature, pinned model, audited dependencies |
| 4 | `FUENTES.md` was being indexed as a document | Moved it out of `documentos/` |
| 5 | Baseline at 71 % hit@5, missing exact words and insurer names | Hybrid search and document-name prefix: 86 % |
| 6 | Stop-word filter dropped "está" | Fixed and covered by a test |
| 7 | An LLM can invent answers | Agent that must quote literally, checked by code |
| 8 | Manual review found a wrong expected answer in the test set | Fixed it; the metric was not changed afterwards |
| 9 | Only usable from Python | MCP server with clear errors for invalid input |
| 10 | Ordinary programs need HTTP | FastAPI service |
| 11 | `/docs` is for developers | A one-page web interface |
| 12 | MCP only tested with a Python client | Used from Claude Desktop |

### What is still open

- A second evaluation set written by someone else.
- Two answerable questions are wrongly refused because the search misses the fragment; a cross-encoder re-ranker is the next thing to try.
- The API and the MCP server only work on the same computer and have no login. That is fine for a personal tool; publishing them online would need authentication and request limits.

## Tests

```bash
python -m pytest
```

52 tests with a fake embedder, a fake LLM, an in-memory MCP client and FastAPI's test client, so they need no API key and no model download.
