# rag-qa-agent

Question-answering over a set of documents (RAG), built to be reusable: the code knows nothing about the domain. Everything that changes between domains lives in `config.yaml` and the `documentos/` folder. The first demo domain is **insurance** (policy terms and the Spanish insurance contract law).

> Work in progress. Done: document loading, chunking, local embeddings, hybrid search (embeddings + BM25), retrieval evaluation and a LangGraph agent with verified citations, an MCP server and an HTTP API (FastAPI).

## How it works

1. **Load** PDFs / `.txt` / `.md` from `documentos/` and keep the file name and page of every text, so answers can cite them.
2. **Chunk** each page into ~900-character pieces with 150 characters of overlap.
3. **Embed** each chunk into a vector and normalise it, so the dot product is the cosine similarity. Default: a local model (`intfloat/multilingual-e5-base`, runs on your machine); the Gemini embeddings API is available as an option.
4. **Search** (`modo_busqueda` in `config.yaml`): `denso` multiplies the question vector by the matrix of chunk vectors; `bm25` scores keyword overlap; `hibrido` (default) merges both rankings with Reciprocal Rank Fusion. Each chunk is indexed with its document name in front, so questions that mention "Caser" or "ASISA" can find it.
5. **Answer** with an agent that must quote the documents literally (see [Agent](#agent)).

The index is saved in `indice/` with a signature (provider, model, chunking, prefix). If the configuration changes, loading it fails on purpose instead of mixing vectors from different models; rebuild it with `--reconstruir`.

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

Tested with Python 3.13 on macOS (Apple silicon).

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-local.txt     # includes PyTorch and sentence-transformers
echo "GEMINI_API_KEY=your_key" > .env     # only needed for the agent (answer generation)
```

Put the documents in `documentos/` (the ones used in the demo are listed in [`FUENTES.md`](FUENTES.md)). The first time, the embedding model (~1.1 GB, a pinned revision of `intfloat/multilingual-e5-base`, `safetensors` format) is downloaded from Hugging Face; after that everything runs offline.

If you do not want PyTorch, use `pip install -r requirements.txt` and set `proveedor_embeddings: gemini` in `config.yaml`. Note that the Gemini free tier limits embeddings (about 100 per minute and 1,000 per day), which is not enough to index the demo corpus in one go.

## Search

```bash
python -m src.buscar --reconstruir                       # (re)build the index: ~20 s locally
python -m src.buscar "¿Está cubierto el robo de joyas en casa?"
python -m src.buscar --completo "..."                    # print whole fragments
```

## Agent

```bash
python -m src.agente "¿Cuántos días tengo para desistir del seguro de ING?"
```

A LangGraph graph: `buscar → responder → verificar → finalizar`. The model must return the answer, the fragments it used and a **literal quote**; a plain-code step checks that the quote really appears in a cited fragment. If it does not, the model gets one more attempt with a warning, and otherwise the agent refuses instead of guessing. It also refuses when the model says the documents do not contain the answer. Document text is passed to the model as data, with an explicit instruction to ignore any orders found inside it.

Generation uses the Gemini API (`modelo_generacion`, pinned to a specific model rather than a `-latest` alias so results stay comparable). The LLM is injected into the agent, so tests use a fake one.

## MCP server

[MCP](https://modelcontextprotocol.io) lets any compatible client (Claude Desktop, Claude Code, your own agent) discover and call tools without custom code. This server exposes three **read-only** tools:

| Tool | What it does | Needs an API key |
|---|---|---|
| `buscar_en_documentos(pregunta, k)` | Returns the most relevant fragments with document and page. No LLM involved: the calling agent reasons over them. | No |
| `listar_documentos()` | Lists the indexed documents and their number of fragments. | No |
| `preguntar_a_documentos(pregunta)` | Answers with the verified-citation agent, or refuses. Only published if `GEMINI_API_KEY` is set. | Yes |

Run it (the client normally launches it as a subprocess and talks over stdin/stdout):

```bash
python -m src.mcp_server
```

To register it in a client, point it at the project's virtualenv. Example for Claude Code (replace the path):

```bash
claude mcp add rag-qa-agent -- sh -c "cd /path/to/rag-qa-agent && .venv/bin/python -m src.mcp_server"
```

and for Claude Desktop (`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "rag-qa-agent": {
      "command": "sh",
      "args": ["-c", "cd /path/to/rag-qa-agent && .venv/bin/python -m src.mcp_server"]
    }
  }
}
```

Design notes: every tool is annotated read-only; questions are limited to 1,000 characters and `k` to 10; invalid input returns a clear tool error instead of a crash; the transport is stdio, so nothing may be printed to stdout (startup output is redirected to stderr); the text returned is data from the documents, and the server instructions say so.

## HTTP API

The same search and agent as a web service, for any program that can make HTTP requests (a web page, an app, another service):

```bash
python -m src.api        # serves on http://127.0.0.1:8000 (this machine only)
```

| Method | Path | Body | Returns |
|---|---|---|---|
| GET | `/salud` | | status, number of documents and fragments, whether the agent is available |
| GET | `/documentos` | | indexed documents and their fragments |
| POST | `/buscar` | `{"pregunta": "...", "k": 5}` | relevant fragments with document and page (no LLM) |
| POST | `/preguntar` | `{"pregunta": "..."}` | answer with verified quote and sources, or a refusal |

```bash
curl -X POST http://127.0.0.1:8000/preguntar -H 'Content-Type: application/json' \
     -d '{"pregunta": "¿Cuántos días tengo para desistir del seguro de ING?"}'
```

Interactive documentation is generated automatically at <http://127.0.0.1:8000/docs>. Requests are validated with Pydantic models (empty or over-long questions and `k` outside 1-10 get HTTP 422); `/preguntar` returns 503 if there is no `GEMINI_API_KEY`. Endpoints are plain `def` functions, so FastAPI runs them in a thread pool and a slow Gemini call does not block other requests. It binds to `127.0.0.1` on purpose: there is no authentication or rate limiting yet, so it is not meant to be exposed to a network.

## Evaluation

`evaluacion/preguntas.json` has 28 questions with a verified evidence phrase and 6 questions the documents cannot answer.

```bash
python -m src.evaluacion --detalle      # retrieval only (search modes, chunk sizes): seconds
python -m src.evaluacion_agente         # full agent: ~12 min on the Gemini free tier
```

Results (n = 28 answerable + 6 unanswerable, so confidence intervals are wide; details and limits in [`evaluacion/RESULTADOS.md`](evaluacion/RESULTADOS.md)):

| | |
|---|---|
| Retrieval, dense only (baseline) | 71 % hit@5, MRR 0.53 |
| Retrieval, hybrid + document-name prefix | 86 % hit@5, MRR 0.67 |
| Agent answers answerable questions | 26 / 28 |
| Agent refuses unanswerable questions | 6 / 6 |
| Answers matching the documents (manual review) | 26 / 26 |

Caveats: the questions were written by the project author and the search was tuned while looking at them, so the retrieval figures are somewhat optimistic; the manual review should be repeated by someone else; the two wrong refusals come from search misses. The agent's answers are saved in [`evaluacion/respuestas_agente.md`](evaluacion/respuestas_agente.md).

## How the project evolved

Each step below was a decision made after measuring or hitting a real problem, not a plan fixed in advance.

| Step | What happened | Decision |
|---|---|---|
| 1. First version | Chunking, Gemini embeddings (`gemini-embedding-001`) and a cosine-similarity search with numpy. Unit tests with a fake embedder. | Keep the code independent of the domain and of the provider: the embedder is injected. |
| 2. Gemini free-tier limits | The first index build failed with HTTP 429 (about 100 embeddings per minute), and after adding waits and retries it hit the daily limit (1,000 embeddings per day) with a corpus of ~1,050 chunks. The partial progress was lost. | Switch to a **local embedding model** (`multilingual-e5-base`): no limits, works offline, documents never leave the machine, and the whole index builds in ~20 s instead of 12+ minutes. Gemini stays available as an option. |
| 3. Making the switch safe | Vectors from different models cannot be compared. | The saved index carries a **signature** (provider, model, chunking, prefix) and refuses to load if it does not match the config. The local model is pinned to an exact revision and loaded from `safetensors`; PyTorch and sentence-transformers were installed from PyPI wheels with pinned versions and checked with `pip-audit`. |
| 4. Data bug | The documentation file `FUENTES.md` was inside `documentos/` and was being indexed as if it were a document. | Moved it out of the folder. |
| 5. First evaluation | 34 questions (28 answerable, 6 not). The dense-only baseline reached 71 % hit@5 and MRR 0.53. Reading the failures showed three weaknesses: exact words and names were missed, questions naming an insurer ("Caser") could not match fragments that never repeat its name, and similarity scores of right and wrong fragments were almost the same (about 0.85 to 0.89), so a similarity threshold cannot tell "no answer". | **Hybrid search** (embeddings + BM25 merged with Reciprocal Rank Fusion) and a **document-name prefix** on every chunk: 86 % hit@5 and MRR 0.67. Chunk size (450 to 1300) showed no clear effect, and splitting by legal article only helped some metrics, so it stays optional. |
| 6. Tokenizer bug | A test caught the BM25 stop-word filter dropping "está" (verb) because it matched "esta" (pronoun) once accents were removed. | Fixed the stop-word list and added a test. |
| 7. The agent | An LLM can invent things, so the answer is not trusted on its own. The model has to quote the documents literally and a plain-code check verifies the quote; otherwise it retries once and then refuses. Built as a LangGraph graph. | Agent with verified citations. The generation model was pinned: the `-latest` alias used in an earlier project stopped working, and `gemini-2.5-flash-lite` is no longer offered to new users, so `gemini-3.5-flash-lite` is set explicitly to keep evaluations comparable. |
| 8. Evaluating the agent | 26/28 answerable questions answered, 6/6 unanswerable refused. A manual review of every answer found 26/26 consistent with the documents. | The review also exposed a wrong expected answer in the evaluation set itself (q18), which was fixed, and four cases where the strict automatic metric marks a valid citation as wrong. The metric was not changed afterwards, to avoid tuning it to the results. |

| 9. MCP server | The search and the agent were only usable from Python. | An MCP server (official SDK, pinned version, audited with `pip-audit`) with three read-only tools. Testing it with a real stdio client showed that an empty question was logged as an unexpected crash with a huge traceback and the client only got a generic message; invalid input now raises the SDK's `ToolError` and the client gets a clear message. |

| 10. HTTP API | MCP serves AI clients; ordinary programs need plain HTTP. | A FastAPI service with four endpoints that reuses the same index and agent (injected, like the MCP server). Tested with FastAPI's test client and then against the running server with `curl`. |

### What is still open

- The evaluation questions were written by the author and the search was tuned while looking at them, so the figures are optimistic. A second, untouched question set written by someone else is needed.
- Two answerable questions are wrongly refused because the search missed the fragment (Caser's definition of jewellery, the definition of deductible). A cross-encoder re-ranker is the next thing to try.
- The HTTP API has no authentication or rate limiting and only listens on localhost.
- The MCP server runs only over stdio on the local machine; there is no authentication or remote transport.

## Another domain

1. Replace the files in `documentos/` and edit `dominio` and `descripcion` in `config.yaml`.
2. Rebuild the index with `python -m src.buscar --reconstruir`.
3. Write new evaluation questions in `evaluacion/preguntas.json` (each needs a `fuente` and an `evidencia` phrase that exists in that document). The current ones are specific to the insurance documents.

## Tests

```bash
python -m pytest
```

50 tests. They use a fake embedder, a fake LLM, an in-memory MCP client and FastAPI's test client, so they need neither an API key nor a model download.
