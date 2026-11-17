# agente-documentos

Question-answering over a set of documents (RAG), built to be reusable: the code knows nothing about the domain. Everything that changes between domains lives in `config.yaml` and the `documentos/` folder. The first demo domain is **insurance** (policy terms and the Spanish insurance contract law).

> Work in progress. Done: document loading, chunking, embeddings and vector search. Next: baseline evaluation, then an agent (LangGraph), an MCP server and a FastAPI service.

## How it works

1. **Load** PDFs / `.txt` / `.md` and keep the file name and page of every text, so answers can cite them.
2. **Chunk** each page into ~900-character pieces with 150 characters of overlap.
3. **Embed** each chunk into a vector (Gemini API, or a local `sentence-transformers` model) and normalise it, so the dot product is the cosine similarity.
4. **Search**: embed the question and multiply it by the matrix of chunk vectors; return the top *k* chunks.

The index is saved in `indice/` with a signature (provider, model, chunking). If the configuration changes, loading it fails on purpose instead of mixing vectors from different models.

## Use

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
echo "GEMINI_API_KEY=your_key" > .env
# put the documents in documentos/ (see documentos/FUENTES.md)
python -m src.buscar --reconstruir
python -m src.buscar "¿Está cubierto el robo de joyas en casa?"
```

The Gemini free tier allows about 100 embeddings per minute, so the first build of ~1,000 chunks takes around 12 minutes. After that the index is cached.

To embed locally instead (documents never leave the machine): `pip install sentence-transformers` and set `proveedor_embeddings: local` in `config.yaml`.

## Another domain

Replace the files in `documentos/`, edit `dominio` and `descripcion` in `config.yaml`, rebuild the index.

## Tests

```bash
python -m pytest
```

The tests use a fake embedder, so they need no API key.
