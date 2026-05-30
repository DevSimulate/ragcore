# RAGCore

A production-grade Retrieval-Augmented Generation (RAG) API built with FastAPI, LangChain, ChromaDB, and PostgreSQL. Supports multi-tenant document ingestion, semantic search, and conversational Q&A over uploaded documents.

## Stack

| Layer | Technology |
|-------|-----------|
| API | FastAPI 0.115 + Uvicorn |
| LLM orchestration | langchain-core / langchain-community / langchain-openai |
| Vector store | ChromaDB (collection-per-tenant) |
| Embeddings | OpenAI text-embedding-3-small |
| Cache | Redis (semantic embedding cache) |
| Database | PostgreSQL + SQLAlchemy async |
| Migrations | Alembic |

## Quick Start

```bash
cp .env.example .env
# add your OPENAI_API_KEY

docker-compose up -d postgres redis chromadb
pip install -r requirements.txt
alembic upgrade head
uvicorn app.main:app --reload
```

API docs: http://localhost:8000/docs

## API Endpoints

### Documents
| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/v1/documents/ingest` | Upload and ingest a PDF/text file |
| `GET` | `/api/v1/documents` | List all documents for the tenant |
| `DELETE` | `/api/v1/documents/{id}` | Delete document and its vectors |

### Queries
| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/v1/queries/ask` | Ask a question (returns full answer + sources) |
| `POST` | `/api/v1/queries/stream` | Stream answer tokens via SSE |

### Collections
| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/v1/collections` | List ChromaDB collections for tenant |
| `DELETE` | `/api/v1/collections/{name}` | Drop a collection |

## Authentication

All requests require an `X-API-Key` header. In production this maps to a tenant. For local dev any string ≥ 8 characters works; the first 8 chars become the `tenant_id`.

```bash
curl -H "X-API-Key: devkey01" http://localhost:8000/api/v1/documents
```

## Project Structure

```
app/
  api/           # Route handlers (documents, queries, collections)
  core/          # Config, database, cache
  models/        # SQLAlchemy ORM models
  schemas/       # Pydantic request/response schemas
  services/      # Business logic
    document_processor.py   # PDF parsing + chunking
    embedding_service.py    # OpenAI embeddings + Redis cache
    vector_store.py         # ChromaDB operations
    rag_chain.py            # LangChain ConversationalRetrievalChain
    llm_service.py          # OpenAI chat completions + token counting
alembic/         # Database migrations
tests/           # Pytest test suite
```

## Running Tests

```bash
pytest tests/ -v
```

## Environment Variables

See `.env.example` for all supported variables.
