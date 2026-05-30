from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import documents, queries, collections
from app.core.config import settings
from app.core.database import engine, Base
from app.core.cache import redis_client


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    await redis_client.aclose()


app = FastAPI(
    title="RAGCore API",
    description="Production RAG pipeline for enterprise document Q&A",
    version="1.4.2",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(documents.router, prefix="/documents", tags=["documents"])
app.include_router(queries.router, prefix="/queries", tags=["queries"])
app.include_router(collections.router, prefix="/collections", tags=["collections"])


@app.get("/health")
async def health():
    return {"status": "ok", "version": "1.4.2"}
