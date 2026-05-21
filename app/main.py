from fastapi import FastAPI
from app.core.config import get_settings
from app.core.logger import setup_logging
from app.api.v1 import health, ingest, rag,admin

settings = get_settings()
setup_logging()

app = FastAPI(
    title=settings.app_name,
    version="0.1.0 — Phase 1",
    docs_url="/docs",
)

PREFIX = "/api/v1"
app.include_router(health.router,  prefix=PREFIX, tags=["Health"])
app.include_router(ingest.router,  prefix=PREFIX, tags=["Ingest"])
app.include_router(rag.router,     prefix=PREFIX, tags=["RAG"])
app.include_router(admin.router,   prefix=PREFIX, tags=["Admin"])