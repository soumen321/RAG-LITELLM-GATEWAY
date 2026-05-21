import pytest
from unittest.mock import patch, MagicMock
from app.rag.ingestion.chunker import chunk_documents
from app.rag.ingestion.loader import load_text, RawDocument


def test_load_text():
    docs = load_text("Hello world. This is a test.", {"source": "test"})
    assert len(docs) == 1
    assert docs[0].content == "Hello world. This is a test."
    assert docs[0].metadata["source"] == "test"


def test_chunk_documents():
    docs = [RawDocument(
        content="word " * 300,   # long enough to split
        metadata={"source": "test"}
    )]
    chunks = chunk_documents(docs, chunk_size=100, chunk_overlap=10)
    assert len(chunks) > 1
    for c in chunks:
        assert "text" in c
        assert "metadata" in c
        assert "chunk_index" in c["metadata"]


def test_chunk_preserves_metadata():
    docs = [RawDocument(
        content="Some text here " * 50,
        metadata={"source": "myfile.pdf", "page": 1}
    )]
    chunks = chunk_documents(docs)
    assert all(c["metadata"]["source"] == "myfile.pdf" for c in chunks)
    assert all(c["metadata"]["page"] == 1 for c in chunks)


@pytest.mark.asyncio
async def test_health_endpoint():
    from httpx import AsyncClient, ASGITransport
    from app.main import app
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as c:
        r = await c.get("/api/v1/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"