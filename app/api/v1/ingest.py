from fastapi import APIRouter, Depends, UploadFile, File
from app.models.request import IngestTextRequest, IngestUrlRequest
from app.models.response import IngestResponse
from app.api.deps import verify_key
from app.rag.pipeline import get_pipeline
import tempfile, shutil, os

router = APIRouter()


@router.post("/ingest/text", response_model=IngestResponse)
async def ingest_text(
    req: IngestTextRequest,
    _: str = Depends(verify_key),
):
    result = get_pipeline().ingest_text(
        req.text, req.metadata, req.chunk_size
    )
    return IngestResponse(**result)


@router.post("/ingest/url", response_model=IngestResponse)
async def ingest_url(
    req: IngestUrlRequest,
    _: str = Depends(verify_key),
):
    result = get_pipeline().ingest_url(req.url, req.chunk_size)
    return IngestResponse(**result)


@router.post("/ingest/pdf", response_model=IngestResponse)
async def ingest_pdf(
    file: UploadFile = File(...),
    _: str = Depends(verify_key),
):
    # Save upload to a temp file then ingest
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        result = get_pipeline().ingest_pdf(tmp_path)
    finally:
        os.unlink(tmp_path)

    return IngestResponse(**result)