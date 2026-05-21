"""
Split documents into overlapping chunks for embedding.
Uses LangChain's RecursiveCharacterTextSplitter — best for general text.
"""
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.rag.ingestion.loader import RawDocument
from app.core.config import get_settings
from app.core.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


dataclass_workaround = None  # just importing for use below


def chunk_documents(
    docs: list[RawDocument],
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[dict]:
    """
    Returns list of dicts: { "text": str, "metadata": dict }
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size or settings.chunk_size,
        chunk_overlap=chunk_overlap or settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = []
    for doc in docs:
        pieces = splitter.split_text(doc.content)
        for i, piece in enumerate(pieces):
            chunks.append({
                "text": piece,
                "metadata": {**doc.metadata, "chunk_index": i},
            })

    logger.info("chunks_created",
                input_docs=len(docs),
                output_chunks=len(chunks))
    return chunks