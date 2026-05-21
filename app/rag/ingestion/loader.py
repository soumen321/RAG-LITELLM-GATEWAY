"""
Load documents from three sources:
  - Plain text string
  - PDF file (path)
  - Web URL
"""
from pathlib import Path
from dataclasses import dataclass
from app.core.logger import get_logger

logger = get_logger(__name__)


@dataclass
class RawDocument:
    content: str
    metadata: dict


def load_text(text: str, metadata: dict | None = None) -> list[RawDocument]:
    """Wrap a raw text string as a document."""
    return [RawDocument(content=text, metadata=metadata or {"source": "text"})]


def load_pdf(file_path: str) -> list[RawDocument]:
    """Extract text from each page of a PDF."""
    from pypdf import PdfReader

    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {file_path}")

    reader = PdfReader(str(path))
    docs = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            docs.append(RawDocument(
                content=text,
                metadata={"source": path.name, "page": i + 1},
            ))

    logger.info("pdf_loaded", file=path.name, pages=len(docs))
    return docs


def load_url(url: str) -> list[RawDocument]:
    """Scrape visible text from a web page."""
    import requests
    from bs4 import BeautifulSoup

    response = requests.get(url, timeout=15)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    # Remove script/style noise
    for tag in soup(["script", "style", "nav", "footer"]):
        tag.decompose()

    text = soup.get_text(separator="\n", strip=True)
    logger.info("url_loaded", url=url, chars=len(text))
    return [RawDocument(content=text, metadata={"source": url})]