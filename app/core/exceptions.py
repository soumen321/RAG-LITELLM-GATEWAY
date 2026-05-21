from fastapi import HTTPException, status


class NotFoundError(HTTPException):
    def __init__(self, detail: str = "Not found"):
        super().__init__(status_code=status.HTTP_404_NOT_FOUND, detail=detail)


class IngestionError(HTTPException):
    def __init__(self, detail: str):
        super().__init__(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Ingestion failed: {detail}",
        )


class GenerationError(HTTPException):
    def __init__(self, detail: str):
        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Generation failed: {detail}",
        )


class AuthError(HTTPException):
    def __init__(self):
        super().__init__(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )
        
# ── NEW in Phase 3 ────────────────────────────────────────────────────────

class AllProvidersFailedError(HTTPException):
    """Raised when every provider in the fallback chain fails."""
    def __init__(self, tried: list[str]):
        super().__init__(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                f"All providers failed: {tried}. "
                f"Please retry later."
            ),
        )


class FallbackTriggeredWarning(Exception):
    """
    Not an HTTP error — raised internally to signal a fallback occurred.
    Caught in generator.py and logged so callers know which model was used.
    """
    def __init__(self, original: str, fallback: str, reason: str):
        self.original = original
        self.fallback = fallback
        self.reason   = reason
        super().__init__(
            f"Fell back from '{original}' to '{fallback}' — reason: {reason}"
        )        