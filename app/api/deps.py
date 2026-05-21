from fastapi import Header
from app.core.exceptions import AuthError
from app.core.config import get_settings

settings = get_settings()


async def verify_key(x_api_key: str = Header(...)) -> str:
    if x_api_key != settings.gateway_api_key:
        raise AuthError()
    return x_api_key