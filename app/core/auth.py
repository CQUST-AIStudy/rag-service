from dataclasses import dataclass
from typing import Annotated

import jwt
from fastapi import Depends, Header
from jwt import InvalidTokenError

from app.core.config import Settings, get_settings
from app.core.responses import ApiError

JWT_ALGORITHMS = ["HS256", "HS384", "HS512"]


@dataclass(frozen=True)
class Principal:
    user_id: str
    username: str
    role: str
    raw_token: str | None = None

    @property
    def is_admin(self) -> bool:
        return self.role.upper() == "ADMIN"


def _dev_principal() -> Principal:
    return Principal(user_id="dev", username="dev", role="TEACHER", raw_token=None)


def parse_bearer_token(authorization: str | None) -> str | None:
    if not authorization:
        return None
    prefix = "Bearer "
    if not authorization.startswith(prefix):
        return None
    token = authorization[len(prefix) :].strip()
    return token or None


def current_principal(
    authorization: Annotated[str | None, Header(alias="Authorization")] = None,
    settings: Settings = Depends(get_settings),
) -> Principal:
    token = parse_bearer_token(authorization)

    if not settings.jwt_secret:
        if settings.is_production:
            raise ApiError(503, "RAG_JWT_SECRET is required in production")
        return _dev_principal()

    if not token:
        raise ApiError(401, "Missing Bearer token")

    try:
        payload = jwt.decode(
            token,
            settings.jwt_secret,
            algorithms=JWT_ALGORITHMS,
            issuer=settings.jwt_issuer,
        )
    except InvalidTokenError as exc:
        raise ApiError(401, "Invalid Bearer token") from exc

    user_id = str(payload.get("uid") or payload.get("sub") or "")
    if not user_id:
        raise ApiError(401, "Token does not contain uid")

    return Principal(
        user_id=user_id,
        username=str(payload.get("sub") or ""),
        role=str(payload.get("role") or "STUDENT").upper(),
        raw_token=token,
    )
