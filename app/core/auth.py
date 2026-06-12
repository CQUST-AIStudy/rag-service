from dataclasses import dataclass
from typing import Annotated

import jwt
from fastapi import Depends, Header
from jwt import InvalidTokenError

from app.core.config import Settings, get_settings
from app.core.responses import ApiError

JWT_ALGORITHMS = ["HS256", "HS384", "HS512"]
PLACEHOLDER_JWT_SECRET = "replace-with-a-long-random-secret-at-least-32-chars"
JAVA_DEV_JWT_SECRET = "dev_only_jwt_secret_not_for_production_use_1234567890"


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


def jwt_secret_candidates(settings: Settings) -> list[str]:
    candidates = []
    if settings.jwt_secret:
        candidates.append(settings.jwt_secret)
    candidates.extend(settings.accepted_jwt_secrets)
    if (
        not settings.is_production
        and settings.jwt_secret == PLACEHOLDER_JWT_SECRET
    ):
        candidates.append(JAVA_DEV_JWT_SECRET)
    return list(dict.fromkeys(item for item in candidates if item))


def decode_jwt_with_candidates(token: str, settings: Settings) -> dict:
    last_error: InvalidTokenError | None = None
    for secret in jwt_secret_candidates(settings):
        try:
            return jwt.decode(
                token,
                secret,
                algorithms=JWT_ALGORITHMS,
                issuer=settings.jwt_issuer,
            )
        except InvalidTokenError as exc:
            last_error = exc
    raise ApiError(401, "Invalid Bearer token") from last_error


def current_principal(
    authorization: Annotated[str | None, Header(alias="Authorization")] = None,
    settings: Settings = Depends(get_settings),
) -> Principal:
    token = parse_bearer_token(authorization)
    candidates = jwt_secret_candidates(settings)

    if settings.is_production and not settings.jwt_secret:
        raise ApiError(503, "RAG_JWT_SECRET is required in production")

    if not candidates:
        return _dev_principal()

    if not token:
        raise ApiError(401, "Missing Bearer token")

    payload = decode_jwt_with_candidates(token, settings)

    user_id = str(payload.get("uid") or payload.get("sub") or "")
    if not user_id:
        raise ApiError(401, "Token does not contain uid")

    return Principal(
        user_id=user_id,
        username=str(payload.get("sub") or ""),
        role=str(payload.get("role") or "STUDENT").upper(),
        raw_token=token,
    )
