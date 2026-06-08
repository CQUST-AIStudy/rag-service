from datetime import UTC, datetime, timedelta

import jwt

from app.core.auth import current_principal
from app.core.config import Settings
from app.core.responses import ApiError


def test_dev_auth_allows_missing_secret():
    principal = current_principal(None, Settings(env="local", jwt_secret="", _env_file=None))
    assert principal.user_id == "dev"
    assert principal.role == "TEACHER"


def test_jwt_auth_parses_tap_token():
    settings = Settings(jwt_secret="x" * 32, jwt_issuer="tap", _env_file=None)
    token = jwt.encode(
        {
            "iss": "tap",
            "sub": "teacher",
            "uid": 12,
            "role": "TEACHER",
            "exp": datetime.now(UTC) + timedelta(minutes=5),
        },
        settings.jwt_secret,
        algorithm="HS256",
    )

    principal = current_principal(f"Bearer {token}", settings)

    assert principal.user_id == "12"
    assert principal.username == "teacher"
    assert principal.role == "TEACHER"


def test_jwt_auth_rejects_missing_token_when_secret_configured():
    settings = Settings(jwt_secret="x" * 32, jwt_issuer="tap", _env_file=None)
    try:
        current_principal(None, settings)
    except ApiError as exc:
        assert exc.status_code == 401
    else:
        raise AssertionError("missing token should be rejected")
