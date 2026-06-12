from datetime import UTC, datetime, timedelta

import jwt
import pytest

from app.core.auth import (
    JAVA_DEV_JWT_SECRET,
    PLACEHOLDER_JWT_SECRET,
    current_principal,
)
from app.core.config import Settings
from app.core.responses import ApiError


def make_token(secret: str, algorithm: str = "HS256") -> str:
    return jwt.encode(
        {
            "iss": "tap",
            "sub": "teacher",
            "uid": 12,
            "role": "TEACHER",
            "exp": datetime.now(UTC) + timedelta(minutes=5),
        },
        secret,
        algorithm=algorithm,
    )


def test_dev_auth_allows_missing_secret():
    principal = current_principal(None, Settings(env="local", jwt_secret="", _env_file=None))
    assert principal.user_id == "dev"
    assert principal.role == "TEACHER"


def test_jwt_auth_parses_tap_token():
    settings = Settings(jwt_secret="x" * 32, jwt_issuer="tap", _env_file=None)
    token = make_token(settings.jwt_secret)

    principal = current_principal(f"Bearer {token}", settings)

    assert principal.user_id == "12"
    assert principal.username == "teacher"
    assert principal.role == "TEACHER"


@pytest.mark.parametrize(
    ("secret", "algorithm"),
    [
        ("x" * 32, "HS256"),
        ("x" * 48, "HS384"),
        ("x" * 64, "HS512"),
    ],
)
def test_jwt_auth_accepts_hmac_algorithms_used_by_jjwt(secret: str, algorithm: str):
    settings = Settings(jwt_secret=secret, jwt_issuer="tap", _env_file=None)
    token = make_token(settings.jwt_secret, algorithm)

    principal = current_principal(f"Bearer {token}", settings)

    assert principal.user_id == "12"
    assert principal.role == "TEACHER"


def test_jwt_auth_accepts_configured_additional_secret():
    settings = Settings(
        jwt_secret="primary-secret-value-with-enough-length",
        jwt_accepted_secrets="old-secret-value-with-enough-length",
        jwt_issuer="tap",
        _env_file=None,
    )
    token = make_token("old-secret-value-with-enough-length")

    principal = current_principal(f"Bearer {token}", settings)

    assert principal.user_id == "12"
    assert principal.role == "TEACHER"


def test_jwt_auth_accepts_any_configured_additional_secret():
    settings = Settings(
        jwt_secret="primary-secret-value-with-enough-length",
        jwt_accepted_secrets=(
            "first-old-secret-value-with-enough-length,"
            "second-old-secret-value-with-enough-length"
        ),
        jwt_issuer="tap",
        _env_file=None,
    )
    token = make_token("second-old-secret-value-with-enough-length")

    principal = current_principal(f"Bearer {token}", settings)

    assert principal.user_id == "12"
    assert principal.role == "TEACHER"


def test_jwt_auth_allows_java_dev_secret_when_local_secret_is_placeholder():
    settings = Settings(
        env="local",
        jwt_secret=PLACEHOLDER_JWT_SECRET,
        jwt_issuer="tap",
        _env_file=None,
    )
    token = make_token(JAVA_DEV_JWT_SECRET)

    principal = current_principal(f"Bearer {token}", settings)

    assert principal.user_id == "12"
    assert principal.role == "TEACHER"


def test_jwt_auth_does_not_allow_java_dev_secret_in_production():
    settings = Settings(
        env="production",
        jwt_secret=PLACEHOLDER_JWT_SECRET,
        jwt_issuer="tap",
        _env_file=None,
    )
    token = make_token(JAVA_DEV_JWT_SECRET)

    with pytest.raises(ApiError) as exc_info:
        current_principal(f"Bearer {token}", settings)

    assert exc_info.value.status_code == 401


def test_jwt_auth_rejects_missing_token_when_secret_configured():
    settings = Settings(jwt_secret="x" * 32, jwt_issuer="tap", _env_file=None)
    try:
        current_principal(None, settings)
    except ApiError as exc:
        assert exc.status_code == 401
    else:
        raise AssertionError("missing token should be rejected")


def test_jwt_auth_requires_primary_secret_in_production():
    settings = Settings(
        env="production",
        jwt_secret="",
        jwt_accepted_secrets="old-secret-value-with-enough-length",
        jwt_issuer="tap",
        _env_file=None,
    )
    token = make_token("old-secret-value-with-enough-length")

    with pytest.raises(ApiError) as exc_info:
        current_principal(f"Bearer {token}", settings)

    assert exc_info.value.status_code == 503
