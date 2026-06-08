import httpx
import pytest

from app.core.config import Settings
from app.core.responses import ApiError
from app.services import dashscope
from app.services.dashscope import DashScopeEmbeddings


def test_embed_documents_batches_requests_and_keeps_order(monkeypatch, tmp_path):
    calls: list[list[str]] = []
    payloads: list[dict] = []

    class FakeClient:
        def __init__(self, timeout):
            assert timeout == 60

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return None

        def post(self, url, headers, json):
            assert url.endswith("/embeddings")
            assert headers["Authorization"] == "Bearer test-key"
            payloads.append(json)
            texts = list(json["input"])
            calls.append(texts)
            return httpx.Response(
                200,
                json={
                    "data": [
                        {"index": index, "embedding": [float(text.rsplit("-", 1)[1])]}
                        for index, text in enumerate(texts)
                    ]
                },
            )

    monkeypatch.setattr(dashscope.httpx, "Client", FakeClient)
    settings = Settings(data_dir=tmp_path, dashscope_api_key="test-key", _env_file=None)
    embeddings = DashScopeEmbeddings(settings)

    vectors = embeddings.embed_documents([f"text-{index}" for index in range(25)])

    assert [len(call) for call in calls] == [10, 10, 5]
    assert vectors == [[float(index)] for index in range(25)]
    assert all(payload["dimensions"] == settings.embedding_dimensions for payload in payloads)


def test_embed_documents_includes_dashscope_error_detail(monkeypatch, tmp_path):
    class FakeClient:
        def __init__(self, timeout):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return None

        def post(self, url, headers, json):
            return httpx.Response(
                400,
                json={
                    "code": "InvalidParameter",
                    "message": "input length must be less than or equal to 10",
                    "request_id": "req-1",
                },
            )

    monkeypatch.setattr(dashscope.httpx, "Client", FakeClient)
    settings = Settings(data_dir=tmp_path, dashscope_api_key="test-key", _env_file=None)
    embeddings = DashScopeEmbeddings(settings)

    with pytest.raises(ApiError) as exc_info:
        embeddings.embed_documents(["text"])

    message = exc_info.value.message
    assert "HTTP 400" in message
    assert "InvalidParameter" in message
    assert "input length must be less than or equal to 10" in message
    assert "req-1" in message
