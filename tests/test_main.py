import json

from app.services.rag_chain import RagChainService


async def test_unhandled_error_response_hides_internal_exception_text():
    from app.main import unhandled_error_handler

    response = await unhandled_error_handler(None, RuntimeError("database path /secret/rag.sqlite3 leaked"))
    body = json.loads(response.body)

    assert response.status_code == 500
    assert body["message"] == "服务内部错误，请稍后重试"
    assert "secret" not in body["message"]
    assert "database path" not in body["message"]


async def test_stream_chat_hides_internal_exception_text(monkeypatch, tmp_path):
    from app.core.config import Settings
    from app.services.database import Database
    from app.services.repository import RagRepository

    settings = Settings(data_dir=tmp_path, _env_file=None)
    db = Database(settings)
    db.initialize()
    service = RagChainService(settings, RagRepository(db), None, None, None)
    monkeypatch.setattr(
        service,
        "_prepare_chat",
        lambda request, principal: (_ for _ in ()).throw(RuntimeError("secret")),
    )

    from app.core.auth import Principal
    from app.schemas.rag import ChatRequest

    events = [
        event
        async for event in service.stream_chat(
            ChatRequest(query="hello"),
            Principal(user_id="1", username="student", role="STUDENT"),
        )
    ]

    assert len(events) == 1
    assert "RAG 生成失败，请稍后重试" in events[0]
    assert "secret" not in events[0]
