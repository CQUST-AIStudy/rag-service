import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware

from app.api import chat, compat, document, health, knowledge_base
from app.core.config import get_settings
from app.core.responses import ApiError, api_error_response
from app.services.dependencies import get_database

settings = get_settings()
logger = logging.getLogger(__name__)

app = FastAPI(title=settings.service_name)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup() -> None:
    get_database().initialize()


@app.exception_handler(ApiError)
async def api_error_handler(_request: Request, exc: ApiError):
    return api_error_response(exc.status_code, exc.message, exc.code)


@app.exception_handler(RequestValidationError)
async def validation_error_handler(_request: Request, exc: RequestValidationError):
    return api_error_response(422, str(exc), 422)


@app.exception_handler(Exception)
async def unhandled_error_handler(_request: Request, exc: Exception):
    logger.error("Unhandled RAG service error", exc_info=(type(exc), exc, exc.__traceback__))
    return api_error_response(500, "服务内部错误，请稍后重试", 500)


app.include_router(health.router)
app.include_router(knowledge_base.router)
app.include_router(document.router)
app.include_router(document.compat_router)
app.include_router(chat.router)
app.include_router(compat.router)
