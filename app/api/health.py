from fastapi import APIRouter

from app.core.responses import api_success

router = APIRouter(tags=["health"])


@router.get("/health")
def health() -> dict:
    return api_success({"status": "ok"})
