from typing import Any

from fastapi import HTTPException
from fastapi.responses import JSONResponse


class ApiError(HTTPException):
    def __init__(self, status_code: int, message: str, code: int | None = None):
        super().__init__(status_code=status_code, detail=message)
        self.code = code or status_code
        self.message = message


def api_success(data: Any = None, message: str = "success") -> dict[str, Any]:
    return {"code": 200, "message": message, "data": data}


def api_error_response(status_code: int, message: str, code: int | None = None) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"code": code or status_code, "message": message, "data": None},
    )
