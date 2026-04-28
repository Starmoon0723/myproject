from dataclasses import dataclass


@dataclass(frozen=True)
class ApiError(Exception):
    """
    业务错误（用于返回 vibe.md 约定的 code + http_status）
    """

    code: int
    message: str
    http_status: int


def map_http_exception_to_api_error(status_code: int, detail: str) -> ApiError:
    """
    将历史遗留的 HTTPException（仅含 status + detail）映射成 vibe.md 的业务错误码。
    仅用于兜底：新代码应直接抛 ApiError。
    """

    if status_code == 404:
        return ApiError(code=1404, message=detail, http_status=404)
    if status_code == 413:
        return ApiError(code=1003, message=detail, http_status=413)
    if status_code == 415:
        return ApiError(code=1004, message=detail, http_status=415)
    if status_code in (401, 403):
        # 401/403 视为鉴权/权限问题
        return ApiError(code=1101 if status_code == 401 else 1103, message=detail, http_status=status_code)
    if status_code == 429:
        return ApiError(code=1201, message=detail, http_status=429)
    if status_code == 400:
        return ApiError(code=1001, message=detail, http_status=400)
    # 其它作为内部错误
    return ApiError(code=1301, message=detail, http_status=500)

