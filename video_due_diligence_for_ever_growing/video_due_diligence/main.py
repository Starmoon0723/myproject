from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi import HTTPException
from .routes.api import api_router
from .utils.result_cache import result_cache
from .utils.errors import ApiError, map_http_exception_to_api_error
from .models.model_analyzer import ModelAnalyzer

app = FastAPI(
    title="Video Due Diligence API",
    description="API for video due diligence and ASR processing",
    version="1.0.0"
)


# 全局结果缓存（单例）
# 注意：result_cache 在 utils/result_cache.py 中初始化，避免多实例/多清理线程


# 注册路由
app.include_router(api_router, prefix="/api/v1")


@app.on_event("startup")
async def init_global_analyzers():
    """
    初始化全局单例：
    - app.state.model_analyzer: 供 ASR/视频尽调后台任务调用
    """
    # ModelAnalyzer 内部会根据 config.yaml 是否完整自动进入 mock 模式，避免启动失败
    app.state.model_analyzer = ModelAnalyzer()


@app.exception_handler(ApiError)
async def api_error_handler(request: Request, exc: ApiError):
    return JSONResponse(
        status_code=exc.http_status,
        content={"code": exc.code, "message": exc.message},
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    mapped = map_http_exception_to_api_error(exc.status_code, str(exc.detail))
    return JSONResponse(
        status_code=mapped.http_status,
        content={"code": mapped.code, "message": mapped.message},
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    # Pydantic/JSON 校验错误统一按 1001 返回
    return JSONResponse(
        status_code=400,
        # 带上具体字段错误，方便联调排查；不影响 code/message 的对齐
        content={
            "code": 1001,
            "message": "参数缺失/非法",
            "errors": exc.errors(),
        },
    )


@app.get("/")
async def root():
    return {"message": "Video Due Diligence API is running"}


@app.get("/health")
async def health_check():
    return {"status": "healthy"}