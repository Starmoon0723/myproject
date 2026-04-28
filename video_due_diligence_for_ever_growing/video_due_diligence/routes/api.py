from fastapi import APIRouter
from ..api import asr, video_guidance

# 创建API路由路由器
api_router = APIRouter()

# 包含所有API路由
api_router.include_router(asr.router, prefix="", tags=["asr"])
api_router.include_router(video_guidance.router, prefix="", tags=["video-guidance"])

__all__ = ["api_router"]