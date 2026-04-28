import uvicorn
from video_due_diligence.main import app
from video_due_diligence.core.config import config

if __name__ == "__main__":
    host = config.get("server.host", "0.0.0.0")
    port = config.get("server.port", 8000)
    reload = config.get("server.reload", True)
    
    uvicorn.run(
        "video_due_diligence.main:app",
        host=host,
        port=port,
        reload=reload
    )