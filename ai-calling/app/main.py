from fastapi import FastAPI

from app.api.rtp_gateway import router as rtp_router
from app.api.websocket import router as websocket_router

app = FastAPI(title="Realtime Voice Gateway")
app.include_router(websocket_router)
app.include_router(rtp_router)


@app.get("/")
async def health() -> dict[str, str]:
    return {"status": "ok", "service": "voice-gateway"}
