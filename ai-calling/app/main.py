from fastapi import FastAPI

from app.api.websocket import router as websocket_router

app = FastAPI(title="Realtime Voice Gateway")
app.include_router(websocket_router)


@app.get("/")
async def health() -> dict[str, str]:
    return {"status": "ok", "service": "voice-gateway"}
