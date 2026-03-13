from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))


services_module = sys.modules.setdefault("services", types.ModuleType("services"))
services_module.__path__ = [str(APP_DIR)]
sys.modules.setdefault("services.api_client", importlib.import_module("api_client"))
sys.modules.setdefault("services.crew_tasks", importlib.import_module("crew_tasks"))


from config.settings import settings  # noqa: E402
from services.crew_tasks import SupportCrew  # noqa: E402


class AskRequest(BaseModel):
    query: str


app = FastAPI(title="AI Brain Service")


@app.get("/")
async def root() -> dict[str, str]:
    return {"status": "AI brain running"}


@app.post("/ask")
async def ask(payload: AskRequest) -> dict[str, str]:
    user_data = {
        "name": "Demo User",
        "email": "demo@example.com",
        "mobile": "9999999999",
        "department_details": {
            "name": "taxation",
            "description": "Handles taxation queries",
        },
        "token": "demo_token_123",
    }

    try:
        crew = SupportCrew(user_data)
        result = await crew.handle_query(payload.query)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {"response": str(result)}
