from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.responses import StreamingResponse
from openai import AsyncOpenAI


APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))


services_module = sys.modules.setdefault("services", types.ModuleType("services"))
services_module.__path__ = [str(APP_DIR)]
sys.modules.setdefault("services.api_client", importlib.import_module("api_client"))
sys.modules.setdefault("services.crew_tasks", importlib.import_module("crew_tasks"))


from config.settings import settings  # noqa: E402
from services.crew_tasks import SupportCrew  # noqa: E402

SYSTEM_PROMPT = (
    "You are a helpful Indian tax support assistant for FSK India. "
    "Keep responses concise and conversational — 1 to 3 short sentences maximum. "
    "LANGUAGE RULES — follow strictly:\n"
    "1. If the user speaks in Hindi or any Indian language, reply ONLY in Hindi using "
    "proper Devanagari script. Never use Romanized Hindi (no 'Aap', 'kaise', 'hain', etc.).\n"
    "2. If the user speaks in English, reply in clear simple English.\n"
    "3. Never mix scripts in a single response.\n"
    "HINDI WRITING RULES (critical for voice output quality):\n"
    "- Write all numbers as Hindi words: १ → एक, १०० → सौ, १००० → हज़ार\n"
    "- Write acronyms as full Hindi words, not letters: "
    "ITR → आयकर रिटर्न, GST → वस्तु एवं सेवा कर, PAN → पैन कार्ड\n"
    "- Never write English acronyms or abbreviations in a Hindi response.\n"
    "- Keep sentences short — maximum 20 words per sentence."
)


class AskRequest(BaseModel):
    query: str


app = FastAPI(title="AI Brain Service")
openai_client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)


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


async def stream_llm_response(query: str):
    stream = await openai_client.chat.completions.create(
        model="gpt-4o-mini",
        stream=True,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ],
    )

    async for chunk in stream:
        delta = chunk.choices[0].delta.content if chunk.choices else None
        if delta:
            yield delta


@app.post("/voice")
async def voice(payload: AskRequest) -> StreamingResponse:
    return StreamingResponse(
        stream_llm_response(payload.query),
        media_type="text/plain",
    )
