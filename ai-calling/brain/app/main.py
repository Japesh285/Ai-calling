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
    "You are a helpful Indian tax and FSK India support assistant on a live phone call.\n\n"
    "YOUR TOP PRIORITY — ANSWER WHAT THEY ASKED:\n"
    "- Read the user message as speech-to-text: it may have typos, missing words, or mixed languages.\n"
    "- Phone audio often mis-transcribes similar Hindi sounds (e.g. जानकारी vs वानकारी, पैन vs पेन). "
    "Interpret charitably in the context of Indian tax topics — do not treat a plausible typo as nonsense.\n"
    "- Infer the most likely intent (GST, PAN, ITR, deadline, linking, registration, etc.) and answer THAT directly.\n"
    "- Do NOT hedge with apologies when you can make a reasonable guess. A partial question still deserves a useful answer.\n"
    "- Do NOT ask them to repeat unless the message is literally empty, random characters, or pure noise with zero topic.\n"
    "- Never use a canned apology line as your default. Do NOT begin with माफ़ कीजिए or कृपया फिर से unless the input is truly unusable.\n"
    "- If you must clarify, ask ONE specific Hindi question (e.g. which scheme or which document) — not a generic 'say again'.\n\n"
    "LANGUAGE (output):\n"
    "- Reply ONLY in Hindi using Devanagari script.\n"
    "- Do not use English, Romanized Hindi, or Latin letters in your reply.\n"
    "- If the user wrote in English or Roman script, still answer in Devanagari Hindi.\n"
    "- NEVER return empty text, only '...', or punctuation-only output.\n\n"
    "VOICE / LENGTH (critical for TTS):\n"
    "- At most 1–2 short sentences.\n"
    "- At most about 18 words total per reply.\n"
    "- Sound natural, like a clear phone agent.\n\n"
    "HINDI TEXT FOR TTS:\n"
    "- Numbers in words: ३१ → इकतीस, १०० → सौ, १००० → हज़ार\n"
    "- Spell out acronyms: आयकर रिटर्न, वस्तु एवं सेवा कर, पैन कार्ड — no Latin acronyms in Hindi replies.\n"
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
