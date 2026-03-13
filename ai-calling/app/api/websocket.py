import asyncio
import base64
import audioop
import json
from datetime import datetime, timezone
from pathlib import Path
import wave

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.audio.audio_inspector import inspect_packet
from app.audio.vad import RealtimeSpeechGate
from app.config.settings import (
    BASE_DIR,
    FREESWITCH_AUDIO_DUMP_DIR,
    FREESWITCH_RAW_DUMP_DIR,
    PCM_CHANNELS,
    PCM_SAMPLE_RATE,
    PCM_SAMPLE_WIDTH_BYTES,
)
from app.pipeline.speech_pipeline import SpeechPipeline
from app.sessions.call_session import CallSession
from app.utils.logger import get_logger

router = APIRouter()
logger = get_logger(__name__)

PCM16_FRAME_BYTES = 320
DEBUG_DUMP_PACKETS = 10
INSPECT_VERBOSE_PACKETS = 20
DEBUG_RAW_DUMP_PATH = BASE_DIR / "logs" / "audio_debug_dump.pcm"


def _dump_inbound_audio(audio_bytes: bytes) -> Path | None:
    if not audio_bytes:
        return None

    FREESWITCH_AUDIO_DUMP_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    output_path = FREESWITCH_AUDIO_DUMP_DIR / f"call_{timestamp}.wav"

    with wave.open(str(output_path), "wb") as wav_file:
        wav_file.setnchannels(PCM_CHANNELS)
        wav_file.setsampwidth(PCM_SAMPLE_WIDTH_BYTES)
        wav_file.setframerate(PCM_SAMPLE_RATE)
        wav_file.writeframes(audio_bytes)

    return output_path


def _dump_raw_audio(audio_bytes: bytes) -> Path | None:
    if not audio_bytes:
        return None

    FREESWITCH_RAW_DUMP_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    output_path = FREESWITCH_RAW_DUMP_DIR / f"call_{timestamp}.alaw"
    output_path.write_bytes(audio_bytes)
    return output_path


@router.websocket("/audio-stream")
@router.websocket("/audio")
@router.websocket("/ws/audio")
async def audio_stream(ws: WebSocket) -> None:
    await ws.accept()
    session = CallSession()
    pipeline = SpeechPipeline(session)
    speech_gate = RealtimeSpeechGate()
    client = getattr(ws, "client", None)
    client_label = f"{client.host}:{client.port}" if client else "unknown"
    packet_counter = 0
    pcm_buffer = bytearray()

    logger.info(
        "Call websocket accepted: client=%s path=%s",
        client_label,
        ws.url.path,
    )

    try:
        while True:
            message = await ws.receive()
            msg_type = message.get("type")

            if msg_type == "websocket.disconnect":
                break

            audio_chunk = message.get("bytes")
            text_frame = message.get("text")
            if text_frame and not audio_chunk:
                audio_chunk = _extract_audio_from_text_frame(text_frame)

            if not audio_chunk:
                continue

            packet_counter += 1
            diagnostics = inspect_packet(audio_chunk)
            if packet_counter <= DEBUG_DUMP_PACKETS:
                await _append_debug_raw_dump(audio_chunk)
            if packet_counter <= INSPECT_VERBOSE_PACKETS or packet_counter % 100 == 0:
                logger.info(
                    "AUDIO_INSPECT packet_bytes=%s frame_size=%s first_bytes=%s unique_bytes=%s likely=%s pcm16_rms=%s",
                    diagnostics["packet_bytes"],
                    PCM16_FRAME_BYTES,
                    diagnostics["first_bytes"],
                    diagnostics["unique_bytes"],
                    diagnostics["likely"],
                    diagnostics["pcm16_rms"],
                )

            pcm_buffer.extend(audio_chunk)

            if packet_counter <= 5 or packet_counter % 100 == 0:
                logger.warning(
                    "PCM stream buffer: packet_bytes=%s buffered_bytes=%s complete_frames=%s",
                    len(audio_chunk),
                    len(pcm_buffer),
                    len(pcm_buffer) // PCM16_FRAME_BYTES,
                )

            audio_out = None
            while len(pcm_buffer) >= PCM16_FRAME_BYTES:
                frame = bytes(pcm_buffer[:PCM16_FRAME_BYTES])
                del pcm_buffer[:PCM16_FRAME_BYTES]
                pcm16_chunk = frame

                session.add_raw_chunk(frame)
                session.add_inbound_chunk(pcm16_chunk)
                vad_result = speech_gate.process_frame(pcm16_chunk)

                if vad_result.speech_started:
                    logger.info(
                        "Speech start: client=%s frame=%s rms=%s",
                        client_label,
                        session.frame_counter,
                        vad_result.rms,
                    )

                if session.frame_counter <= 5 or session.frame_counter % 50 == 0:
                    logger.info(
                        "Call audio frame checkpoint: frame=%s raw_bytes=%s pcm_bytes=%s rms=%s speech=%s",
                        session.frame_counter,
                        len(frame),
                        len(pcm16_chunk),
                        vad_result.rms,
                        vad_result.is_speech,
                    )

                if vad_result.speech_chunk:
                    frame_audio_out = await pipeline.process_audio_chunk(vad_result.speech_chunk)
                    audio_out = audio_out or frame_audio_out

                if vad_result.speech_stopped:
                    logger.info(
                        "Speech stop: client=%s frame=%s buffered_speech_bytes=%s",
                        client_label,
                        session.frame_counter,
                        len(session.stt_buffer),
                    )
                    frame_audio_out = await pipeline.flush_on_speech_stop()
                    audio_out = audio_out or frame_audio_out

                if audio_out:
                    logger.info("Sending TTS audio back to FreeSWITCH: %s bytes", len(audio_out))
                    await ws.send_bytes(audio_out)
                    audio_out = None

    except WebSocketDisconnect:
        logger.info("Call websocket disconnected: client=%s path=%s", client_label, ws.url.path)
    finally:
        if pcm_buffer:
            logger.info(
                "Discarding trailing partial PCM frame bytes=%s on disconnect",
                len(pcm_buffer),
            )

        final_audio = await pipeline.flush_remaining()
        if final_audio:
            try:
                logger.info("Sending final TTS audio back to FreeSWITCH: %s bytes", len(final_audio))
                await ws.send_bytes(final_audio)
            except Exception:
                logger.info("Could not send final TTS audio after disconnect")

        raw_audio_path = await _save_raw_audio_dump(session)
        if raw_audio_path:
            logger.info("Saved raw FreeSWITCH audio to %s", raw_audio_path)

        inbound_audio_path = await _save_inbound_audio_dump(session)
        if inbound_audio_path:
            logger.info("Saved inbound FreeSWITCH audio to %s", inbound_audio_path)

        decoded_audio = session.get_inbound_audio()
        if decoded_audio:
            total_rms = audioop.rms(decoded_audio, PCM_SAMPLE_WIDTH_BYTES)
            duration_seconds = len(decoded_audio) / (
                PCM_SAMPLE_RATE * PCM_CHANNELS * PCM_SAMPLE_WIDTH_BYTES
            )
            logger.info(
                "Call audio summary: duration=%.2fs decoded_bytes=%s raw_bytes=%s rms=%s",
                duration_seconds,
                len(decoded_audio),
                len(session.get_raw_audio()),
                total_rms,
            )

        logger.info(
            "Call session closed: client=%s total_frames=%s speech_chunks=%s",
            client_label,
            session.frame_counter,
            session.speech_chunk_counter,
        )


def _extract_audio_from_text_frame(text_frame: str) -> bytes | None:
    try:
        payload = json.loads(text_frame)
    except json.JSONDecodeError:
        logger.info("FreeSWITCH text frame: %s", text_frame)
        return None

    if isinstance(payload, dict):
        event_type = payload.get("event") or payload.get("type")
        audio_base64 = payload.get("audio") or payload.get("data")
        if event_type:
            logger.info("FreeSWITCH event: %s", event_type)
        if isinstance(audio_base64, str):
            try:
                return base64.b64decode(audio_base64)
            except Exception:
                logger.warning("Invalid base64 audio payload in text frame")
                return None

    logger.info("FreeSWITCH text payload received")
    return None


async def _save_inbound_audio_dump(session: CallSession) -> Path | None:
    inbound_audio = session.get_inbound_audio()
    if not inbound_audio:
        return None

    return await asyncio.to_thread(_dump_inbound_audio, inbound_audio)


async def _save_raw_audio_dump(session: CallSession) -> Path | None:
    raw_audio = session.get_raw_audio()
    if not raw_audio:
        return None

    return await asyncio.to_thread(_dump_raw_audio, raw_audio)


async def _append_debug_raw_dump(packet: bytes) -> None:
    def _write() -> None:
        DEBUG_RAW_DUMP_PATH.parent.mkdir(parents=True, exist_ok=True)
        with DEBUG_RAW_DUMP_PATH.open("ab") as handle:
            handle.write(packet)

    await asyncio.to_thread(_write)
