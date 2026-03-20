import asyncio
import base64
import audioop
import errno
import fcntl
import io
import json
import os
import tempfile
import threading
import time
from datetime import datetime, timezone
from math import gcd
from pathlib import Path
from typing import Optional
from collections.abc import AsyncIterator
import wave

import httpx
import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from scipy.signal import resample_poly

from app.audio.tts_streamer import (
    generate_tts,
)
from app.config.settings import (
    BARGE_IN_RMS_THRESHOLD,
    FREESWITCH_AUDIO_DUMP_DIR,
    FREESWITCH_RAW_DUMP_DIR,
    INDIC_TTS_URL,
    PCM_CHANNELS,
    PCM_FRAME_BYTES,
    PCMA_FRAME_BYTES,
    PCM_SAMPLE_RATE,
    PCM_SAMPLE_WIDTH_BYTES,
    TTS_PLAYBACK_GUARD_MS,
    TTS_WORKERS,
    USE_FIFO_PLAYBACK,
)
from app.pipeline.speech_pipeline import (
    SpeechSegmenter,
    append_brain_response,
    process_transcript_to_reply_streaming,
    transcribe_audio,
    transcribe_audio_streaming,
)
from app.clients.esl_client import esl_api
from app.rtp.outbound import RtpPcmSink
from app.sessions.call_session import CallSession
from app.utils.logger import get_logger

router = APIRouter()
logger = get_logger(__name__)
PLAYBACK_LOCK = asyncio.Lock()

# When present, AI playback goes out as RTP to the symmetric peer instead of uuid_broadcast.
_rtp_sinks: dict[str, RtpPcmSink] = {}
TTS_WORKER_COUNT = 2  # Parallel TTS workers for reduced latency

# Global monotonically increasing sequence ID - NEVER reset
_global_seq_id = 0

INDIC_STREAM_SAMPLE_RATE = 24000
INDIC_STREAM_SAMPLE_WIDTH_BYTES = 2
INDIC_STREAM_CHANNELS = 1
# Must match indic_server INDIC_STREAM_PACKET_MS (default 20 ms @ 24 kHz → 960 bytes PCM16).
INDIC_STREAM_SEGMENT_SECONDS = float(os.environ.get("INDIC_STREAM_SEGMENT_SECONDS", "0.02"))
INDIC_STREAM_WAV_HEADER_BYTES = 44
INDIC_STREAM_SEGMENT_PCM_BYTES = int(
    INDIC_STREAM_SAMPLE_RATE
    * INDIC_STREAM_SAMPLE_WIDTH_BYTES
    * INDIC_STREAM_CHANNELS
    * INDIC_STREAM_SEGMENT_SECONDS
)


class _CallFifo:
    """
    Per-response named pipe (FIFO) that streams raw 8 kHz PCM16 audio
    directly to FreeSWITCH — eliminating /tmp .wav file creation entirely.

    Lifecycle per AI response
    -------------------------
    fifo = _CallFifo(call_uuid)
    fifo.create()                  # mkfifo /tmp/ai_<uuid>_<gen>.r8
    fifo.broadcast()               # esl uuid_broadcast … .r8  (FS opens read end)
    fifo.write(pcm_bytes)          # first call opens write end, subsequent calls write
    fifo.write(more_pcm_bytes)     # …
    fifo.close_write()             # FS reads EOF → playback stops cleanly
    fifo.cleanup()                 # unlink pipe from filesystem

    On barge-in call close_write() early — FS drains its buffer (~1 frame) and stops.
    """

    def __init__(self, call_uuid: str, generation: int) -> None:
        self.call_uuid  = call_uuid
        self.generation = generation
        self.path       = f"/tmp/ai_{call_uuid}_{generation}.r8"
        self._fd: Optional[int] = None
        self._lock      = threading.Lock()
        self._broadcast_done = False
        self._open_event = threading.Event()   # signals write end is open

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def create(self) -> None:
        """Create the named pipe on the filesystem."""
        if os.path.exists(self.path):
            try:
                os.unlink(self.path)
            except OSError:
                pass
        os.mkfifo(self.path)
        logger.debug("FIFO created: %s", self.path)

    def broadcast(self) -> None:
        """
        Tell FreeSWITCH to open and play the pipe.
        ESL returns immediately; FS opens the read end asynchronously.
        Must be called BEFORE write() so the read end is ready.
        """
        if not self._broadcast_done:
            response = esl_api(f"uuid_broadcast {self.call_uuid} {self.path} aleg")
            logger.info("FIFO broadcast: %s → %s", self.path, response.strip())
            self._broadcast_done = True

    def write(self, pcm16_bytes: bytes, open_timeout: float = 15.0) -> bool:
        """
        Write raw PCM16 bytes to the pipe.

        On the very first call this opens the write end of the pipe.  Opening
        blocks until FreeSWITCH has opened the read end (typically < 50 ms
        after broadcast()).  We do this in the calling thread (always run via
        asyncio.to_thread so it never blocks the event loop).

        The fd is switched to *blocking* after open so large writes are not
        truncated by O_NONBLOCK / EAGAIN (which caused silent gaps).

        Returns False if the pipe could not be opened or the reader closed it.
        """
        if not pcm16_bytes:
            return True

        with self._lock:
            if self._fd is None:
                fd = self._open_write_end(open_timeout)
                if fd is None:
                    return False
                # Critical: non-blocking open is only for the poll loop; writes
                # must block or we lose audio on full kernel pipe buffers.
                try:
                    flags = fcntl.fcntl(fd, fcntl.F_GETFL)
                    fcntl.fcntl(fd, fcntl.F_SETFL, flags & ~os.O_NONBLOCK)
                except OSError as exc:
                    logger.warning("FIFO could not set blocking mode: %s", exc)
                self._fd = fd
                self._open_event.set()
                logger.debug("FIFO write end opened: %s", self.path)

            try:
                data = memoryview(pcm16_bytes)
                offset = 0
                while offset < len(data):
                    n = os.write(self._fd, data[offset:])
                    offset += n
                return True
            except BrokenPipeError:
                logger.warning("FIFO broken pipe (FS closed read end): %s", self.path)
                self._fd = None
                return False
            except OSError as exc:
                logger.warning("FIFO write error %s: %s", self.path, exc)
                self._fd = None
                return False

    def close_write(self) -> None:
        """
        Close the write end.  FreeSWITCH reads EOF and stops playing.
        Safe to call multiple times.
        """
        with self._lock:
            if self._fd is not None:
                try:
                    os.close(self._fd)
                except OSError:
                    pass
                self._fd = None
                logger.debug("FIFO write end closed: %s", self.path)

    def cleanup(self) -> None:
        """Close write end (if still open) and remove pipe from filesystem."""
        self.close_write()
        try:
            os.unlink(self.path)
            logger.debug("FIFO removed: %s", self.path)
        except OSError:
            pass

    # ── Internal ─────────────────────────────────────────────────────────────

    def _open_write_end(self, timeout: float) -> Optional[int]:
        """
        Open the write end of the pipe, waiting up to *timeout* seconds for
        FreeSWITCH to open the read end.

        We poll with O_NONBLOCK so we don't hold the GIL indefinitely.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                fd = os.open(self.path, os.O_WRONLY | os.O_NONBLOCK)
                return fd
            except OSError as exc:
                if exc.errno == errno.ENXIO:
                    # No reader yet — FreeSWITCH is still opening the pipe
                    time.sleep(0.005)
                    continue
                logger.error("FIFO open failed %s: %s", self.path, exc)
                return None
        logger.error("FIFO open timed out after %.1fs: %s", timeout, self.path)
        return None


def _normalize_stream_pcm_to_freeswitch(pcm_bytes: bytes) -> bytes:
    """Convert streamed Indic PCM into FreeSWITCH's 8kHz mono PCM16 format."""
    if not pcm_bytes:
        return b""

    if INDIC_STREAM_CHANNELS > 1:
        pcm_bytes = audioop.tomono(
            pcm_bytes,
            INDIC_STREAM_SAMPLE_WIDTH_BYTES,
            0.5,
            0.5,
        )

    if INDIC_STREAM_SAMPLE_WIDTH_BYTES != PCM_SAMPLE_WIDTH_BYTES:
        pcm_bytes = audioop.lin2lin(
            pcm_bytes,
            INDIC_STREAM_SAMPLE_WIDTH_BYTES,
            PCM_SAMPLE_WIDTH_BYTES,
        )

    if INDIC_STREAM_SAMPLE_RATE != PCM_SAMPLE_RATE:
        pcm_array = np.frombuffer(pcm_bytes, dtype=np.int16)
        g = gcd(PCM_SAMPLE_RATE, INDIC_STREAM_SAMPLE_RATE)
        up = PCM_SAMPLE_RATE // g
        down = INDIC_STREAM_SAMPLE_RATE // g
        resampled = resample_poly(pcm_array, up, down)
        pcm_bytes = np.clip(resampled, -32768, 32767).astype(np.int16).tobytes()

    return pcm_bytes


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


def _schedule_temp_delete(path: str, delay: float = 60.0) -> None:
    def delayed_delete() -> None:
        time.sleep(delay)
        try:
            os.unlink(path)
        except Exception:
            pass

    threading.Thread(target=delayed_delete, daemon=True).start()


def _write_stream_segment_wav(pcm_bytes: bytes) -> str:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
        temp_path = handle.name

    with wave.open(temp_path, "wb") as wav_file:
        wav_file.setnchannels(PCM_CHANNELS)
        wav_file.setsampwidth(PCM_SAMPLE_WIDTH_BYTES)
        wav_file.setframerate(PCM_SAMPLE_RATE)
        wav_file.writeframes(pcm_bytes)

    _schedule_temp_delete(temp_path)
    return temp_path


def play_to_caller(uuid: str, pcm16_audio: bytes) -> None:
    if not uuid:
        raise ValueError("Call UUID is required for playback")
    if not pcm16_audio:
        raise ValueError("PCM16 audio is required for playback")

    def delayed_delete(path: str, delay: float) -> None:
        time.sleep(delay)
        try:
            os.unlink(path)
        except Exception:
            pass

    temp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
            temp_path = handle.name

        with wave.open(temp_path, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(8000)
            wav_file.writeframes(pcm16_audio)

        response = esl_api(f"uuid_broadcast {uuid} {temp_path} aleg")
        logger.info("Playback ESL response for call_uuid=%s file=%s: %s", uuid, temp_path, response.strip())

        # Keep file alive long enough for FreeSWITCH to play it.
        # All chunks are broadcast nearly simultaneously but play sequentially —
        # a later chunk's file must survive while all earlier chunks play.
        # Use 3× audio duration (min 30s) as a safe margin.
        audio_seconds = len(pcm16_audio) / (8000 * 2)
        delete_delay = max(30.0, audio_seconds * 3)
        if temp_path:
            threading.Thread(target=delayed_delete, args=(temp_path, delete_delay), daemon=True).start()
    except Exception:
        logger.exception("Playback failed for call_uuid=%s", uuid)
        if temp_path:
            threading.Thread(target=delayed_delete, args=(temp_path, 30.0), daemon=True).start()


def register_rtp_sink(call_uuid: str, sink: RtpPcmSink) -> None:
    _rtp_sinks[call_uuid] = sink


def unregister_rtp_sink(call_uuid: str) -> None:
    sink = _rtp_sinks.pop(call_uuid, None)
    if sink:
        sink.close()


async def deliver_outbound_pcm(call_uuid: str, pcm16_audio: bytes) -> None:
    """Play 8 kHz mono PCM16 to caller: RTP if a sink is registered, else uuid_broadcast file."""
    if not pcm16_audio:
        return
    sink = _rtp_sinks.get(call_uuid)
    if sink is not None:
        await sink.feed_pcm16(pcm16_audio)
        await sink.flush_playout()
        return
    await asyncio.to_thread(play_to_caller, call_uuid, pcm16_audio)


def _read_wav_pcm16_mono(path: str) -> bytes:
    with wave.open(path, "rb") as wf:
        if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getframerate() != 8000:
            raise ValueError(
                f"Expected 8 kHz mono PCM16 WAV, got ch={wf.getnchannels()} "
                f"width={wf.getsampwidth()} rate={wf.getframerate()}"
            )
        return wf.readframes(wf.getnframes())


def stop_ai_playback(call_uuid: str) -> bool:
    """
    Stop AI playback by breaking current audio playback.
    Uses uuid_break with 'both' direction to stop playback without hanging up.
    Returns True if successful, False otherwise.
    """
    logger.info("Stopping AI playback for call_uuid=%s", call_uuid)
    # Do NOT use "both" — it breaks all queued apps including sleep(300000)
    # which would cause the dialplan to end and hang up the call.
    # Plain uuid_break only stops the current playback (depth=1 uuid_broadcast).
    try:
        response = esl_api(f"uuid_break {call_uuid}")
        logger.info("Break ESL response for call_uuid=%s: %s", call_uuid, response.strip())
        if "-ERR" in response:
            logger.warning("Playback break failed for call_uuid=%s: %s", call_uuid, response.strip())
            return False
        return True
    except Exception:
        logger.exception("Playback break failed for call_uuid=%s", call_uuid)
        return False


async def _iter_ws_pcm_chunks(ws: WebSocket) -> AsyncIterator[bytes]:
    """Yield 8 kHz mono PCM16 chunks as received from FreeSWITCH (any size)."""
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

            if audio_chunk:
                yield audio_chunk
    except WebSocketDisconnect:
        return


async def run_voice_call_session(
    call_uuid: str,
    client_label: str,
    pcm_source: AsyncIterator[bytes],
) -> None:
    """
    Shared realtime path: inbound PCM stream → VAD/STT/brain/TTS → outbound (RTP or ESL).
    """
    logger.info("Voice session start call_uuid=%s client=%s", call_uuid, client_label)
    session = CallSession()
    segmenter = SpeechSegmenter()
    pcm_buffer = bytearray()
    segment_tasks: set[asyncio.Task[None]] = set()
    response_tasks: set[asyncio.Task[None]] = set()
    utterance_states: dict[int, dict[str, object]] = {}
    sentence_queue: asyncio.Queue[tuple[int, int, str] | None] = asyncio.Queue()
    playback_queue: asyncio.Queue[tuple[int, int, bytes] | None] = asyncio.Queue()
    playback_state: dict[str, float] = {"suppress_until": 0.0}
    ai_state: dict[str, int | float | bool] = {
        "speaking": False,
        "speaking_until": 0.0,
        "playback_ends_at": 0.0,
        "generation": 0,
        "playback_token": 0,
        "sentence_seq": 0,
    }
    active_fifo: dict[str, Optional[_CallFifo]] = {"fifo": None}

    tts_tasks = [
        asyncio.create_task(
            _tts_generation_worker(
                worker_id=0,
                sentence_queue=sentence_queue,
                playback_queue=playback_queue,
                client_label=client_label,
                ai_state=ai_state,
                playback_state=playback_state,
                call_uuid=call_uuid,
                active_fifo=active_fifo,
            )
        )
    ]
    playback_task = asyncio.create_task(
        _stream_playback_worker(
            playback_queue,
            client_label,
            call_uuid,
            playback_state,
            ai_state,
            active_fifo,
        )
    )

    try:
        async for audio_chunk in pcm_source:
            pcm_buffer.extend(audio_chunk)

            while len(pcm_buffer) >= 320:
                frame = bytes(pcm_buffer[:320])
                del pcm_buffer[:320]

                session.add_raw_chunk(frame)
                session.add_inbound_chunk(frame)

                if time.monotonic() < playback_state["suppress_until"] and not _is_ai_speaking(ai_state):
                    segmenter.reset()
                    continue

                segment_result = segmenter.process_frame(frame)

                if segment_result.speech_started and _is_ai_speaking(ai_state):
                    if segment_result.rms < BARGE_IN_RMS_THRESHOLD:
                        logger.info(
                            "Ignoring speech_start rms=%s during AI playback "
                            "(barge-in needs rms>=%s); likely echo",
                            segment_result.rms,
                            BARGE_IN_RMS_THRESHOLD,
                        )
                        segmenter.reset()
                        continue
                    else:
                        logger.info("Barge-in detected")
                        asyncio.create_task(
                            _handle_barge_in(
                                call_uuid,
                                sentence_queue,
                                playback_queue,
                                playback_state,
                                ai_state,
                                response_tasks,
                                segmenter,
                                active_fifo,
                            )
                        )

                if segment_result.segment_completed and segment_result.segment_audio:
                    state = utterance_states.setdefault(
                        segment_result.utterance_id,
                        {"llm_started": False},
                    )
                    if state.get("llm_started"):
                        utterance_states.pop(segment_result.utterance_id, None)
                        continue

                    task = asyncio.create_task(
                        _process_final_segment(
                            segment_result.segment_audio,
                            client_label,
                            segment_result.utterance_id,
                            sentence_queue,
                            ai_state,
                            state,
                        )
                    )
                    segment_tasks.add(task)
                    task.add_done_callback(segment_tasks.discard)
                    utterance_states.pop(segment_result.utterance_id, None)

    finally:
        trailing_segment = segmenter.flush()
        if trailing_segment.segment_completed and trailing_segment.segment_audio:
            state = utterance_states.setdefault(
                trailing_segment.utterance_id,
                {"llm_started": False},
            )
            if not state.get("llm_started"):
                task = asyncio.create_task(
                    _process_final_segment(
                        trailing_segment.segment_audio,
                        client_label,
                        trailing_segment.utterance_id,
                        sentence_queue,
                        ai_state,
                        state,
                    )
                )
                segment_tasks.add(task)
                task.add_done_callback(segment_tasks.discard)

        if response_tasks:
            await asyncio.gather(*response_tasks, return_exceptions=True)

        if segment_tasks:
            await asyncio.gather(*segment_tasks, return_exceptions=True)

        await sentence_queue.put(None)
        await asyncio.gather(*tts_tasks, return_exceptions=True)
        await playback_queue.put(None)
        await playback_task

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

        unregister_rtp_sink(call_uuid)


@router.websocket("/audio-stream")
@router.websocket("/audio")
@router.websocket("/ws/audio")
async def audio_stream(ws: WebSocket) -> None:
    await ws.accept()
    call_uuid = ws.query_params.get("uuid")
    if not call_uuid:
        logger.error("Missing FreeSWITCH call UUID in websocket query params")
        await ws.close(code=1008, reason="Missing call UUID")
        return

    logger.info("Connected FreeSWITCH call UUID: %s", call_uuid)
    client = getattr(ws, "client", None)
    client_label = f"{client.host}:{client.port}" if client else "unknown"

    await run_voice_call_session(call_uuid, client_label, _iter_ws_pcm_chunks(ws))


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


async def _handle_barge_in(
    call_uuid: str,
    sentence_queue: asyncio.Queue[tuple[int, int, str] | None],
    playback_queue: asyncio.Queue[tuple[int, int, bytes] | None],
    playback_state: dict[str, float],
    ai_state: dict[str, int | float | bool],
    response_tasks: set[asyncio.Task[None]],
    segmenter: SpeechSegmenter,
    active_fifo: dict[str, Optional[_CallFifo]],
) -> None:
    """
    Handle barge-in by stopping AI playback and clearing queues.
    This function is fire-and-forget and should not block the main loop.
    """
    try:
        old_generation = int(ai_state["generation"])
        ai_state["generation"] = old_generation + 1
        ai_state["speaking"] = False
        ai_state["speaking_until"] = 0.0
        ai_state["playback_token"] = int(ai_state["playback_token"]) + 1
        ai_state["sentence_seq"] = 0

        # Clear suppression immediately so user speech after barge-in flows through.
        playback_state["suppress_until"] = 0.0
        playback_state["pure_audio_end"] = 0.0

        # Close the FIFO write end → FreeSWITCH reads EOF → playback stops.
        # This replaces uuid_break entirely: no race condition with sleep(300000).
        fifo = active_fifo.get("fifo")
        if fifo is not None:
            await asyncio.to_thread(fifo.close_write)
            asyncio.create_task(_async_fifo_cleanup(fifo))
            active_fifo["fifo"] = None
            logger.info("Barge-in: FIFO closed, FreeSWITCH will stop playing")

        rtp_sink = _rtp_sinks.get(call_uuid)
        if rtp_sink is not None:
            rtp_sink.halt_playback()
            logger.info("Barge-in: RTP sink halted")

        # Clear queues and signal worker
        _clear_sentence_queue(sentence_queue, 1)
        _clear_playback_queue(playback_queue)

        # Cancel pending LLM responses
        for response_task in list(response_tasks):
            if not response_task.done():
                response_task.cancel()
        response_tasks.clear()

        logger.info("Barge-in handled successfully, listening resumed")
    except Exception:
        logger.exception("Barge-in handling failed")


async def _async_fifo_cleanup(fifo: _CallFifo, delay: float = 3.0) -> None:
    """Remove the FIFO pipe from filesystem after FreeSWITCH finishes reading."""
    await asyncio.sleep(delay)
    fifo.cleanup()


async def _process_streaming_segment(
    audio_bytes: bytes,
    client_label: str,
    utterance_id: int,
    sentence_queue: asyncio.Queue[tuple[int, int, str] | None],
    ai_state: dict[str, int | float | bool],
    state: dict[str, object],
) -> None:
    """
    Process streaming STT audio and start LLM response incrementally.
    This allows the AI to start responding before the user finishes speaking.
    """
    try:
        if state.get("llm_started"):
            return

        transcript = await transcribe_audio(audio_bytes)
        if not transcript:
            logger.info("Streaming STT returned empty transcript for %s", client_label)
            return
        if state.get("llm_started"):
            return

        state["llm_started"] = True
        generation = int(ai_state["generation"])
        response = await _stream_response_from_transcript(
            transcript,
            generation,
            sentence_queue,
            ai_state,
        )
        await append_brain_response(transcript, response)
        print(f"User (streaming): {transcript}")
        print(f"AI: {response}")
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.exception(
            "Streaming STT/LLM processing failed for %s utterance=%s",
            client_label,
            utterance_id,
        )


async def _start_response_from_partial_stt(
    audio_bytes: bytes,
    client_label: str,
    utterance_id: int,
    sentence_queue: asyncio.Queue[tuple[int, int, str] | None],
    ai_state: dict[str, int | float | bool],
    state: dict[str, object],
) -> None:
    try:
        transcript = await transcribe_audio(audio_bytes)
        if not transcript:
            logger.info("Partial STT returned empty transcript for %s", client_label)
            return
        if state.get("llm_started"):
            return

        state["llm_started"] = True
        generation = int(ai_state["generation"])
        response = await _stream_response_from_transcript(
            transcript,
            generation,
            sentence_queue,
            ai_state,
        )
        await append_brain_response(transcript, response)
        print(f"User: {transcript}")
        print(f"AI: {response}")
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.exception(
            "Partial STT/LLM processing failed for %s utterance=%s",
            client_label,
            utterance_id,
        )


async def _process_final_segment(
    audio_bytes: bytes,
    client_label: str,
    utterance_id: int,
    sentence_queue: asyncio.Queue[tuple[int, int, str] | None],
    ai_state: dict[str, int | float | bool],
    state: dict[str, object],
) -> None:
    try:
        if state.get("llm_started"):
            return

        transcript = await transcribe_audio(audio_bytes)
        if not transcript:
            logger.info("STT returned empty transcript for %s", client_label)
            return
        if state.get("llm_started"):
            return

        state["llm_started"] = True
        generation = int(ai_state["generation"])
        response = await _stream_response_from_transcript(
            transcript,
            generation,
            sentence_queue,
            ai_state,
        )
        await append_brain_response(transcript, response)
        print(f"User: {transcript}")
        print(f"AI: {response}")
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.exception("Speech segment response failed for %s utterance=%s", client_label, utterance_id)


async def _stream_response_from_transcript(
    transcript: str,
    generation: int,
    sentence_queue: asyncio.Queue[tuple[int, int, str] | None],
    ai_state: dict[str, int | float | bool],
) -> str:
    """Stream LLM response and forward chunks to TTS queue."""
    response_queue: asyncio.Queue[str | None] = asyncio.Queue()
    forward_task = asyncio.create_task(
        _forward_sentence_chunks(response_queue, sentence_queue, generation, ai_state)
    )
    try:
        return await process_transcript_to_reply_streaming(transcript, response_queue)
    finally:
        await response_queue.put(None)
        await forward_task


async def _forward_sentence_chunks(
    source_queue: asyncio.Queue[str | None],
    sentence_queue: asyncio.Queue[tuple[int, int, str] | None],
    generation: int,
    ai_state: dict[str, int | float | bool],
) -> None:
    """
    Forward LLM chunks to TTS queue.
    TTS worker will handle streaming + early broadcast.
    """
    sequence = 0
    
    while True:
        chunk = await source_queue.get()
        if chunk is None:
            break

        if int(ai_state["generation"]) != generation:
            continue

        sequence += 1
        logger.info(
            "Forwarding chunk %s (words=%s): %s",
            sequence, len(chunk.split()), chunk[:50]
        )
        
        # Put in queue for TTS worker
        await sentence_queue.put((generation, sequence, chunk))


async def _send_chunk_to_tts(
    chunk_text: str,
    target_queue: asyncio.Queue[tuple[int, int, str] | None],
    generation: int,
    ai_state: dict[str, int | float | bool],
    sequence: int,
    is_first_chunk: bool,
) -> None:
    """Send a chunk to TTS queue with logging."""
    global _global_seq_id
    _global_seq_id += 1
    seq_id = _global_seq_id

    word_count = len(chunk_text.split())
    logger.info(
        "Sending chunk to TTS: global_seq=%s chunk_idx=%s words=%s first=%s text=%.50s...",
        seq_id, sequence, word_count, is_first_chunk, chunk_text
    )
    ai_state["sentence_seq"] = seq_id
    await target_queue.put((generation, seq_id, chunk_text))


def _clear_sentence_queue(sentence_queue: asyncio.Queue[tuple[int, int, str] | None], worker_count: int = 1) -> None:
    """Clear sentence queue and send termination signals for all workers."""
    while True:
        try:
            sentence_queue.get_nowait()
        except asyncio.QueueEmpty:
            break
    # Send None for each worker to stop them cleanly
    for _ in range(worker_count):
        try:
            sentence_queue.put_nowait(None)
        except asyncio.QueueFull:
            pass
    logger.info("Sentence queue cleared for %s workers", worker_count)


def _clear_playback_queue(playback_queue: asyncio.Queue[tuple[int, int, bytes] | None]) -> None:
    while True:
        try:
            playback_queue.get_nowait()
        except asyncio.QueueEmpty:
            break
    logger.info("Playback queue cleared")


def _is_ai_speaking(ai_state: dict[str, int | float | bool]) -> bool:
    return bool(ai_state["speaking"]) or time.monotonic() < float(ai_state["speaking_until"])


async def _indic_blocking_synthesize_pcm(
    client: httpx.AsyncClient,
    text: str,
    language: str = "hi",
) -> bytes | None:
    """
    Blocking Indic TTS: POST /synthesize → full WAV → 8 kHz mono PCM16 for FreeSWITCH.
    Used when /synthesize/stream returns no bytes (wrong port, empty body, stream bug).
    """
    try:
        r = await client.post(
            f"{INDIC_TTS_URL}/synthesize",
            json={"text": text, "language": language},
            timeout=120.0,
        )
        if r.status_code != 200:
            logger.error(
                "Indic blocking TTS HTTP %s from %s/synthesize: %s",
                r.status_code,
                INDIC_TTS_URL,
                (r.text or "")[:1200],
            )
            return None
        with wave.open(io.BytesIO(r.content), "rb") as wf:
            ch, sw, fr = wf.getnchannels(), wf.getsampwidth(), wf.getframerate()
            if ch != 1 or sw != 2:
                logger.error(
                    "Indic WAV format unsupported: ch=%s width=%s rate=%s",
                    ch, sw, fr,
                )
                return None
            pcm = wf.readframes(wf.getnframes())
    except Exception:
        logger.exception("Indic blocking TTS parse/request failed")
        return None

    return await asyncio.to_thread(_normalize_stream_pcm_to_freeswitch, pcm)


def _accumulate_tts_playback_schedule(
    ai_state: dict[str, int | float | bool],
    playback_state: dict[str, float],
    pcm_chunk: bytes,
) -> float:
    """
    Mirror _stream_playback_worker timing: extend speaking_until / suppress_until
    so the gateway knows AI audio is playing (echo vs barge-in, uuid_break safety).
    *pcm_chunk* is 8 kHz mono PCM16 (after _normalize_stream_pcm_to_freeswitch).
    """
    if not pcm_chunk:
        return float(ai_state.get("speaking_until") or 0.0)

    playback_seconds = (len(pcm_chunk) / PCM_FRAME_BYTES) * 0.02
    now = time.monotonic()
    suppress = float(playback_state.get("suppress_until", 0.0))
    audio_end = max(suppress, now) + playback_seconds
    speaking_until = audio_end + (TTS_PLAYBACK_GUARD_MS / 1000.0)
    playback_state["suppress_until"] = speaking_until
    ai_state["speaking_until"] = speaking_until

    pure_audio_end = float(playback_state.get("pure_audio_end", now))
    pure_chunk_end = max(pure_audio_end, now) + playback_seconds
    playback_state["pure_audio_end"] = pure_chunk_end
    ai_state["playback_ends_at"] = pure_chunk_end
    ai_state["speaking"] = True
    return speaking_until


async def _tts_generation_worker(
    worker_id: int,
    sentence_queue: asyncio.Queue[tuple[int, int, str] | None],
    playback_queue: asyncio.Queue[tuple[int, int, bytes] | None],
    client_label: str,
    ai_state: dict[str, int | float | bool],
    playback_state: dict[str, float],
    call_uuid: str,
    active_fifo: dict[str, Optional["_CallFifo"]],
) -> None:
    """
    Single TTS worker: streams Indic TTS, then plays to FreeSWITCH.

    - USE_FIFO_PLAYBACK True: one named pipe + one uuid_broadcast, PCM streamed
      in ~0.5s writes for low latency.
    - USE_FIFO_PLAYBACK False: buffer full utterance PCM, one temp WAV +
      one uuid_broadcast (no per-segment broadcasts).
    """
    while True:
        item = await sentence_queue.get()

        if item is None:
            break

        generation, sequence, sentence = item
        current_gen = int(ai_state["generation"])

        # Skip if generation changed (barge-in occurred)
        if generation != current_gen:
            logger.info(
                "TTS worker-%s skipping seq=%s (generation mismatch)",
                worker_id, sequence,
            )
            continue

        # Only skip empty chunks (old filter len<10 dropped short Hindi e.g. "ठीक है।")
        if not sentence.strip():
            continue

        logger.info(
            "TTS worker-%s STARTED global_seq=%s words=%s fifo=%s",
            worker_id, sequence, len(sentence.split()), USE_FIFO_PLAYBACK,
        )
        tts_start = time.monotonic()

        fifo: Optional[_CallFifo] = None
        if USE_FIFO_PLAYBACK:
            fifo = _CallFifo(call_uuid, generation)
            fifo.create()
            active_fifo["fifo"] = fifo
        else:
            active_fifo["fifo"] = None

        chunk_count  = 0
        segment_index = 0
        header_bytes  = bytearray()
        pcm_buffer    = bytearray()
        streamed_pcm_bytes = 0
        fifo_broadcast_issued = False
        # Track outbound duration so _is_ai_speaking / barge-in echo rejection work
        # (uuid_broadcast path never fed playback_queue, so this was missing before).
        tts_playback_token: list[int | None] = [None]
        last_speaking_until: list[float] = [0.0]

        def _register_played_8k_pcm(pcm: bytes) -> None:
            if not pcm or int(ai_state["generation"]) != current_gen:
                return
            if tts_playback_token[0] is None:
                ai_state["playback_token"] = int(ai_state["playback_token"]) + 1
                tts_playback_token[0] = int(ai_state["playback_token"])
            last_speaking_until[0] = _accumulate_tts_playback_schedule(
                ai_state, playback_state, pcm
            )

        async with httpx.AsyncClient(timeout=120.0) as client:
            try:
                async with client.stream(
                    "POST",
                    f"{INDIC_TTS_URL}/synthesize/stream",
                    json={"text": sentence, "language": "hi"},
                ) as response:
                    if response.status_code != 200:
                        err_raw = await response.aread()
                        try:
                            err_txt = err_raw[:1500].decode("utf-8", errors="replace")
                        except Exception:
                            err_txt = repr(err_raw[:300])
                        logger.error(
                            "Indic TTS stream HTTP %s %s/synthesize/stream: %s",
                            response.status_code,
                            INDIC_TTS_URL,
                            err_txt,
                        )
                        if fifo is not None:
                            fifo.cleanup()
                        active_fifo["fifo"] = None
                    else:
                        async for chunk in response.aiter_bytes():
                            if not chunk:
                                continue

                            # Abort if generation changed mid-stream (barge-in)
                            if int(ai_state["generation"]) != current_gen:
                                break

                            chunk_count += 1
                            chunk_view = memoryview(chunk)

                            # Strip WAV header from first bytes
                            if len(header_bytes) < INDIC_STREAM_WAV_HEADER_BYTES:
                                header_needed = INDIC_STREAM_WAV_HEADER_BYTES - len(header_bytes)
                                header_bytes.extend(chunk_view[:header_needed])
                                chunk_view = chunk_view[header_needed:]
                                if not chunk_view:
                                    continue

                            pcm_buffer.extend(chunk_view)

                            # FIFO: flush 0.5s segments as they accumulate (low latency).
                            # Non-FIFO: buffer entire stream → one WAV + one uuid_broadcast below.
                            if USE_FIFO_PLAYBACK and fifo is not None:
                                while len(pcm_buffer) >= INDIC_STREAM_SEGMENT_PCM_BYTES:
                                    segment_pcm = bytes(pcm_buffer[:INDIC_STREAM_SEGMENT_PCM_BYTES])
                                    del pcm_buffer[:INDIC_STREAM_SEGMENT_PCM_BYTES]
                                    streamed_pcm_bytes += len(segment_pcm)
                                    segment_index += 1

                                    freeswitch_pcm = await asyncio.to_thread(
                                        _normalize_stream_pcm_to_freeswitch, segment_pcm,
                                    )

                                    if not fifo_broadcast_issued:
                                        fifo.broadcast()
                                        fifo_broadcast_issued = True
                                    ok = await asyncio.to_thread(fifo.write, freeswitch_pcm)
                                    _register_played_8k_pcm(freeswitch_pcm)
                                    logger.info(
                                        "TTS worker-%s FIFO segment seq=%s seg=%s bytes=%s ok=%s",
                                        worker_id, sequence, segment_index,
                                        len(freeswitch_pcm), ok,
                                    )

                        # Remaining PCM: FIFO tail, or non-FIFO full utterance in one shot
                        if pcm_buffer and int(ai_state["generation"]) == current_gen:
                            segment_pcm = bytes(pcm_buffer)
                            streamed_pcm_bytes += len(segment_pcm)
                            segment_index += 1
                            freeswitch_pcm = await asyncio.to_thread(
                                _normalize_stream_pcm_to_freeswitch, segment_pcm,
                            )
                            if USE_FIFO_PLAYBACK and fifo is not None:
                                if not fifo_broadcast_issued:
                                    fifo.broadcast()
                                    fifo_broadcast_issued = True
                                ok = await asyncio.to_thread(fifo.write, freeswitch_pcm)
                                _register_played_8k_pcm(freeswitch_pcm)
                                logger.info(
                                    "TTS worker-%s FIFO final seq=%s seg=%s bytes=%s ok=%s",
                                    worker_id, sequence, segment_index,
                                    len(freeswitch_pcm), ok,
                                )
                            else:
                                _register_played_8k_pcm(freeswitch_pcm)
                                segment_path = await asyncio.to_thread(
                                    _write_stream_segment_wav, freeswitch_pcm,
                                )
                                await _broadcast_wav_segment(
                                    worker_id,
                                    sequence,
                                    segment_index,
                                    segment_path,
                                    current_gen,
                                    call_uuid,
                                )

                # Stream often returns 200 with zero bytes if nothing listens on INDIC_TTS_URL
                # or the worker errors before yielding — fall back to blocking /synthesize.
                if (
                    tts_playback_token[0] is None
                    and int(ai_state["generation"]) == current_gen
                    and sentence.strip()
                ):
                    if fifo is not None:
                        try:
                            fifo.cleanup()
                        except Exception:
                            pass
                        active_fifo["fifo"] = None
                        fifo = None
                    logger.warning(
                        "Indic stream produced no playable audio (chunks=%s); "
                        "trying blocking POST %s/synthesize (%s chars)",
                        chunk_count,
                        INDIC_TTS_URL,
                        len(sentence),
                    )
                    fs_pcm = await _indic_blocking_synthesize_pcm(client, sentence, "hi")
                    if fs_pcm and int(ai_state["generation"]) == current_gen:
                        segment_index = 1
                        streamed_pcm_bytes = len(fs_pcm)
                        chunk_count = max(chunk_count, 1)
                        _register_played_8k_pcm(fs_pcm)
                        segment_path = await asyncio.to_thread(
                            _write_stream_segment_wav, fs_pcm,
                        )
                        await _broadcast_wav_segment(
                            worker_id,
                            sequence,
                            segment_index,
                            segment_path,
                            current_gen,
                            call_uuid,
                        )

            except Exception:
                logger.exception("TTS streaming failed for %s seq=%s", client_label, sequence)

        if tts_playback_token[0] is not None and last_speaking_until[0] > 0:
            asyncio.create_task(
                _mark_playback_complete(
                    ai_state, last_speaking_until[0], tts_playback_token[0]
                )
            )

        if USE_FIFO_PLAYBACK and fifo is not None:
            await asyncio.to_thread(fifo.close_write)

            async def _delayed_fifo_cleanup(f: _CallFifo, delay: float = 5.0) -> None:
                await asyncio.sleep(delay)
                f.cleanup()

            asyncio.create_task(_delayed_fifo_cleanup(fifo))

        active_fifo["fifo"] = None

        tts_duration = (time.monotonic() - tts_start) * 1000
        logger.info(
            "TTS worker-%s DONE global_seq=%s duration=%.0fms chunks=%s segments=%s pcm_bytes=%s",
            worker_id, sequence, tts_duration, chunk_count, segment_index, streamed_pcm_bytes,
        )

        from app.pipeline.speech_pipeline import _tts_compute_times
        _tts_compute_times.append(tts_duration)
        if len(_tts_compute_times) > 10:
            _tts_compute_times.pop(0)

        if tts_playback_token[0] is None:
            logger.warning(
                "No TTS audio after stream + blocking fallback for %s global_seq=%s — "
                "check Indic on %s (is `python indic_server.py` running on 8002?)",
                client_label,
                sequence,
                INDIC_TTS_URL,
            )


async def _broadcast_wav_segment(
    worker_id: int,
    sequence: int,
    segment_index: int,
    temp_path: str,
    generation: int,
    call_uuid: str,
) -> None:
    """
    Deliver a finalized WAV segment to the caller: RTP if registered, else uuid_broadcast.
    """
    try:
        if _rtp_sinks.get(call_uuid) is not None:
            pcm = await asyncio.to_thread(_read_wav_pcm16_mono, temp_path)
            await deliver_outbound_pcm(call_uuid, pcm)
            logger.info(
                "Segment delivered (RTP) worker-%s seq=%s seg=%s pcm_bytes=%s gen=%s",
                worker_id, sequence, segment_index, len(pcm), generation,
            )
            return

        response = esl_api(f"uuid_broadcast {call_uuid} {temp_path} aleg")
        if "-ERR" in response:
            logger.error(
                "Segment broadcast failed for worker-%s seq=%s segment=%s: %s",
                worker_id, sequence, segment_index, response.strip()
            )
        else:
            logger.info(
                "Segment broadcast succeeded for worker-%s seq=%s segment=%s: %s",
                worker_id, sequence, segment_index, response.strip()
            )
    except Exception as e:
        logger.exception(
            "Segment broadcast error for worker-%s seq=%s segment=%s: %s",
            worker_id, sequence, segment_index, e,
        )


async def _stream_playback_worker(
    playback_queue: asyncio.Queue[tuple[int, int, bytes] | None],
    client_label: str,
    call_uuid: str,
    playback_state: dict[str, float],
    ai_state: dict[str, int | float | bool],
    active_fifo: dict[str, Optional["_CallFifo"]],
) -> None:
    """
    Ordered playback worker with pipeline safety buffer.
    - Maintains playback buffer (dict: seq → audio)
    - Only plays when sequence ID matches next expected ID
    - Ensures strict ordering even with parallel TTS workers
    - Pipeline safety: waits for 2 chunks before starting playback
    """
    pending_audio: dict[int, bytes] = {}
    playback_generation = int(ai_state["generation"])
    next_sequence: int | None = None  # Will be set to first chunk's seq_id
    first_chunk_received_at: float | None = None  # For pipeline buffer timing
    PLAYBACK_BUFFER_DELAY_MS = 1500  # Wait for 2nd chunk or timeout
    MAX_QUEUE_SIZE = 3  # Limit queue to prevent explosion

    while True:
        item = await playback_queue.get()
        if item is None:
            break

        generation, sequence, pcm16_audio = item
        current_generation = int(ai_state["generation"])

        # Handle generation change (barge-in)
        if generation > playback_generation:
            playback_generation = generation
            pending_audio.clear()
            next_sequence = None
            first_chunk_received_at = None
            logger.info(
                "Playback: generation changed to %s, queue cleared, next_sequence reset",
                generation
            )

        if generation != current_generation or generation != playback_generation:
            if generation > current_generation:
                pending_audio.clear()
                next_sequence = None
                first_chunk_received_at = None
            continue

        # Discard old chunks (from previous generation or already played)
        if next_sequence is not None and sequence < next_sequence:
            logger.info(
                "Playback: DISCARDED old chunk seq=%s next=%s buffer_keys=%s",
                sequence, next_sequence, list(pending_audio.keys())
            )
            continue

        # Initialize next_sequence to first chunk's seq_id
        if next_sequence is None:
            next_sequence = sequence
            first_chunk_received_at = time.monotonic()
            logger.info(
                "Playback: FIRST chunk seq=%s, pipeline buffer started, buffer_size=%s",
                sequence, len(pending_audio) + 1
            )

        # Add to buffer
        pending_audio[sequence] = pcm16_audio
        logger.info(
            "Playback: received global_seq=%s buffer_size=%s next=%s buffer_keys=%s",
            sequence, len(pending_audio), next_sequence, sorted(pending_audio.keys())
        )

        # Play in order
        while next_sequence is not None and next_sequence in pending_audio:
            # Pipeline safety buffer: wait for 2nd chunk or timeout
            if first_chunk_received_at is not None:
                elapsed_ms = (time.monotonic() - first_chunk_received_at) * 1000
                has_multiple_chunks = len(pending_audio) >= 2
                
                if not has_multiple_chunks and elapsed_ms < PLAYBACK_BUFFER_DELAY_MS:
                    # Wait for 2nd chunk or timeout (pipeline safety)
                    await asyncio.sleep(0.1)  # Check every 100ms
                    continue
                
                # Pipeline buffer ready
                buffer_status = "2_chunks" if has_multiple_chunks else f"timeout_{int(elapsed_ms)}ms"
                first_chunk_received_at = None
                logger.info(
                    "Playback: pipeline buffer ready (%s), starting playback",
                    buffer_status
                )

            current_audio = pending_audio.pop(next_sequence)
            gen_at_playback = int(ai_state["generation"])
            playback_start_time = time.monotonic()

            if generation != gen_at_playback:
                pending_audio.clear()
                logger.info(
                    "Playback: generation changed during playback, clearing buffer, next_sequence reset"
                )
                break

            if not ai_state["speaking"] and gen_at_playback != generation:
                break

            playback_seconds = (len(current_audio) / PCM_FRAME_BYTES) * 0.02
            now = time.monotonic()
            audio_end = max(playback_state["suppress_until"], now) + playback_seconds
            speaking_until = audio_end + (TTS_PLAYBACK_GUARD_MS / 1000.0)
            playback_state["suppress_until"] = speaking_until
            ai_state["speaking"] = True
            ai_state["speaking_until"] = speaking_until

            # Track pure audio end WITHOUT guard accumulation
            pure_audio_end = playback_state.get("pure_audio_end", now)
            pure_chunk_end = max(pure_audio_end, now) + playback_seconds
            playback_state["pure_audio_end"] = pure_chunk_end
            ai_state["playback_ends_at"] = pure_chunk_end
            ai_state["playback_token"] = int(ai_state["playback_token"]) + 1
            playback_token = int(ai_state["playback_token"])

            async with PLAYBACK_LOCK:
                if int(ai_state["generation"]) != generation:
                    pending_audio.clear()
                    logger.info(
                        "Playback: generation changed during playback lock, clearing buffer"
                    )
                    break

                # playback_queue path (rare): WAV is reliable; FIFO optional
                if USE_FIFO_PLAYBACK:
                    fifo = active_fifo.get("fifo")
                    if fifo is None:
                        fifo = _CallFifo(call_uuid, generation)
                        fifo.create()
                        fifo.broadcast()
                        active_fifo["fifo"] = fifo
                    await asyncio.to_thread(fifo.write, current_audio)
                else:
                    await deliver_outbound_pcm(call_uuid, current_audio)

            playback_duration_ms = (time.monotonic() - playback_start_time) * 1000
            logger.info(
                "Audio chunk PLAYED global_seq=%s duration=%.2fs playback_time=%.0fms next=%s buffer_keys=%s",
                next_sequence, playback_seconds, playback_duration_ms,
                next_sequence + 1, sorted(pending_audio.keys())
            )
            asyncio.create_task(_mark_playback_complete(ai_state, speaking_until, playback_token))
            next_sequence += 1


async def _mark_playback_complete(
    ai_state: dict[str, int | float | bool],
    speaking_until: float,
    playback_token: int,
) -> None:
    delay_seconds = max(0.0, speaking_until - time.monotonic())
    await asyncio.sleep(delay_seconds)
    if playback_token == int(ai_state["playback_token"]):
        ai_state["speaking"] = False
        ai_state["playback_ends_at"] = 0.0


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
