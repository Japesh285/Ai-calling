"""
Register raw RTP sessions (symmetric RTP/PCMA) alongside the voice pipeline.

Typical flow:
1. POST /rtp/sessions {\"call_uuid\": \"<call uuid>\"} -> get listen_port
2. From FreeSWITCH (or a small RTP forwarder), send inbound RTP to RTP_ADVERTISE_HOST:listen_port
3. Gateway learns the far-end address from the first datagram and sends AI audio back there.

TTS stays HTTP; run indic_server.py and set INDIC_TTS_URL if not on :8002.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.api.websocket import register_rtp_sink, run_voice_call_session, unregister_rtp_sink
from app.config.settings import RTP_ADVERTISE_HOST, RTP_OUT_PAYLOAD_TYPE
from app.rtp.outbound import RtpPcmSink
from app.rtp.pcm_rtp import rtp_packet_to_pcm16

router = APIRouter()

_rtp_tasks: dict[str, asyncio.Task[None]] = {}


class RtpRegisterRequest(BaseModel):
    call_uuid: str


class RtpSessionInfo(BaseModel):
    advertise_host: str
    listen_port: int
    call_uuid: str
    payload_type: int
    note: str


class RtpRecvProtocol(asyncio.DatagramProtocol):
    def __init__(self, sink: RtpPcmSink, packet_queue: asyncio.Queue[bytes]) -> None:
        self._sink = sink
        self._packet_queue = packet_queue

    def datagram_received(self, data: bytes, addr: Any) -> None:
        self._sink.set_peer(addr)
        try:
            self._packet_queue.put_nowait(data)
        except asyncio.QueueFull:
            pass


async def _iter_rtp_pcm(queue: asyncio.Queue[bytes]) -> AsyncIterator[bytes]:
    while True:
        pkt = await queue.get()
        pcm = rtp_packet_to_pcm16(pkt)
        if pcm:
            yield pcm


def _task_done_cb(uuid_key: str, task: asyncio.Task[None]) -> None:
    _rtp_tasks.pop(uuid_key, None)


@router.post("/rtp/sessions", response_model=RtpSessionInfo)
async def create_rtp_session(body: RtpRegisterRequest) -> RtpSessionInfo:
    call_uuid = body.call_uuid.strip()
    if not call_uuid:
        raise HTTPException(status_code=400, detail="call_uuid required")

    old = _rtp_tasks.pop(call_uuid, None)
    if old and not old.done():
        old.cancel()
        try:
            await old
        except asyncio.CancelledError:
            pass
    unregister_rtp_sink(call_uuid)

    packet_q: asyncio.Queue[bytes] = asyncio.Queue(maxsize=1024)
    sink = RtpPcmSink(payload_type=RTP_OUT_PAYLOAD_TYPE)

    loop = asyncio.get_running_loop()
    transport, _protocol = await loop.create_datagram_endpoint(
        lambda: RtpRecvProtocol(sink, packet_q),
        local_addr=("0.0.0.0", 0),
    )
    sink.attach_transport(transport)
    register_rtp_sink(call_uuid, sink)

    task = asyncio.create_task(
        run_voice_call_session(
            call_uuid,
            f"rtp:{call_uuid[:8]}",
            _iter_rtp_pcm(packet_q),
        ),
    )
    _rtp_tasks[call_uuid] = task
    task.add_done_callback(lambda t, u=call_uuid: _task_done_cb(u, t))

    port = transport.get_extra_info("sockname")[1]
    return RtpSessionInfo(
        advertise_host=RTP_ADVERTISE_HOST,
        listen_port=port,
        call_uuid=call_uuid,
        payload_type=RTP_OUT_PAYLOAD_TYPE,
        note="Send RTP (PT 8 PCMA or PT 0 PCMU, 8 kHz) here; return audio uses the same PT.",
    )


@router.delete("/rtp/sessions/{call_uuid}")
async def delete_rtp_session(call_uuid: str) -> dict[str, bool]:
    t = _rtp_tasks.pop(call_uuid, None)
    if t and not t.done():
        t.cancel()
        try:
            await t
        except asyncio.CancelledError:
            pass
    unregister_rtp_sink(call_uuid)
    return {"ok": True}
