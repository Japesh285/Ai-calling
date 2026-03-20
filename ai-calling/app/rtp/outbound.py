"""Send 8 kHz mono PCM16 toward FreeSWITCH as RTP (G.711), with jitter-buffered playout."""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

from app.config.settings import RTP_PLAYOUT_PREBUFFER_MS

from app.rtp.pcm_rtp import (
    SAMPLES_PER_PACKET,
    build_rtp_packet,
    pcm16_le_to_alaw,
    pcm16_le_to_ulaw,
    random_ssrc,
)

logger = logging.getLogger(__name__)

Address = tuple[str, int]

_FRAME_BYTES = SAMPLES_PER_PACKET * 2  # 20 ms @ 8 kHz mono PCM16


class RtpPcmSink:
    """
    Symmetric RTP sender: peer from first inbound datagram.

    *feed_pcm16* / *flush_playout* split lets you push bursty HTTP TTS chunks while RTP
    is emitted on a steady ~20 ms clock after an initial pre-buffer (reduces gaps from
    Indic streaming bursts). *send_pcm16* is feed+flush for one blob (back compat).
    """

    def __init__(self, *, payload_type: int = 8, prebuffer_ms: Optional[float] = None) -> None:
        self._payload_type = payload_type
        ms = float(RTP_PLAYOUT_PREBUFFER_MS if prebuffer_ms is None else prebuffer_ms)
        self._prebuffer_bytes = max(_FRAME_BYTES, int(8000 * 2 * (ms / 1000.0)))
        self._transport: Optional[asyncio.DatagramTransport] = None
        self._peer: Optional[Address] = None
        self._peer_event = asyncio.Event()
        self._seq = 0
        self._ts = 0
        self._ssrc = random_ssrc()
        self._halt_id = 0

        self._buf = bytearray()
        self._buf_lock = asyncio.Lock()
        self._feed_done = False
        self._primed = False
        self._drain_task: Optional[asyncio.Task[None]] = None
        self._marker_next = True

    def attach_transport(self, transport: asyncio.DatagramTransport) -> None:
        self._transport = transport

    def set_peer(self, addr: Address) -> None:
        if self._peer is None:
            self._peer = addr
            self._peer_event.set()

    async def wait_peer(self, timeout: float = 30.0) -> bool:
        try:
            await asyncio.wait_for(self._peer_event.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            return False
        return self._peer is not None

    def halt_playback(self) -> None:
        """Barge-in: stop drain, clear buffer, allow pre-buffer on next phrase."""
        self._halt_id += 1
        self._ssrc = random_ssrc()
        self._buf.clear()
        self._feed_done = False
        self._primed = False
        self._marker_next = True
        t = self._drain_task
        self._drain_task = None
        if t and not t.done():
            t.cancel()

    def _encode_payload(self, pcm_chunk: bytes) -> bytes:
        if self._payload_type == 8:
            return pcm16_le_to_alaw(pcm_chunk)
        if self._payload_type == 0:
            return pcm16_le_to_ulaw(pcm_chunk)
        raise ValueError(f"Unsupported RTP PT {self._payload_type} (use 8=PCMA, 0=PCMU)")

    async def feed_pcm16(self, pcm: bytes) -> None:
        if not pcm or not self._transport:
            return
        if not await self.wait_peer():
            return
        if self._peer is None:
            return
        async with self._buf_lock:
            self._buf.extend(pcm)
        await self._ensure_drain()

    async def flush_playout(self) -> None:
        """Mark end of current push; wait until all buffered PCM has been sent."""
        async with self._buf_lock:
            self._feed_done = True
        await self._ensure_drain()
        if self._drain_task:
            try:
                await self._drain_task
            except asyncio.CancelledError:
                pass
            self._drain_task = None
        async with self._buf_lock:
            self._feed_done = False

    async def _ensure_drain(self) -> None:
        if self._drain_task is None or self._drain_task.done():
            self._drain_task = asyncio.create_task(self._drain_loop())

    async def _drain_loop(self) -> None:
        play_id = self._halt_id
        peer = self._peer
        transport = self._transport
        if not peer or not transport:
            return

        try:
            while True:
                if play_id != self._halt_id:
                    return

                frame: Optional[bytes] = None
                wait_for_data = False

                async with self._buf_lock:
                    n = len(self._buf)
                    feed_done = self._feed_done

                    if not self._primed:
                        if n >= self._prebuffer_bytes:
                            self._primed = True
                        elif feed_done and n >= _FRAME_BYTES:
                            self._primed = True
                        elif feed_done and 0 < n < _FRAME_BYTES:
                            self._primed = True
                        else:
                            wait_for_data = not feed_done or n < _FRAME_BYTES
                    if self._primed and n >= _FRAME_BYTES:
                        frame = bytes(self._buf[:_FRAME_BYTES])
                        del self._buf[:_FRAME_BYTES]
                    elif self._primed and feed_done and n > 0:
                        frame = bytes(self._buf) + (b"\x00" * (_FRAME_BYTES - n))
                        self._buf.clear()
                    elif feed_done and n == 0 and self._primed:
                        return
                    elif feed_done and n == 0 and not self._primed:
                        return
                    elif not wait_for_data:
                        wait_for_data = not feed_done

                if frame is None:
                    if wait_for_data:
                        await asyncio.sleep(0.01)
                    else:
                        await asyncio.sleep(0.002)
                    continue

                try:
                    payload = self._encode_payload(frame)
                except ValueError:
                    return

                pkt = build_rtp_packet(
                    self._seq,
                    self._ts,
                    self._ssrc,
                    payload,
                    marker=self._marker_next,
                    pt=self._payload_type,
                )
                self._marker_next = False
                transport.sendto(pkt, peer)
                self._seq = (self._seq + 1) & 0xFFFF
                self._ts = (self._ts + SAMPLES_PER_PACKET) & 0xFFFFFFFF
                await asyncio.sleep(0.02)
        except asyncio.CancelledError:
            raise

    async def send_pcm16(self, pcm: bytes) -> None:
        """One-shot: queue all *pcm* and drain (pre-buffer applies once per halted session)."""
        await self.feed_pcm16(pcm)
        await self.flush_playout()

    def close(self) -> None:
        self.halt_playback()
        if self._transport:
            self._transport.close()
            self._transport = None
