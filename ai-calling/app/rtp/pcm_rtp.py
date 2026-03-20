"""
Minimal RTP helpers: PCMA (PT 8) / PCMU (PT 0) payload ↔ 8 kHz mono PCM16 LE.

Intended for symmetric RTP to FreeSWITCH (learn peer from first inbound datagram).
"""

from __future__ import annotations

import audioop
import random
import struct
from typing import Optional, Tuple

# 8 kHz, 20 ms = 160 samples
SAMPLES_PER_PACKET = 160


def _parse_rtp_fixed_header(data: bytes) -> Optional[Tuple[int, int, int, int, int]]:
    """
    Return (version, pt, seq, ts, ssrc_header_len) or None.
    ssrc_header_len = 12 + 4 * cc  (start of payload), excludes extension payload.
    """
    if len(data) < 12:
        return None
    v = (data[0] >> 6) & 0x3
    if v != 2:
        return None
    cc = data[0] & 0x0F
    pt = data[1] & 0x7F
    seq = struct.unpack_from("!H", data, 2)[0]
    ts = struct.unpack_from("!I", data, 4)[0]
    header_end = 12 + cc * 4
    if len(data) < header_end:
        return None
    if data[0] & 0x10:  # X — header extension present
        if len(data) < header_end + 4:
            return None
        ext_len = struct.unpack_from("!H", data, header_end + 2)[0]
        header_end += 4 + ext_len * 4
        if len(data) < header_end:
            return None
    return v, pt, seq, ts, header_end


def rtp_packet_to_pcm16(rtp: bytes) -> bytes:
    """Extract RTP payload and convert to PCM16 LE mono 8 kHz."""
    parsed = _parse_rtp_fixed_header(rtp)
    if parsed is None:
        return b""
    _v, pt, _seq, _ts, off = parsed
    payload = rtp[off:]
    if not payload:
        return b""
    if pt == 8:  # PCMA (A-law)
        return audioop.alaw2lin(payload, 2)
    if pt == 0:  # PCMU (µ-law)
        return audioop.ulaw2lin(payload, 2)
    # L16 mono @ 8k (PT 11 in some profiles; also seen as dynamic)
    if pt == 11 and len(payload) % 2 == 0:
        return payload
    return b""


def build_rtp_packet(
    seq: int,
    timestamp: int,
    ssrc: int,
    payload_alaw: bytes,
    *,
    marker: bool = False,
    pt: int = 8,
) -> bytes:
    """One RTP packet for G.711 A-law (PT 8) or PCMU (PT 0)."""
    b0 = 0x80
    b1 = ((1 if marker else 0) << 7) | (pt & 0x7F)
    return struct.pack("!BBHII", b0, b1, seq & 0xFFFF, timestamp & 0xFFFFFFFF, ssrc & 0xFFFFFFFF) + payload_alaw


def pcm16_le_to_alaw(pcm: bytes) -> bytes:
    return audioop.lin2alaw(pcm, 2)


def pcm16_le_to_ulaw(pcm: bytes) -> bytes:
    return audioop.lin2ulaw(pcm, 2)


def random_ssrc() -> int:
    return random.randint(1, 0xFFFFFFFF)
