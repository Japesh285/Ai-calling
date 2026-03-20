"""
FreeSWITCH ESL (Event Socket Library) client.

Replaces fs_cli subprocess calls with a persistent TCP connection to
mod_event_socket on 127.0.0.1:8021.  This eliminates the ~200 ms per-command
subprocess spawn overhead, reducing first-audio latency by ~600 ms for a
typical 3-chunk AI response.

Architecture
------------
A small thread-safe connection pool (default size = 3) is kept open for the
lifetime of the process.  Each call to ``esl_api()`` checks out one connection,
sends the command, reads the response, and returns the connection to the pool.

If the pool is exhausted a fresh short-lived connection is opened (never blocks
the caller).  Stale / broken connections are detected on the next use and
replaced transparently.

Usage
-----
    from app.clients.esl_client import esl_api

    esl_api(f"uuid_broadcast {uuid} {path} aleg")
    esl_api(f"uuid_break {uuid}")
"""

import logging
import queue
import socket
import threading
from typing import Optional

logger = logging.getLogger(__name__)

# ── Configuration ────────────────────────────────────────────────────────────
_ESL_HOST     = "127.0.0.1"
_ESL_PORT     = 8021
_ESL_PASSWORD = "ClueCon"
_POOL_SIZE    = 3          # number of persistent connections kept open
_CMD_TIMEOUT  = 5.0        # seconds to wait for a command response
_CONN_TIMEOUT = 2.0        # seconds to wait while connecting


# ── Low-level connection helpers ─────────────────────────────────────────────

class _ESLConnection:
    """A single persistent ESL TCP connection."""

    def __init__(self) -> None:
        self._sock: Optional[socket.socket] = None
        self._lock = threading.Lock()

    def _connect(self) -> None:
        """Open socket and authenticate.  Raises on failure."""
        s = socket.create_connection((_ESL_HOST, _ESL_PORT), timeout=_CONN_TIMEOUT)
        s.settimeout(_CMD_TIMEOUT)

        # FreeSWITCH sends "Content-Type: auth/request\n\n" on connect
        _recv_response(s)

        # Authenticate
        s.sendall(b"auth " + _ESL_PASSWORD.encode() + b"\n\n")
        reply = _recv_response(s)
        if "+OK accepted" not in reply:
            s.close()
            raise ConnectionError(f"ESL auth rejected: {reply!r}")

        self._sock = s

    def send(self, command: str) -> str:
        """Send an API command and return the response body."""
        with self._lock:
            # Lazy connect / reconnect on stale socket
            if self._sock is None:
                self._connect()

            try:
                self._sock.sendall(f"api {command}\n\n".encode())
                return _recv_response(self._sock)
            except OSError:
                # Socket died — close and reconnect once
                try:
                    self._sock.close()
                except Exception:
                    pass
                self._sock = None
                self._connect()
                self._sock.sendall(f"api {command}\n\n".encode())
                return _recv_response(self._sock)

    def close(self) -> None:
        with self._lock:
            if self._sock:
                try:
                    self._sock.close()
                except Exception:
                    pass
                self._sock = None


def _recv_response(sock: socket.socket) -> str:
    """
    Read one ESL response from *sock* and return the full text
    (headers + body) as a string.

    ESL responses have the form:
        Content-Type: <type>\\n
        [Content-Length: N\\n]
        \\n
        [body of N bytes]

    We read headers until the blank line, then read the body if
    Content-Length is present.  The complete text is returned so callers
    can inspect both header fields (e.g. Reply-Text: +OK) and the body.
    """
    raw = b""
    while b"\n\n" not in raw:
        chunk = sock.recv(4096)
        if not chunk:
            raise ConnectionError("ESL socket closed by remote")
        raw += chunk

    header_part, _, body_start = raw.partition(b"\n\n")
    headers = header_part.decode(errors="replace")

    # Parse Content-Length
    content_length = 0
    for line in headers.splitlines():
        if line.lower().startswith("content-length:"):
            try:
                content_length = int(line.split(":", 1)[1].strip())
            except ValueError:
                pass

    # Read remaining body bytes
    body = body_start
    while len(body) < content_length:
        chunk = sock.recv(content_length - len(body))
        if not chunk:
            raise ConnectionError("ESL socket closed while reading body")
        body += chunk

    # Return full message so callers can inspect headers (Reply-Text etc.)
    return headers + "\n\n" + body.decode(errors="replace")


# ── Connection pool ───────────────────────────────────────────────────────────

class _ESLPool:
    """Thread-safe pool of persistent ESL connections."""

    def __init__(self, size: int = _POOL_SIZE) -> None:
        self._pool: queue.Queue[_ESLConnection] = queue.Queue()
        for _ in range(size):
            conn = _ESLConnection()
            try:
                conn._connect()
            except Exception as exc:
                logger.warning("ESL pool: initial connection failed: %s", exc)
            self._pool.put(conn)

    def api(self, command: str) -> str:
        """
        Execute *command* via a pooled connection.

        If all pool connections are busy a fresh temporary connection is used
        so the caller never blocks.
        """
        try:
            conn = self._pool.get_nowait()
        except queue.Empty:
            # Pool exhausted — open a one-shot connection
            logger.debug("ESL pool exhausted, using temporary connection")
            tmp = _ESLConnection()
            return tmp.send(command)

        try:
            response = conn.send(command)
            return response
        except Exception as exc:
            logger.warning("ESL command failed: %s — %s", command, exc)
            raise
        finally:
            self._pool.put(conn)


# ── Module-level singleton ────────────────────────────────────────────────────

_pool: Optional[_ESLPool] = None
_pool_lock = threading.Lock()


def _get_pool() -> _ESLPool:
    global _pool
    if _pool is None:
        with _pool_lock:
            if _pool is None:
                _pool = _ESLPool()
    return _pool


def esl_api(command: str, timeout: float = _CMD_TIMEOUT) -> str:
    """
    Send a FreeSWITCH API command via ESL and return the response string.

    Falls back to an empty string (never raises) so callers don't need
    try/except for non-critical playback commands.

    Examples
    --------
        esl_api(f"uuid_broadcast {uuid} /tmp/reply.wav aleg")
        esl_api(f"uuid_break {uuid}")
    """
    try:
        return _get_pool().api(command)
    except Exception as exc:
        logger.error("esl_api(%r) failed: %s", command, exc)
        return ""
