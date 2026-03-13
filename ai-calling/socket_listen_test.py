import asyncio

import websockets


async def handler(websocket):
    remote = getattr(websocket, "remote_address", None)
    path = getattr(websocket, "path", "")
    print(f"FreeSWITCH connected from {remote} path={path!r}")

    frame_count = 0
    total_bytes = 0

    try:
        async for message in websocket:
            if isinstance(message, bytes):
                frame_count += 1
                total_bytes += len(message)

                if frame_count <= 5 or frame_count % 50 == 0:
                    print(
                        f"Audio frame {frame_count}: {len(message)} bytes "
                        f"(total={total_bytes})"
                    )
            else:
                print("Text message:", message)
    except websockets.ConnectionClosed as exc:
        print(f"Connection closed: code={exc.code} reason={exc.reason!r}")
    finally:
        print(
            f"Stream ended from {remote}: frames={frame_count} total_bytes={total_bytes}"
        )


async def main():
    async with websockets.serve(handler, "0.0.0.0", 8765, max_size=None):
        print("WebSocket server listening on ws://0.0.0.0:8765")
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
