import json
import logging

import pyjs
from asyncio import Event, Queue, create_task
from contextlib import AsyncExitStack
from uuid import uuid4

import httpx2
import wsproto
from fps import Module, get_root_module, initialize
from jupyverse_api.app import App
from httpx2.websockets import ASGIWebSocketTransport, WebSocketDisconnect

logger = logging.getLogger("microverse")
logging.getLogger("httpx2").setLevel(logging.WARNING)


async def run_sync(callable, *args, **kwargs):
    return callable(*args)

import anyio.to_thread
anyio.to_thread.run_sync = run_sync

async def sleep_forever():
    # PyJS schedules sleeps with JavaScript setTimeout, where an infinite
    # delay fires immediately. Keep background services alive without a timer.
    await Event().wait()

anyio.sleep_forever = sleep_forever

server_ready = Event()

async def wait_server_ready():
    await server_ready.wait()

class Client:
    def __init__(self, root_module):
        self._root_module = root_module
        self._websockets = {}

    async def __aenter__(self):
        transport = ASGIWebSocketTransport(app=self._call_app)
        async with AsyncExitStack() as stack:
            self._client = await stack.enter_async_context(httpx2.AsyncClient(transport=transport, base_url="http://testserver"))
            self._exit_stack = stack.pop_all()
        return self

    async def _call_app(self, scope, receive, send):
        try:
            await self._root_module.app(scope, receive, send)
        except Exception:
            logger.exception("ASGI application failed: %s %s", scope["type"], scope["path"])
            raise

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return await self._exit_stack.__aexit__(exc_type, exc_val, exc_tb)

    async def open_websocket(self, url):
        idx = uuid4().hex
        status = Queue()
        self._websockets[idx] = {
            "task": create_task(self._open_websocket(url, idx)),
            "status": status,
            "close": Event(),
        };
        val = await status.get()
        if val == "error":
            return
        return idx

    async def _open_websocket(self, url, idx):
        socket = self._websockets[idx]
        try:
            async with self._client.websocket(url, keepalive_ping_interval_seconds=None) as ws:
                socket["ws"] = ws
                socket["status"].put_nowait("ok")
                await socket["close"].wait()
        except BaseException as exception:
            if (
                url.split("?", 1)[0] == "/microverse/api/events/subscribe"
                and isinstance(exception, WebSocketDisconnect)
                and exception.code == 1000
            ):
                logger.debug("Events subscription is unavailable")
            else:
                logger.exception("WebSocket connection failed: %s", url)
            socket["status"].put_nowait("error")
        finally:
            socket["close"].set()
            self._websockets.pop(idx, None)

    async def close_websocket(self, idx):
        socket = self._websockets.get(idx)
        if socket is not None:
            socket["close"].set()
        return "ok"

    async def send_websocket(self, idx, data, binary):
        socket = self._websockets.get(idx)
        if socket is None or socket["close"].is_set():
            return
        try:
            data = bytes(pyjs.to_py(data))
            if binary:
                await socket["ws"].send_bytes(data)
            else:
                data = data.decode()
                await socket["ws"].send_text(data)
        except Exception as e:
            if not socket["close"].is_set():
                logger.exception("WebSocket send failed: %s", idx)

    async def receive_websocket(self, idx):
        socket = self._websockets.get(idx)
        if socket is None or socket["close"].is_set():
            return
        try:
            event = await socket["ws"].receive(10)
            if isinstance(event, wsproto.events.TextMessage):
                data = bytes([1]) + event.data.encode()
            else:
                data = bytes([0]) + bytes(event.data)
            return pyjs.buffer_to_js_typed_array(data)
        except TimeoutError:
            return pyjs.buffer_to_js_typed_array(b"")
        except Exception as e:
            if socket["close"].is_set():
                return
            logger.exception("WebSocket receive failed: %s", idx)

    async def send_request(self, method, url, body, headers):
        if body is not None:
            body = bytes(pyjs.to_py(body))
        headers = json.loads(headers)
        if method == "GET":
            response = await self._client.get(url, headers=headers)
        elif method == "POST":
            response = await self._client.post(url, headers=headers, data=body)
        elif method == "PUT":
            response = await self._client.put(url, headers=headers, data=body)
        elif method == "PATCH":
            response = await self._client.patch(url, headers=headers, data=body)
        elif method == "DELETE":
            response = await self._client.delete(url, headers=headers)
        body = None
        try:
            body = response.json()
        except Exception as exception:
            try:
                body = response.text
            except Exception as exception:
                logger.exception("Could not decode response: %s", url)
        return json.dumps({"status": response.status_code, "body": body, "headers": dict(response.headers)})


async def main(base_url):
    global client

    try:
        config = {
            "jupyverse": {
                "type": "jupyverse",
                "config": {
                    "start_server": False,
                },
                "modules": {
                    "app": {
                        "type": "app",
                        "config": {
                            "mount_path": "/microverse",
                        }
                    },
                    "auth": {
                        "type": "noauth",
                    },
                    "contents": {
                        "type": "contents",
                    },
                    "frontend": {
                        "type": "frontend",
                        "config": {
                            "base_url": f"{base_url}microverse/",
                            "collaborative": True,
                        }
                    },
                    "kernel_web_worker": {
                        "type": "kernel_web_worker",
                    },
                    "kernels": {
                        "type": "kernels",
                    },
                    "lab": {
                        "type": "lab",
                    },
                    "jupyterlab": {
                        "type": "jupyterlab",
                    },
                    "file_watcher_poll": {
                        "type": "file_watcher_poll",
                    },
                    "file_id": {
                        "type": "file_id",
                    },
                    "yjs": {
                        "type": "yjs",
                    },
                    "yrooms": {
                        "type": "yrooms",
                    },
                    "ystore_sqlite": {
                        "type": "ystore_sqlite",
                    },
                    "page_config": {
                        "type": "page_config",
                    },
                },
            },
        }
        root_module = get_root_module(config)
        root_module._global_start_timeout = 10
        initialize(root_module)
        async with (
            root_module,
            Client(root_module) as client,
        ):
            server_ready.set()
            await Event().wait()
    except BaseException as exception:
        logger.exception("Microverse server failed")
