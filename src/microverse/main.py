import json

import pyjs
from asyncio import Event, Queue, create_task
from contextlib import AsyncExitStack
from uuid import uuid4

import httpx
from anyio import create_task_group
from fps import Module, get_root_module, initialize
from jupyverse_api.app import App
from httpx_ws import aconnect_ws

ASGIWEBSOCKETTRANSPORT

async def run_sync(callable, *args):
    return callable(*args)

import anyio.to_thread
anyio.to_thread.run_sync = run_sync

server_ready = Event()

async def wait_server_ready():
    await server_ready.wait()

class Client:
    def __init__(self, root_module):
        self._root_module = root_module
        self._websockets = {}

    async def __aenter__(self):
        transport = ASGIWebSocketTransport(app=self._root_module.app)
        async with AsyncExitStack() as stack:
            self._client = await stack.enter_async_context(httpx.AsyncClient(transport=transport, base_url="http://testserver"))
            self._exit_stack = stack.pop_all()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return await self._exit_stack(exc_type, exc_val, exc_tb)

    async def create_websocket(self, url):
        idx = uuid4().hex
        status = Queue()
        self._websockets[idx] = {
            "task": create_task(self._create_websocket(url, idx)),
            "status": status,
        };
        val = await status.get()
        if val == "error":
            print(f"websocket failed {url=}")
            return "error"
        return idx

    async def _create_websocket(self, url, idx):
        try:
            async with aconnect_ws(url, self._client, keepalive_ping_interval_seconds=None) as ws:
                self._websockets[idx]["ws"] = ws
                self._websockets[idx]["status"].put_nowait("ok")
                await Event().wait()
        except BaseException as e:
            self._websockets[idx]["status"].put_nowait("error")
            print(f"{e=} {parsed_url.path=}")
            import traceback
            print(traceback.format_exc())

    async def send_websocket(self, idx, data):
        try:
            data = bytes(pyjs.to_py(data))
            await self._websockets[idx]["ws"].send_json(json.loads(data))
        except BaseException as e:
            print(f"send_websocket {e=}")

    async def receive_websocket(self, idx):
        try:
            msg = await self._websockets[idx]["ws"].receive_json()
            return json.dumps(msg)
        except BaseException as e:
            print(f"receive_websocket {e=}")

    async def send_request(self, method, url, body, headers):
        if body is not None:
            body = bytes(pyjs.to_py(body))
        headers = json.loads(headers)
        if method == "GET":
            response = await self._client.get(url[len("http://127.0.0.1:8000"):], headers=headers)
        elif method == "POST":
            response = await self._client.post(url[len("http://127.0.0.1:8000"):], headers=headers, data=body)
        elif method == "PUT":
            response = await self._client.put(url[len("http://127.0.0.1:8000"):], headers=headers, data=body)
        elif method == "PATCH":
            response = await self._client.patch(url[len("http://127.0.0.1:8000"):], headers=headers, data=body)
        body = None
        try:
            body = response.json()
        except Exception as exception:
            try:
                body = response.text
            except Exception as exception:
                print(f"{exception=}")
        return json.dumps({"status": response.status_code, "body": body, "headers": dict(response.headers)})


async def main():
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
                        "type": "auth",
                    },
                    "contents": {
                        "type": "contents",
                    },
                    "frontend": {
                        "type": "frontend",
                        "config": {
                            "base_url": "/microverse/",
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
                },
            },
        }
        root_module = get_root_module(config)
        initialize(root_module)
        async with (
            root_module,
            Client(root_module) as client,
        ):
            server_ready.set()
            await Event().wait()
    except BaseException as exception:
        print(f"{exception=}")

main_task = create_task(main())
