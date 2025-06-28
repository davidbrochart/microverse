import json

from asyncio import Event, Queue, create_task, sleep

import httpx
import pyjs
from fps import get_root_module, initialize

async def run_sync(callable, *args):
    return callable(*args)

import anyio.to_thread
anyio.to_thread.run_sync = run_sync

server_ready = Event()

async def wait_server_ready():
    await server_ready.wait()

class Client:
    def __init__(self, app):
        transport = httpx.ASGITransport(app=app)
        self._client = httpx.AsyncClient(transport=transport, base_url="http://testserver")

    async def send_request(self, request):
        request_body = request["body"]
        if request_body in ("null", ""):
            request_body = None
        else:
            request_body = bytes([int(bstring) for bstring in request_body.split(",")])
        request_headers = json.loads(request["headers"])
        if request["method"] == "GET":
            response = await self._client.get(request["url"][len("http://127.0.0.1:8000"):], headers=request_headers)
        elif request["method"] == "POST":
            response = await self._client.post(request["url"][len("http://127.0.0.1:8000"):], headers=request_headers, data=request_body)
        elif request["method"] == "PUT":
            response = await self._client.put(request["url"][len("http://127.0.0.1:8000"):], headers=request_headers, data=request_body)
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
        async with root_module:
            client = Client(root_module.app)
            server_ready.set()
            await Event().wait()
    except BaseException as exception:
        print(f"{exception=}")

main_task = create_task(main())
