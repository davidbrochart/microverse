import json
import pyjs
from asyncio import Queue, create_task

ASYNCTESTCLIENT

from fastapi import FastAPI

app = FastAPI()

@app.get("/")
async def get():
    return {"Hello": "World!"}

client = AsyncTestClient(app)

queue_to_py = Queue()
queue_to_js = Queue()

def send_to_py(msg):
    queue_to_py.put_nowait(msg)

async def main():
    try:
        while True:
            request = await queue_to_py.get()
            print(f"{request=}")
            response = await client.get("/")
            if response.status_code == 200:
                await queue_to_js.put(json.dumps(response.json()))
            else:
                await queue_to_js.put(f"{response.status_code=}")
    except BaseException as exception:
        pyjs.js.console.log(f"{exception=}")

main_task = create_task(main())

async def receive_from_py():
    msg = await queue_to_js.get()
    return msg

