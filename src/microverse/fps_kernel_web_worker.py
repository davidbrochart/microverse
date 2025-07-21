from uuid import uuid4

import pyjs
from anyio import TASK_STATUS_IGNORED, Event, create_task_group
from anyio.abc import TaskStatus
from jupyverse_api.kernel import Kernel as foo__Kernel


class KernelWebWorker(foo__Kernel):
    def __init__(self, *args, **kwargs):
        super().__init__()

    async def start(self, *, task_status: TaskStatus[None] = TASK_STATUS_IGNORED) -> None:
        async with (
            self._to_shell_send_stream,
            self._to_shell_receive_stream,
            self._from_shell_send_stream,
            self._from_shell_receive_stream,
            self._to_control_send_stream,
            self._to_control_receive_stream,
            self._from_control_send_stream,
            self._from_control_receive_stream,
            self._to_stdin_send_stream,
            self._to_stdin_receive_stream,
            self._from_stdin_send_stream,
            self._from_stdin_receive_stream,
            self._from_iopub_send_stream,
            self._from_iopub_receive_stream,
            create_task_group() as self.task_group,
        ):
            self.kernel_id = uuid4().hex
            kernel_ready = Event()

            def callback(msg):
                if "msg" in msg:
                    print(f"callback {msg['msg']=}")
                if msg["type"] == "started":
                    kernel_ready.set()
                elif msg["type"] == "shell":
                    msg = [bytes(pyjs.to_py(m)) for m in msg["msg"]]
                    print(f"kernel received shell {msg=}")
                    self.task_group.start_soon(self._from_shell_send_stream.send, msg)
                elif msg["type"] == "control":
                    msg = [bytes(pyjs.to_py(m)) for m in msg["msg"]]
                    print(f"kernel received control {msg=}")
                    self.task_group.start_soon(self._from_control_send_stream.send, msg)
                elif msg["type"] == "stdin":
                    msg = [bytes(pyjs.to_py(m)) for m in msg["msg"]]
                    print(f"kernel received stdin {msg=}")
                    self.task_group.start_soon(self._from_stdin_send_stream.send, msg)
                elif msg["type"] == "iopub":
                    msg = [bytes(pyjs.to_py(m)) for m in msg["msg"]]
                    print(f"kernel received iopub {msg=}")
                    self.task_group.start_soon(self._from_iopub_send_stream.send, msg)

            js_callable, self.js_py_object = pyjs.create_callable(callback)
            higher_order_function = pyjs.js.Function("callback", "action", "kernel_id", "kernel_web_worker(action, kernel_id, 0, callback);")
            #higher_order_function = pyjs.js.Function("cb", "kernel_web_worker({action: 'start', kernel_id: '" + self.kernel_id + "', callback: cb});")
            higher_order_function(js_callable, "start", self.kernel_id)
            await kernel_ready.wait()
            print("kernel ready!")

            self.task_group.start_soon(self.forward_messages_to_shell)
            self.task_group.start_soon(self.forward_messages_to_control)
            self.task_group.start_soon(self.forward_messages_to_stdin)

            task_status.started()

    async def stop(self) -> None:
        self.js_py_object.delete()
        self.task_group.cancel_scope.cancel()

    async def interrupt(self) -> None:
        pass

    async def forward_messages_to_shell(self) -> None:
        async for msg in self._to_shell_receive_stream:
            print(f"{msg=}")
            #msg = (b"<|>".join(msg)).decode()  # FIXME: binary buffers
            try:
                #pyjs.js.Function("kernel_web_worker({action: 'shell', kernel_id: '" + self.kernel_id + "', msg: '" + msg + "'});")()
                msg = pyjs.to_js(msg)
                pyjs.js.Function("action", "kernel_id", "msg", "kernel_web_worker(action, kernel_id, msg, 0);")("shell", self.kernel_id, msg)
                #msg.delete()
            except BaseException as e:
                print(f"{e=}")

    async def forward_messages_to_control(self) -> None:
        async for msg in self._to_control_receive_stream:
            msg = (b"<|>".join(msg)).decode()  # FIXME: binary buffers
            try:
                pyjs.js.Function("kernel_web_worker({action: 'control', kernel_id: '" + self.kernel_id + "', msg: '" + msg + "'});")()
            except BaseException as e:
                print(f"{e=}")

    async def forward_messages_to_stdin(self) -> None:
        async for msg in self._to_stdin_receive_stream:
            msg = (b"<|>".join(msg)).decode()  # FIXME: binary buffers
            try:
                pyjs.js.Function("kernel_web_worker({action: 'stdin', kernel_id: '" + self.kernel_id + "', msg: '" + msg + "'});")()
            except BaseException as e:
                print(f"{e=}")


from fps import Module

from jupyverse_api.kernel import DefaultKernelFactory


class KernelWebWorkerModule(Module):
    async def prepare(self) -> None:
        default_kernel_factory = DefaultKernelFactory(KernelWebWorker)
        self.put(default_kernel_factory)
