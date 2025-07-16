from jupyverse_api.kernel import DefaultKernelFactory, Kernel
from fps_akernel_task.akernel_task import AKernelTask


class FakeKernel(Kernel):
    def __init__(self, *args, **kwargs):
        super().__init__()

    async def start(self, *, task_status):
        try:
            async with (
                create_task_group() as tg,
                self._to_shell_send_stream,
                self._to_shell_receive_stream,
                self._to_control_send_stream,
                self._to_control_receive_stream,
                self._to_stdin_send_stream,
                self._to_stdin_receive_stream,
                self._from_shell_send_stream,
                self._from_shell_receive_stream,
                self._from_control_send_stream,
                self._from_control_receive_stream,
                self._from_stdin_send_stream,
                self._from_stdin_receive_stream,
                self._from_iopub_send_stream,
                self._from_iopub_receive_stream,
            ):
                tg.start_soon(self.receive)
                task_status.started()
                await self._from_iopub_send_stream.send([b"<IDS|MSG>", b"0", b'{"msg_id": "foo", "msg_type": "bar"}', b'{"session": "foo"}', b'{"metadata": 0}', b'{"content": 0}'])
        except BaseException as exc:
            print(f"{exc=}")
            import traceback
            print(traceback.format_exc())

    async def receive(self):
        try:
            async for msg in self._to_shell_receive_stream:
                print(f"received shell stream {msg=}")
        except BaseException as exc:
            print(f"{exc=}")
            import traceback
            print(traceback.format_exc())

    async def stop(self): pass
    async def interrupt(self): pass


class FakeKernelModule(Module):
    async def prepare(self):
        #default_kernel_factory = DefaultKernelFactory(FakeKernel)
        default_kernel_factory = DefaultKernelFactory(AKernelTask)
        self.put(default_kernel_factory, DefaultKernelFactory)

