# file: kernel_driver/message.py

import hashlib
import hmac
import json
from datetime import datetime, timezone
from typing import Any, cast
from uuid import uuid4

from dateutil.parser import parse as dateutil_parse

protocol_version_info = (5, 3)
protocol_version = ".".join(map(str, protocol_version_info))

DELIM = b"<IDS|MSG>"


def feed_identities(msg_list: list[bytes]) -> tuple[list[bytes], list[bytes]]:
    idx = msg_list.index(DELIM)
    return msg_list[:idx], msg_list[idx + 1 :]  # noqa


def str_to_date(obj: dict[str, Any]) -> dict[str, Any]:
    if "date" in obj:
        obj["date"] = dateutil_parse(obj["date"])
    return obj


def date_to_str(obj: dict[str, Any]):
    if "date" in obj and not isinstance(obj["date"], str):
        obj["date"] = obj["date"].isoformat().replace("+00:00", "Z")
    return obj


def utcnow() -> datetime:
    return datetime.now(tz=timezone.utc)


def create_message_header(msg_type: str, session_id: str, msg_id: str) -> dict[str, Any]:
    if not session_id:
        session_id = msg_id = uuid4().hex
    else:
        msg_id = f"{session_id}_{msg_id}"
    header = {
        "date": utcnow().isoformat().replace("+00:00", "Z"),
        "msg_id": msg_id,
        "msg_type": msg_type,
        "session": session_id,
        "username": "",
        "version": protocol_version,
    }
    return header


def create_message(
    msg_type: str,
    content: dict = {},
    session_id: str = "",
    msg_id: str = "",
    buffers: list = [],
) -> dict[str, Any]:
    header = create_message_header(msg_type, session_id, msg_id)
    msg = {
        "header": header,
        "msg_id": header["msg_id"],
        "msg_type": header["msg_type"],
        "parent_header": {},
        "content": content,
        "metadata": {},
        "buffers": buffers,
    }
    return msg


def dumps(o: Any, **kwargs) -> bytes:
    return json.dumps(o, **kwargs).encode("utf8")


def loads(s: bytes | str, **kwargs) -> dict | list | str | int | float:
    if isinstance(s, bytes):
        s = s.decode("utf8")
    return json.loads(s, **kwargs)


def pack(obj: dict[str, Any]) -> bytes:
    return dumps(obj)


def unpack(s: bytes) -> dict[str, Any]:
    return cast(dict[str, Any], loads(s))


def sign(msg_list: list[bytes], key: str) -> bytes:
    auth = hmac.new(key.encode("ascii"), digestmod=hashlib.sha256)
    h = auth.copy()
    for m in msg_list:
        h.update(m)
    return h.hexdigest().encode()


def serialize_message(
    msg: dict[str, Any], key: str, change_date_to_str: bool = False
) -> list[bytes]:
    _date_to_str = date_to_str if change_date_to_str else lambda x: x
    message = [
        pack(_date_to_str(msg["header"])),
        pack(_date_to_str(msg["parent_header"])),
        pack(_date_to_str(msg["metadata"])),
        pack(_date_to_str(msg.get("content", {}))),
    ]
    to_send = [DELIM, sign(message, key)] + message + msg.get("buffers", [])
    return to_send


def deserialize_message(
    msg_list: list[bytes],
    parent_header: dict[str, Any] | None = None,
    change_str_to_date: bool = False,
) -> dict[str, Any]:
    _str_to_date = str_to_date if change_str_to_date else lambda x: x
    message: dict[str, Any] = {}
    header = unpack(msg_list[1])
    message["header"] = _str_to_date(header)
    message["msg_id"] = header["msg_id"]
    message["msg_type"] = header["msg_type"]
    if parent_header:
        message["parent_header"] = parent_header
    else:
        message["parent_header"] = _str_to_date(unpack(msg_list[2]))
    message["metadata"] = unpack(msg_list[3])
    message["content"] = unpack(msg_list[4])
    message["buffers"] = [memoryview(b) for b in msg_list[5:]]
    return message

# file kernel_driver/path.py

import glob
import os
import sys
import tempfile
import uuid


def _expand_path(s):
    s = os.path.expandvars(os.path.expanduser(s))
    return s


def _filefind(filename, path_dirs=()):
    filename = filename.strip('"').strip("'")
    if os.path.isabs(filename) and os.path.isfile(filename):
        return filename

    path_dirs = path_dirs or ("",)

    for path in path_dirs:
        if path == ".":
            path = os.getcwd()
        testname = _expand_path(os.path.join(path, filename))
        if os.path.isfile(testname):
            return os.path.abspath(testname)

    raise OSError(f"File {filename} does not exist in any of the search paths: {path_dirs}")


def get_home_dir():
    home = os.path.expanduser("~")
    home = os.path.realpath(home)
    return home


_dtemps: dict = {}


def _mkdtemp_once(name):
    if name in _dtemps:
        return _dtemps[name]
    d = _dtemps[name] = tempfile.mkdtemp(prefix=name + "-")
    return d


def jupyter_config_dir():
    if os.environ.get("JUPYTER_NO_CONFIG"):
        return _mkdtemp_once("jupyter-clean-cfg")
    if "JUPYTER_CONFIG_DIR" in os.environ:
        return os.environ.env["JUPYTER_CONFIG_DIR"]
    home = get_home_dir()
    return os.path.join(home, ".jupyter")


def jupyter_data_dir():
    if "JUPYTER_DATA_DIR" in os.environ:
        return os.environ["JUPYTER_DATA_DIR"]

    home = get_home_dir()

    if sys.platform == "darwin":
        return os.path.join(home, "Library", "Jupyter")
    elif os.name == "nt":
        appdata = os.environ.get("APPDATA", None)
        if appdata:
            return os.path.join(appdata, "jupyter")
        else:
            return os.path.join(jupyter_config_dir(), "data")
    else:
        xdg = os.environ.get("XDG_DATA_HOME", None)
        if not xdg:
            xdg = os.path.join(home, ".local", "share")
        return os.path.join(xdg, "jupyter")


def jupyter_runtime_dir():
    if "JUPYTER_RUNTIME_DIR" in os.environ:
        return os.environ("JUPYTER_RUNTIME_DIR")
    return os.path.join(jupyter_data_dir(), "runtime")


def find_connection_file(
    filename: str = "kernel-*.json",
    paths: list[str] = [],
) -> str:
    if not paths:
        paths = [".", jupyter_runtime_dir()]

    path = _filefind(filename, paths)
    if path:
        return path

    if "*" in filename:
        pat = filename
    else:
        pat = f"*{filename}*"

    matches = []
    for p in paths:
        matches.extend(glob.glob(os.path.join(p, pat)))

    matches = [os.path.abspath(m) for m in matches]
    if not matches:
        raise OSError(f"Could not find {filename} in {paths}")
    elif len(matches) == 1:
        return matches[0]
    else:
        return sorted(matches, key=lambda f: os.stat(f).st_atime)[-1]

# file kernel_driver/kernelspec.py

import os
import sys

if os.name == "nt":
    programdata = os.environ.get("PROGRAMDATA", None)
    if programdata:
        SYSTEM_JUPYTER_PATH = [os.path.join(programdata, "jupyter")]
    else:  # PROGRAMDATA is not defined by default on XP
        SYSTEM_JUPYTER_PATH = [os.path.join(sys.prefix, "share", "jupyter")]
else:
    SYSTEM_JUPYTER_PATH = [
        "/usr/local/share/jupyter",
        "/usr/share/jupyter",
    ]

ENV_JUPYTER_PATH = [os.path.join(sys.prefix, "share", "jupyter")]


def jupyter_path(*subdirs):
    paths = []
    # highest priority is env
    if os.environ.get("JUPYTER_PATH"):
        paths.extend(p.rstrip(os.sep) for p in os.environ["JUPYTER_PATH"].split(os.pathsep))
    # then user dir
    paths.append(jupyter_data_dir())
    # then sys.prefix
    for p in ENV_JUPYTER_PATH:
        if p not in SYSTEM_JUPYTER_PATH:
            paths.append(p)
    # finally, system
    paths.extend(SYSTEM_JUPYTER_PATH)

    # add subdir, if requested
    if subdirs:
        paths = [os.path.join(p, *subdirs) for p in paths]
    return paths


def kernelspec_dirs():
    return jupyter_path("kernels")


def _is_kernel_dir(path):
    return os.path.isdir(path) and os.path.isfile(os.path.join(path, "kernel.json"))


def _list_kernels_in(kernel_dir):
    if kernel_dir is None or not os.path.isdir(kernel_dir):
        return {}
    kernels = {}
    for f in os.listdir(kernel_dir):
        path = os.path.join(kernel_dir, f)
        if _is_kernel_dir(path):
            key = f.lower()
            kernels[key] = path
    return kernels


def find_kernelspec(kernel_name):
    d = {}
    for kernel_dir in kernelspec_dirs():
        kernels = _list_kernels_in(kernel_dir)
        for kname, spec in kernels.items():
            if kname not in d:
                d[kname] = os.path.join(spec, "kernel.json")
    return d.get(kernel_name, "")

# file: kernel_driver/driver.py

import time
import uuid
from typing import Any

from anyio import (
    TASK_STATUS_IGNORED,
    Event,
    create_memory_object_stream,
    create_task_group,
    fail_after,
    move_on_after,
)
from anyio.abc import TaskGroup, TaskStatus
from anyio.streams.stapled import StapledObjectStream
from pycrdt import Array, Map, Text

from jupyverse_api.kernel import DefaultKernelFactory, Kernel
from jupyverse_api.yjs import Yjs


def deadline_to_timeout(deadline: float) -> float:
    return max(0, deadline - time.time())


class KernelDriver:
    task_group: TaskGroup

    def __init__(
        self,
        default_kernel_factory: DefaultKernelFactory,
        kernel_name: str = "",
        kernelspec_path: str = "",
        kernel_cwd: str = "",
        connection_file: str = "",
        write_connection_file: bool = True,
        capture_kernel_output: bool = True,
        yjs: Yjs | None = None,
    ) -> None:
        kernelspec_path = kernelspec_path or find_kernelspec(kernel_name)
        self.yjs = yjs
        if not kernelspec_path:
            raise RuntimeError("Could not find a kernel, maybe you forgot to install one?")
        self.kernel = default_kernel_factory(
            write_connection_file,
            kernelspec_path,
            connection_file,
            kernel_cwd,
            capture_kernel_output,
        )
        self.session_id = uuid.uuid4().hex
        self.msg_cnt = 0
        self.execute_requests: dict[str, dict[str, StapledObjectStream]] = {}
        self.comm_messages: StapledObjectStream = StapledObjectStream(
            *create_memory_object_stream[dict](max_buffer_size=1024)
        )
        self.stopped_event = Event()

    async def restart(self, startup_timeout: float = float("inf")) -> None:
        self.task_group.cancel_scope.cancel()
        await self.stopped_event.wait()
        self.stopped_event = Event()
        async with create_task_group() as tg:
            self.task_group = tg
            msg = create_message("shutdown_request", content={"restart": True})
            msg_ser = serialize_message(msg, self.kernel.key, change_date_to_str=True)
            await self.kernel.control_stream.send(msg_ser)
            while True:
                msg_ser = await self.kernel.control_stream.receive()
                idents, msg_list = feed_identities(msg_ser)
                msg = deserialize_message(msg_list, change_str_to_date=True)
                if msg["msg_type"] == "shutdown_reply" and msg["content"]["restart"]:
                    break
            await self._wait_for_ready(startup_timeout)
            self.listen_channels()
            tg.start_soon(self._handle_comms)

    async def start(
        self,
        startup_timeout: float = float("inf"),
        *,
        task_status: TaskStatus[None] = TASK_STATUS_IGNORED,
    ) -> None:
        async with create_task_group() as tg:
            self.task_group = tg
            await tg.start(self.kernel.start)
            await self._wait_for_ready(startup_timeout)
            self.listen_channels()
            self.task_group.start_soon(self._handle_comms)
            task_status.started()
        self.stopped_event.set()

    def listen_channels(self):
        self.task_group.start_soon(self.listen_iopub)
        self.task_group.start_soon(self.listen_shell)

    async def stop(self) -> None:
        await self.kernel.stop()
        self.task_group.cancel_scope.cancel()

    async def listen_iopub(self):
        while True:
            msg_ser = await self.kernel.iopub_stream.receive()
            idents, msg_list = feed_identities(msg_ser)
            msg = deserialize_message(msg_list, change_str_to_date=True)
            parent_id = msg["parent_header"].get("msg_id")
            if msg["msg_type"] in ("comm_open", "comm_msg"):
                await self.comm_messages.send(msg)
            elif parent_id in self.execute_requests.keys():
                await self.execute_requests[parent_id]["iopub_msg"].send(msg)

    async def listen_shell(self):
        while True:
            msg_ser = await self.kernel.shell_stream.receive()
            idents, msg_list = feed_identities(msg_ser)
            msg = deserialize_message(msg_list, change_str_to_date=True)
            msg_id = msg["parent_header"].get("msg_id")
            if msg_id in self.execute_requests.keys():
                await self.execute_requests[msg_id]["shell_msg"].send(msg)

    async def execute(
        self,
        ycell: Map,
        timeout: float = float("inf"),
        msg_id: str = "",
        wait_for_executed: bool = True,
    ) -> None:
        if ycell["cell_type"] != "code":
            return
        ycell["execution_state"] = "busy"
        content = {"code": str(ycell["source"]), "silent": False}
        msg = create_message(
            "execute_request", content, session_id=self.session_id, msg_id=str(self.msg_cnt)
        )
        if msg_id:
            msg["header"]["msg_id"] = msg_id
        else:
            msg_id = msg["header"]["msg_id"]
        self.msg_cnt += 1
        msg_ser = serialize_message(msg, self.kernel.key, change_date_to_str=True)
        await self.kernel.shell_stream.send(msg_ser)
        self.execute_requests[msg_id] = {
            "iopub_msg": StapledObjectStream(
                *create_memory_object_stream[dict](max_buffer_size=1024)
            ),
            "shell_msg": StapledObjectStream(
                *create_memory_object_stream[dict](max_buffer_size=1024)
            ),
        }
        if wait_for_executed:
            deadline = time.time() + timeout
            while True:
                try:
                    with fail_after(deadline_to_timeout(deadline)):
                        msg = await self.execute_requests[msg_id]["iopub_msg"].receive()
                except TimeoutError:
                    error_message = f"Kernel didn't respond in {timeout} seconds"
                    raise RuntimeError(error_message)
                await self._handle_outputs(ycell["outputs"], msg)
                if (
                    msg["header"]["msg_type"] == "status"
                    and msg["content"]["execution_state"] == "idle"
                ):
                    break
            try:
                with fail_after(deadline_to_timeout(deadline)):
                    msg = await self.execute_requests[msg_id]["shell_msg"].receive()
            except TimeoutError:
                error_message = f"Kernel didn't respond in {timeout} seconds"
                raise RuntimeError(error_message)
            with ycell.doc.transaction():
                ycell["execution_count"] = msg["content"]["execution_count"]
                ycell["execution_state"] = "idle"
            del self.execute_requests[msg_id]
        else:
            self.task_group.start_soon(lambda: self._handle_iopub(msg_id, ycell))

    async def _handle_iopub(self, msg_id: str, ycell: Map) -> None:
        while True:
            msg = await self.execute_requests[msg_id]["iopub_msg"].receive()
            await self._handle_outputs(ycell["outputs"], msg)
            if (
                msg["header"]["msg_type"] == "status"
                and msg["content"]["execution_state"] == "idle"
            ):
                msg = await self.execute_requests[msg_id]["shell_msg"].receive()
                with ycell.doc.transaction():
                    ycell["execution_count"] = msg["content"]["execution_count"]
                    ycell["execution_state"] = "idle"
                break

    async def _handle_comms(self) -> None:
        if self.yjs is None or self.yjs.widgets is None:  # type: ignore
            return

        while True:
            msg = await self.comm_messages.receive()
            msg_type = msg["header"]["msg_type"]
            if msg_type == "comm_open":
                comm_id = msg["content"]["comm_id"]
                comm = Comm(comm_id, self.session_id, self.kernel, self.task_group)
                self.yjs.widgets.comm_open(msg, comm)  # type: ignore
            elif msg_type == "comm_msg":
                self.yjs.widgets.comm_msg(msg)  # type: ignore

    async def _wait_for_ready(self, timeout):
        deadline = time.time() + timeout
        new_timeout = timeout
        while True:
            msg = create_message(
                "kernel_info_request", session_id=self.session_id, msg_id=str(self.msg_cnt)
            )
            self.msg_cnt += 1
            msg_ser = serialize_message(msg, self.kernel.key, change_date_to_str=True)
            await self.kernel.shell_stream.send(msg_ser)
            try:
                with fail_after(new_timeout):
                    msg_ser = await self.kernel.shell_stream.receive()
                    idents, msg_list = feed_identities(msg_ser)
                    msg = deserialize_message(msg_list, change_str_to_date=True)
            except TimeoutError:
                error_message = f"Kernel didn't respond in {timeout} seconds"
                raise RuntimeError(error_message)
            if msg["msg_type"] == "kernel_info_reply":
                with move_on_after(0.2):
                    msg_ser = await self.kernel.iopub_stream.receive()
                    idents, msg_list = feed_identities(msg_ser)
                    msg = deserialize_message(msg_list, change_str_to_date=True)
                    return
            new_timeout = deadline_to_timeout(deadline)

    async def _handle_outputs(self, outputs: Array, msg: dict[str, Any]):
        msg_type = msg["header"]["msg_type"]
        content = msg["content"]
        if msg_type == "stream":
            with outputs.doc.transaction():
                if (not outputs) or (outputs[-1]["name"] != content["name"]):  # type: ignore
                    outputs.append(
                        Map(
                            {
                                "name": content["name"],
                                "output_type": msg_type,
                                "text": Text(content["text"]),
                            }
                        )
                    )
                else:
                    text = outputs[-1]["text"]
                    text += content["text"]  # type: ignore
        elif msg_type in ("display_data", "execute_result"):
            if "application/vnd.jupyter.ywidget-view+json" in content["data"]:
                # this is a collaborative widget
                model_id = content["data"]["application/vnd.jupyter.ywidget-view+json"]["model_id"]
                if self.yjs is not None and self.yjs.widgets is not None:  # type: ignore
                    if model_id in self.yjs.widgets.widgets:  # type: ignore
                        doc = self.yjs.widgets.widgets[model_id]["model"].ydoc  # type: ignore
                        path = f"ywidget:{doc.guid}"
                        await self.yjs.room_manager.websocket_server.get_room(path, ydoc=doc)  # type: ignore
                        outputs.append(doc)
            else:
                output = {
                    "data": content["data"],
                    "metadata": {},
                    "output_type": msg_type,
                }
                if msg_type == "execute_result":
                    output["execution_count"] = content["execution_count"]
                outputs.append(output)
        elif msg_type == "error":
            outputs.append(
                {
                    "ename": content["ename"],
                    "evalue": content["evalue"],
                    "output_type": "error",
                    "traceback": content["traceback"],
                }
            )


class Comm:
    def __init__(self, comm_id: str, session_id: str, kernel: Kernel, task_group: TaskGroup):
        self.comm_id = comm_id
        self.session_id = session_id
        self.kernel = kernel
        self.task_group = task_group
        self.msg_cnt = 0

    def send(self, buffers):
        msg = create_message(
            "comm_msg",
            content={"comm_id": self.comm_id},
            session_id=self.session_id,
            msg_id=self.msg_cnt,
            buffers=buffers,
        )
        self.msg_cnt += 1

        self.task_group.start_soon(self.send_message, msg)

    async def send_message(self, msg):
        msg_ser = serialize_message(msg, self.kernel.key, change_date_to_str=True)
        await self.kernel.shell_stream.send(msg_ser)

# file: kernel_server/message/py

import json
import struct
from typing import Any


def to_binary(msg: dict[str, Any]) -> bytes | None:
    if not msg["buffers"]:
        return None
    buffers = msg.pop("buffers")
    bmsg = json.dumps(msg).encode("utf8")
    buffers.insert(0, bmsg)
    n = len(buffers)
    offsets = [4 * (n + 1)]
    for b in buffers[:-1]:
        offsets.append(offsets[-1] + len(b))
    header = struct.pack("!" + "I" * (n + 1), n, *offsets)
    buffers.insert(0, header)
    return b"".join(buffers)


def from_binary(bmsg: bytes) -> dict[str, Any]:
    n = struct.unpack("!i", bmsg[:4])[0]
    offsets = list(struct.unpack("!" + "I" * n, bmsg[4 : 4 * (n + 1)]))  # noqa
    offsets.append(None)
    buffers = []
    for start, stop in zip(offsets[:-1], offsets[1:]):
        buffers.append(bmsg[start:stop])
    msg = json.loads(buffers[0].decode("utf8"))
    msg["buffers"] = buffers[1:]
    return msg


def deserialize_msg_from_ws_v1(ws_msg: bytes) -> tuple[str, list[bytes]]:
    offset_number = int.from_bytes(ws_msg[:8], "little")
    offsets = [
        int.from_bytes(ws_msg[8 * (i + 1) : 8 * (i + 2)], "little")  # noqa
        for i in range(offset_number)
    ]
    channel = ws_msg[offsets[0] : offsets[1]].decode("utf-8")  # noqa
    msg_list = [ws_msg[offsets[i] : offsets[i + 1]] for i in range(1, offset_number - 1)]  # noqa
    return channel, msg_list


def serialize_msg_to_ws_v1(msg_list: list[bytes], channel: str) -> list[bytes]:
    msg_list = msg_list[1:]
    channel_b = channel.encode("utf-8")
    offsets = []
    offsets.append(8 * (1 + 1 + len(msg_list) + 1))
    offsets.append(len(channel_b) + offsets[-1])
    for msg in msg_list:
        offsets.append(len(msg) + offsets[-1])
    offset_number = len(offsets).to_bytes(8, byteorder="little")
    offsets_b = [offset.to_bytes(8, byteorder="little") for offset in offsets]
    bin_msg = [offset_number] + offsets_b + [channel_b] + msg_list
    return bin_msg


def get_parent_header(parts: list[bytes]) -> dict[str, Any]:
    return unpack(parts[2])

# file: kernel_server/server.py

import json
from collections.abc import Iterable
from datetime import datetime, timezone

from anyio import TASK_STATUS_IGNORED, Event, create_task_group, move_on_after
from anyio.abc import TaskStatus
from fastapi import WebSocket
from starlette.websockets import WebSocketState

from jupyverse_api.kernel import DefaultKernelFactory, KernelFactory

kernels: dict = {}


class AcceptedWebSocket:
    _websocket: WebSocket
    _accepted_subprotocol: str | None

    def __init__(self, websocket, accepted_subprotocol):
        self._websocket = websocket
        self._accepted_subprotocol = accepted_subprotocol

    @property
    def websocket(self):
        return self._websocket

    @property
    def accepted_subprotocol(self):
        return self._accepted_subprotocol


class KernelServer:
    def __init__(
        self,
        default_kernel_factory: DefaultKernelFactory,
        kernelspec_path: str = "",
        kernel_cwd: str = "",
        connection_file: str = "",
        write_connection_file: bool = True,
        capture_kernel_output: bool = True,
    ) -> None:
        self.default_kernel_factory = default_kernel_factory
        self.capture_kernel_output = capture_kernel_output
        self.kernelspec_path = kernelspec_path
        self.kernel_cwd = kernel_cwd
        self.connection_file = connection_file
        self.write_connection_file = write_connection_file
        self.sessions: dict[str, AcceptedWebSocket] = {}
        # blocked messages and allowed messages are mutually exclusive
        self.blocked_messages: list[str] = []
        self.allowed_messages: list[str] | None = None  # when None, all messages are allowed
        # when [], no message is allowed

    def block_messages(self, message_types: Iterable[str] = []):
        # if using blocked messages, discard allowed messages
        self.allowed_messages = None
        if isinstance(message_types, str):
            message_types = [message_types]
        self.blocked_messages = list(message_types)

    def allow_messages(self, message_types: Iterable[str] | str | None = None):
        # if using allowed messages, discard blocked messages
        self.blocked_messages = []
        if message_types is None:
            self.allowed_messages = None
            return
        if isinstance(message_types, str):
            message_types = [message_types]
        self.allowed_messages = list(message_types)

    @property
    def connections(self) -> int:
        return len(self.sessions)

    async def start(
        self,
        launch_kernel: bool = True,
        kernel_factory: KernelFactory | None = None,
        *,
        task_status: TaskStatus[None] = TASK_STATUS_IGNORED,
    ) -> None:
        try:
            async with create_task_group() as self.task_group:
                self.last_activity = {
                    "date": datetime.now(tz=timezone.utc).isoformat().replace("+00:00", "Z"),
                    "execution_state": "starting",
                }
                if launch_kernel:
                    if not self.kernelspec_path:
                        raise RuntimeError("Could not find a kernel, maybe you forgot to install one?")
                    if kernel_factory is None:
                        self.kernel = self.default_kernel_factory(
                            write_connection_file=self.write_connection_file,
                            kernelspec_path=self.kernelspec_path,
                            connection_file=self.connection_file,
                            kernel_cwd=self.kernel_cwd,
                            capture_output=self.capture_kernel_output,
                        )
                    else:
                        self.kernel = kernel_factory(
                            kernelspec_path=self.kernelspec_path,
                            connection_file=self.connection_file,
                            kernel_cwd=self.kernel_cwd,
                            capture_output=self.capture_kernel_output,
                        )
                    await self.task_group.start(self.kernel.start)
                task_status.started()
                if self.kernel.wait_for_ready:
                    await self._wait_for_ready()
                async with create_task_group() as tg:
                    tg.start_soon(lambda: self.listen("shell"))
                    tg.start_soon(lambda: self.listen("stdin"))
                    tg.start_soon(lambda: self.listen("control"))
                    tg.start_soon(lambda: self.listen("iopub"))
        except BaseException as e:
            print(f"{e=}")
            import traceback
            print(traceback.format_exc())

    async def stop(self) -> None:
        await self.kernel.stop()
        self.task_group.cancel_scope.cancel()

    async def interrupt(self) -> None:
        await self.kernel.interrupt()

    async def restart(self, *, task_status: TaskStatus[None] = TASK_STATUS_IGNORED) -> None:
        await self.stop()
        await self.start(task_status=task_status)

    async def serve(
        self,
        websocket: AcceptedWebSocket,
        session_id: str,
        permissions: dict[str, list[str]] | None,
    ):
        self.sessions[session_id] = websocket
        self.can_execute = permissions is None or "execute" in permissions.get("kernels", [])
        stop_event = Event()
        self.task_group.start_soon(self.listen_web, websocket, stop_event)
        await stop_event.wait()

        # the session could have been removed through the REST API, so check if it still exists
        if session_id in self.sessions:
            del self.sessions[session_id]

    async def listen_web(self, websocket: AcceptedWebSocket, stop_event: Event):
        try:
            await self.send_to_kernel(websocket)
        except BaseException:
            pass
        finally:
            stop_event.set()

    async def listen(self, channel_name: str):
        channel = {
            "shell": self.kernel.shell_stream,
            "control": self.kernel.control_stream,
            "iopub": self.kernel.iopub_stream,
            "stdin": self.kernel.stdin_stream,
        }[channel_name]

        while True:
            try:
                msg = await channel.receive()
            except:
                return
            print(f"{msg=} {channel_name=}")
            idents, parts = feed_identities(msg)
            print(f"{idents=}, {parts=}")
            parent_header = get_parent_header(parts)
            print(f"{self.sessions=}")
            if channel_name == "iopub":
                # broadcast to all web clients
                websockets = list(self.sessions.values())
                for websocket in websockets:
                    await self.send_to_ws(websocket, parts, parent_header, channel_name)
            else:
                session = parent_header["session"]
                if session in self.sessions:
                    websocket = self.sessions[session]
                    await self.send_to_ws(websocket, parts, parent_header, channel_name)
                ##if session in self.sessions:
                #if True:
                #    #websocket = self.sessions[session]
                #    websockets = list(self.sessions.values())
                #    for websocket in websockets:
                #        await self.send_to_ws(websocket, parts, parent_header, channel_name)

    async def _wait_for_ready(self) -> None:
        while True:
            msg = create_message("kernel_info_request")
            msg_ser = serialize_message(msg, self.kernel.key)
            await self.kernel.shell_stream.send(msg_ser)
            with move_on_after(0.2) as scope:
                msg_ser = await self.kernel.shell_stream.receive()
                idents, msg_list = feed_identities(msg_ser)
                msg = deserialize_message(msg_list)
            if not scope.cancelled_caught and msg["msg_type"] == "kernel_info_reply":
                with move_on_after(0.2) as scope:
                    msg_ser = await self.kernel.iopub_stream.receive()
                    idents, msg_list = feed_identities(msg_ser)
                    msg = deserialize_message(msg_list)
                if scope.cancelled_caught:
                    # IOPub not connected, start over
                    pass
                else:
                    return

    async def send_to_kernel(self, websocket):
        if not websocket.accepted_subprotocol:
            while True:
                try:
                    msg = await receive_json_or_bytes(websocket.websocket)
                except BaseExceptions as e:
                    print(f"{e=}")
                    return
                if not self.can_execute:
                    continue
                msg_type = msg["header"]["msg_type"]
                if (msg_type in self.blocked_messages) or (
                    self.allowed_messages is not None and msg_type not in self.allowed_messages
                ):
                    continue
                channel = msg.pop("channel")
                msg_ser = serialize_message(msg, self.kernel.key)
                if channel == "shell":
                    await self.kernel.shell_stream.send(msg_ser)
                elif channel == "control":
                    await self.kernel.control_stream.send(msg_ser)
                elif channel == "stdin":
                    await self.kernel.stdin_stream.send(msg_ser)
        elif websocket.accepted_subprotocol == "v1.kernel.websocket.jupyter.org":
            while True:
                msg = await websocket.websocket.receive_bytes()
                if not self.can_execute:
                    continue
                channel, parts = deserialize_msg_from_ws_v1(msg)
                # NOTE: we parse the header for message filtering
                # it is not as bad as parsing the content
                header = json.loads(parts[0])
                msg_type = header["msg_type"]
                if (msg_type in self.blocked_messages) or (
                    self.allowed_messages is not None and msg_type not in self.allowed_messages
                ):
                    continue
                msg = parts[:4]
                buffers = parts[4:]
                to_send = [DELIM, sign(msg, self.kernel.key)] + msg + buffers
                if channel == "shell":
                    await self.kernel.shell_stream.send(to_send)
                elif channel == "control":
                    await self.kernel.control_stream.send(to_send)
                elif channel == "stdin":
                    await self.kernel.stdin_stream.send(to_send)

    async def send_to_ws(self, websocket, parts, parent_header, channel_name):
        print(f"send_to_ws {parts=} {parent_header=} {channel_name=}")
        try:
            if not websocket.accepted_subprotocol:
                # default, "legacy" protocol
                msg = deserialize_message(parts, parent_header=parent_header)
                msg["channel"] = channel_name
                try:
                    await send_json_or_bytes(websocket.websocket, msg)
                except BaseException as exc:
                    print(f"{exc=}")
                if channel_name == "iopub":
                    if "content" in msg and "execution_state" in msg["content"]:
                        self.last_activity = {
                            "date": msg["header"]["date"],
                            "execution_state": msg["content"]["execution_state"],
                        }
            elif websocket.accepted_subprotocol == "v1.kernel.websocket.jupyter.org":
                bin_msg = b"".join(serialize_msg_to_ws_v1(parts, channel_name))
                try:
                    await websocket.websocket.send_bytes(bin_msg)
                except BaseException:
                    pass
                # FIXME: update last_activity
                # but we don't want to parse the content!
                # or should we request it from the control channel?
        except BaseException as exc:
            print(f"{exc=}")


async def receive_json_or_bytes(websocket):
    assert websocket.application_state == WebSocketState.CONNECTED
    message = await websocket.receive()
    websocket._raise_on_disconnect(message)
    if "text" in message:
        return json.loads(message["text"])
    msg = from_binary(message["bytes"])
    return msg


async def send_json_or_bytes(websocket, msg):
    bmsg = to_binary(msg)
    if bmsg is None:
        await websocket.send_json(msg)
    else:
        await websocket.send_bytes(bmsg)

# file: routes.py

import json
import uuid
from functools import partial
from http import HTTPStatus
from pathlib import Path

import structlog
from anyio import TASK_STATUS_IGNORED, Event, Lock, create_task_group
from anyio.abc import TaskStatus
from fastapi import HTTPException, Response
from fastapi.responses import FileResponse
from starlette.requests import Request

try:
    from watchfiles import Change, awatch
    watchfiles_installed = True
except ImportError:
    watchfiles_installed = False

from jupyverse_api.app import App
from jupyverse_api.auth import Auth, User
from jupyverse_api.frontend import FrontendConfig
from jupyverse_api.kernel import DefaultKernelFactory, KernelFactory
from jupyverse_api.kernels import Kernels, KernelsConfig
from jupyverse_api.kernels.models import CreateSession, Execution, Kernel as _Kernel, Notebook, Session
from jupyverse_api.main import Lifespan
from jupyverse_api.yjs import Yjs

logger = structlog.get_logger()


class _Kernels(Kernels):
    def __init__(
        self,
        app: App,
        kernels_config: KernelsConfig,
        auth: Auth,
        frontend_config: FrontendConfig,
        yjs: Yjs | None,
        lifespan: Lifespan,
        default_kernel_factory: DefaultKernelFactory,
    ) -> None:
        super().__init__(app=app, auth=auth)
        self.kernels_config = kernels_config
        self.frontend_config = frontend_config
        self.yjs = yjs
        self.lifespan = lifespan
        self.default_kernel_factory = default_kernel_factory

        self.kernelspecs: dict = {}
        self.kernel_id_to_connection_file: dict[str, str] = {}
        self.sessions: dict[str, Session] = {}
        self.kernels = kernels
        self._app = app
        self.stop_event = Event()
        self._stop_lock = Lock()
        self.kernel_factories: dict[str, KernelFactory] = {}

    async def watch_connection_files(self, path: Path) -> None:
        pass

    async def start(self, *, task_status: TaskStatus[None] = TASK_STATUS_IGNORED) -> None:
        async with create_task_group() as tg:
            self.task_group = tg
            tg.start_soon(self.on_shutdown)
            task_status.started()
            await self.stop_event.wait()

    async def stop(self) -> None:
        async with self._stop_lock:
            if self.stop_event.is_set():
                return

            async with create_task_group() as tg:
                for kernel_id, kernel in self.kernels.items():
                    logger.info("Stopping kernel", kernel_id=kernel_id)
                    tg.start_soon(kernel["server"].stop)
                    if kernel["driver"] is not None:
                        tg.start_soon(kernel["driver"].stop)
            self.stop_event.set()
            self.task_group.cancel_scope.cancel()

    async def on_shutdown(self):
        await self.lifespan.shutdown_request.wait()
        await self.stop()

    async def get_status(
        self,
        user: User,
    ):
        started = self._app.started_time.isoformat().replace("+00:00", "Z")
        last_activity = self._app.last_activity.isoformat().replace("+00:00", "Z")
        connections = sum(
            [kernel["server"].connections for kernel in kernels.values() if "server" in kernel]
        )
        return {
            "started": started,
            "last_activity": last_activity,
            "kernels": len(kernels),
            "connections": connections,
        }

    async def get_kernelspecs(
        self,
        user: User,
    ):
        for search_path in kernelspec_dirs():
            for path in Path(search_path).glob("*/kernel.json"):
                with open(path) as f:
                    spec = json.load(f)
                name = path.parent.name
                resources = {
                    f.stem: f"{self.frontend_config.base_url}kernelspecs/{name}/{f.name}"
                    for f in path.parent.iterdir()
                    if f.is_file() and f.name != "kernel.json"
                }
                self.kernelspecs[name] = {"name": name, "spec": spec, "resources": resources}
        return {"default": self.kernels_config.default_kernel, "kernelspecs": self.kernelspecs}

    async def get_kernelspec(
        self,
        kernel_name,
        file_name,
        user: User,
    ):
        for search_path in kernelspec_dirs():
            file_path = Path(search_path) / kernel_name / file_name
            if file_path.exists():
                return FileResponse(file_path)

        raise HTTPException(
            status_code=404, detail=f"Kernelspec {kernel_name}/{file_name} not found"
        )

    async def get_kernels(
        self,
        user: User,
    ):
        results = []
        for kernel_id, kernel in kernels.items():
            if kernel["server"]:
                connections = kernel["server"].connections
                last_activity = kernel["server"].last_activity["date"]
                execution_state = kernel["server"].last_activity["execution_state"]
            else:
                connections = 0
                last_activity = ""
                execution_state = "idle"
            results.append(
                {
                    "id": kernel_id,
                    "name": kernel["name"],
                    "connections": connections,
                    "last_activity": last_activity,
                    "execution_state": execution_state,
                }
            )
        return results

    async def delete_session(
        self,
        session_id: str,
        user: User,
    ):
        kernel_id = self.sessions[session_id].kernel.id
        kernel_server = kernels[kernel_id]["server"]
        await kernel_server.stop()
        del kernels[kernel_id]
        if kernel_id in self.kernel_id_to_connection_file:
            del self.kernel_id_to_connection_file[kernel_id]
        del self.sessions[session_id]
        return Response(status_code=HTTPStatus.NO_CONTENT.value)

    async def rename_session(
        self,
        request: Request,
        user: User,
    ):
        rename_session = await request.json()
        session_id = rename_session.pop("id")
        for key, value in rename_session.items():
            setattr(self.sessions[session_id], key, value)
        return self.sessions[session_id]

    async def get_sessions(
        self,
        user: User,
    ):
        for session in self.sessions.values():
            kernel_id = session.kernel.id
            if kernel_id in kernels:
                kernel_server = kernels[kernel_id]["server"]
                session.kernel.last_activity = kernel_server.last_activity["date"]
                session.kernel.execution_state = kernel_server.last_activity["execution_state"]
        return list(self.sessions.values())

    async def create_session(
        self,
        request: Request,
        user: User,
    ):
        try:
            create_session = CreateSession(**(await request.json()))
            kernel_id = create_session.kernel.id
            kernel_name = create_session.kernel.name
            if kernel_name is not None:
                # launch a new ("internal") kernel
                kernel_cwd = Path(create_session.path).parent
                while True:
                    if kernel_cwd.is_dir():
                        break
                    kernel_cwd = kernel_cwd.parent
                kernel_server = KernelServer(
                    kernelspec_path=Path(find_kernelspec(kernel_name)).as_posix(),
                    kernel_cwd=str(kernel_cwd),
                    default_kernel_factory=self.default_kernel_factory,
                )
                kernel_id = str(uuid.uuid4())
                kernels[kernel_id] = {"name": kernel_name, "server": kernel_server, "driver": None}
                logger.info("Starting kernel", kernel_id=kernel_id, kernel_name=kernel_name)
                kernel_factory = self.kernel_factories.get(kernel_name)
                logger.info("Starting kernel", kernel_factory=kernel_factory)
                await self.task_group.start(partial(kernel_server.start, kernel_factory=kernel_factory))
            elif kernel_id is not None:
                # external or already running kernel
                if kernel_id not in kernels:
                    raise HTTPException(status_code=404, detail=f"Kernel ID not found: {kernel_id}")
                kernel_name = kernels[kernel_id]["name"]
                if kernels[kernel_id]["server"] is None:
                    kernel_server = KernelServer(
                        connection_file=self.kernel_id_to_connection_file[kernel_id],
                        write_connection_file=False,
                        default_kernel_factory=self.default_kernel_factory,
                    )
                    kernels[kernel_id]["server"] = kernel_server
                    await self.task_group.start(partial(kernel_server.start, launch_kernel=False))
                else:
                    kernel_server = kernels[kernel_id]["server"]
            else:
                return
            #session_id = "foo"#str(uuid.uuid4())
            session_id = str(uuid.uuid4())
            session = Session(
                id=session_id,
                path=create_session.path,
                name=create_session.name,
                type=create_session.type,
                kernel=_Kernel(
                    id=kernel_id,
                    name=kernel_name,
                    connections=kernel_server.connections,
                    last_activity=kernel_server.last_activity["date"],
                    execution_state=kernel_server.last_activity["execution_state"],
                ),
                notebook=Notebook(
                    path=create_session.path,
                    name=create_session.name,
                ),
            )
            self.sessions[session_id] = session
            return session
        except BaseException as e:
            print(f"{e=}")

    async def interrupt_kernel(
        self,
        kernel_id,
        user: User,
    ):
        if kernel_id in kernels:
            kernel = kernels[kernel_id]
            await kernel["server"].interrupt()
            result = {
                "id": kernel_id,
                "name": kernel["name"],
                "connections": kernel["server"].connections,
                "last_activity": kernel["server"].last_activity["date"],
                "execution_state": kernel["server"].last_activity["execution_state"],
            }
            return result

    async def restart_kernel(
        self,
        kernel_id,
        user: User,
    ):
        if kernel_id in kernels:
            kernel = kernels[kernel_id]
            await self.task_group.start(kernel["server"].restart)
            result = {
                "id": kernel_id,
                "name": kernel["name"],
                "connections": kernel["server"].connections,
                "last_activity": kernel["server"].last_activity["date"],
                "execution_state": kernel["server"].last_activity["execution_state"],
            }
            return result

    async def execute_cell(
        self,
        request: Request,
        kernel_id,
        user: User,
    ):
        pass

    async def get_kernel(
        self,
        kernel_id,
        user: User,
    ):
        if kernel_id in kernels:
            kernel = kernels[kernel_id]
            result = {
                "id": kernel_id,
                "name": kernel["name"],
                "connections": kernel["server"].connections,
                "last_activity": kernel["server"].last_activity["date"],
                "execution_state": kernel["server"].last_activity["execution_state"],
            }
            return result

    async def shutdown_kernel(
        self,
        kernel_id,
        user: User,
    ):
        logger.info("Stopping kernel", kernel_id=kernel_id)
        if kernel_id in kernels:
            await kernels[kernel_id]["server"].stop()
            del kernels[kernel_id]
        for session_id in [k for k, v in self.sessions.items() if v.kernel.id == kernel_id]:
            del self.sessions[session_id]
        return Response(status_code=HTTPStatus.NO_CONTENT.value)

    async def kernel_channels(
        self,
        kernel_id,
        session_id,
        websocket_permissions,
    ):
        if websocket_permissions is None:
            return
        websocket, permissions = websocket_permissions
        subprotocol = (
            "v1.kernel.websocket.jupyter.org"
            if "v1.kernel.websocket.jupyter.org" in websocket["subprotocols"]
            else None
        )
        await websocket.accept(subprotocol=subprotocol)
        accepted_websocket = AcceptedWebSocket(websocket, subprotocol)
        if kernel_id in kernels:
            kernel_server = kernels[kernel_id]["server"]
            print(f"{kernel_server=}")
            if kernel_server is None:
                # this is an external kernel
                # kernel is already launched, just start a kernel server
                kernel_server = KernelServer(
                    connection_file=self.kernel_id_to_connection_file[kernel_id],
                    write_connection_file=False,
                )
                await self.task_group.start(partial(kernel_server.start, launch_kernel=False))
                kernels[kernel_id]["server"] = kernel_server
            print("kernel_channels")
            await kernel_server.serve(accepted_websocket, session_id, permissions)

    def register_kernel_factory(
        self,
        kernel_name: str,
        kernel_factory: KernelFactory,
    ) -> None:
        self.kernel_factories[kernel_name] = kernel_factory

# file: main.py

import structlog
from anyio import create_task_group
from fps import Module

from jupyverse_api.app import App
from jupyverse_api.auth import Auth
from jupyverse_api.frontend import FrontendConfig
from jupyverse_api.kernel import DefaultKernelFactory
from jupyverse_api.kernels import Kernels, KernelsConfig
from jupyverse_api.main import Lifespan
from jupyverse_api.yjs import Yjs

log = structlog.get_logger()


class KernelsModule(Module):
    def __init__(self, name: str, **kwargs):
        super().__init__(name)
        self.config = KernelsConfig(**kwargs)

    async def prepare(self) -> None:
        self.put(self.config, KernelsConfig)

        app = await self.get(App)
        auth = await self.get(Auth)  # type: ignore[type-abstract]
        frontend_config = await self.get(FrontendConfig)
        lifespan = await self.get(Lifespan)
        yjs = (
            await self.get(Yjs)  # type: ignore[type-abstract]
            if self.config.require_yjs
            else None
        )
        default_kernel_factory = await self.get(DefaultKernelFactory)

        self.kernels = _Kernels(
            app,
            self.config,
            auth,
            frontend_config,
            yjs,
            lifespan,
            default_kernel_factory,
        )
        self.put(self.kernels, Kernels, teardown_callback=self.kernels.stop)

        async with create_task_group() as tg:
            tg.start_soon(self.kernels.start)
            self.done()
