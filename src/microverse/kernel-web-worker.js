importScripts("pyjs_runtime_browser.js");

var kernel_id;
var pyjs;
var msg;

const startKernel = async (kernel_id) => {
  let locateFile = function(filename){
      if(filename.endsWith('pyjs_runtime_browser.wasm')){
          return './pyjs_runtime_browser.wasm'; // location of the wasm 
                                              // file on the server relative 
                                              // to the pyjs_runtime_browser.js file
      }
  };
  pyjs = await createModule({locateFile});
  await pyjs.bootstrap_from_empack_packed_environment(
      './empack_env_meta.json', // location of the environment 
                                // meta file on the server
      '.'                       // location of the packed 
                                // environment on the server
  );

  // FIXME: here contents should not be copied from the static assets,
  // but synchronized with jupyverse's contents.
  var response = await fetch("contents.json");
  var data = await response.text();
  var contents = JSON.parse(data);
  pyjs.exec(`
import base64
import os
from pathlib import Path

p = Path()
(p / "contents").mkdir()
os.chdir("contents")
`);
  const set_dir_content = async (contents, cur_dir) => {
    for (const k in contents) {
      if (contents[k]) {
        pyjs.exec(`(p / "${cur_dir}" / "${k}").mkdir()`);
        if (cur_dir !== "") {
          await set_dir_content(contents[k], `${cur_dir}/${k}`);
        } else {
          await set_dir_content(contents[k], k);
        }
      } else {
        var content_path;
        if (cur_dir !== "") {
          content_path = `contents/${cur_dir}/${k}`;
        } else {
          content_path = `contents/${k}`;
        }
        response = await fetch(content_path);
        data = await response.text();
        pyjs.exec(`content_bytes = base64.b64decode("${data}"); (p / "${cur_dir}" / "${k}").write_bytes(content_bytes)`);
      }
    }
  };
  await set_dir_content(contents, "");

  task = pyjs.exec(`
import pyjs
from asyncio import Event, create_task
from fps_akernel_task.akernel_task import AKernelTask  # FIXME: support other kernels

kernel_id = "${kernel_id}"
akernel_task = AKernelTask()

async def main():
    task0 = create_task(akernel_task.start())
    await akernel_task.started.wait()

    async def from_shell_receive_stream():
        async for msg in akernel_task._from_shell_receive_stream:
            try:
                msg = pyjs.to_js(msg)
                pyjs.js.Function("message", "msg = message;")(msg)
                pyjs.js.Function("self.postMessage({type: 'shell', kernel_id: '" + kernel_id + "', msg: msg});")()
            except BaseException as e:
                print(f"{e=}")

    async def from_control_receive_stream():
        async for msg in akernel_task._from_control_receive_stream:
            try:
                msg = pyjs.to_js(msg)
                pyjs.js.Function("message", "msg = message;")(msg)
                pyjs.js.Function("self.postMessage({type: 'control', kernel_id: '" + kernel_id + "', msg: msg});")()
            except BaseException as e:
                print(f"{e=}")

    async def from_stdin_receive_stream():
        async for msg in akernel_task._from_stdin_receive_stream:
            try:
                msg = pyjs.to_js(msg)
                pyjs.js.Function("message", "msg = message;")(msg)
                pyjs.js.Function("self.postMessage({type: 'stdin', kernel_id: '" + kernel_id + "', msg: msg});")()
            except BaseException as e:
                print(f"{e=}")

    async def from_iopub_receive_stream():
        async for msg in akernel_task._from_iopub_receive_stream:
            try:
                msg = pyjs.to_js(msg)
                pyjs.js.Function("message", "msg = message;")(msg)
                pyjs.js.Function("self.postMessage({type: 'iopub', kernel_id: '" + kernel_id + "', msg: msg});")()
            except BaseException as e:
                print(f"{e=}")

    task1 = create_task(from_shell_receive_stream())
    task2 = create_task(from_control_receive_stream())
    task3 = create_task(from_stdin_receive_stream())
    task4 = create_task(from_iopub_receive_stream())

    await Event().wait()

task = create_task(main())
`);
};

self.onmessage = async (msg) => {
  const message = msg.data;
  if (message.type === "start") {
    kernel_id = message.kernel_id;
    await startKernel(message.kernel_id);
    self.postMessage({type: "started", kernel_id: message.kernel_id});
  } else {
    var py_send_stream = pyjs.exec_eval(`
def send_stream(msg):
    try:
        msg = [bytes(pyjs.to_py(m)) for m in msg]
        create_task(akernel_task._to_${message.type}_send_stream.send(msg))
    except BaseException as e:
        print(f"{e=}")

send_stream
`
    )
    try {
      py_send_stream.py_call(message.msg)
    } catch (error) {
      console.log(error);
    }
  }
};
