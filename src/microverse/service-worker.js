importScripts("pyjs_runtime_browser.js");

var pyjs;
var baseUrl = '';

const startServer = async () => {
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
  pyjs.exec(`MAIN`);
  pyjs.exec_eval(`main_task = create_task(main('${baseUrl}')); main_task`);
  const serverReady = pyjs.exec_eval(`task = create_task(wait_server_ready()); task`);
  await serverReady;
};

self.addEventListener("install", (event) => {
  event.waitUntil(
    startServer(),
  );
});

const kernel_web_worker_resolve = [null];
const kernel_web_worker_promise = [new Promise((_resolve) => {kernel_web_worker_resolve[0] = _resolve;})];
const kernel_callbacks = {};

const kernel_web_worker = (action, kernel_id, msg, callback) => {
  kernel_web_worker_resolve[0]({action, kernel_id, msg, callback});
}

const waitForKernelWebWorkerRequest = async () => {
  const msg = await kernel_web_worker_promise[0];
  kernel_web_worker_promise[0] = new Promise((_resolve) => {kernel_web_worker_resolve[0] = _resolve;});
  return msg;
}

const responseFromServer = async (request) => {
  const url = request.url.slice(baseUrl.length - 1);
  const headers = {};
  for (const pair of request.headers.entries()) {
    if (!pair[0].startsWith("sec-ch-ua")) {
      headers[pair[0]] = pair[1];
    }
  }
  var request_body = null;
  if (request.body) {
    myArrays = [];
    for await (const chunk of request.body) {
      myArrays = myArrays.concat([chunk]);
    }
    let length = 0;
    myArrays.forEach(item => {
      length += item.length;
    });
    let mergedArray = new Uint8Array(length);
    let offset = 0;
    myArrays.forEach(item => {
      mergedArray.set(item, offset);
      offset += item.length;
    });
    request_body = mergedArray;
  }

  if (url.includes("api/kernels") && url.includes("channels")) {
    const task = pyjs.exec_eval(`task = create_task(client.create_websocket('${url}')); task`);
    const ws_id = await task
    if (ws_id === "error") {
      const response = new Response(ws_id, {status: 404});
      return response;
    } else {
      const response = new Response(ws_id, {status: 200});
      return response;
    }
  }
  if (url.includes("/microverse/websocket/send/")) {
    const id = url.slice(-32);
    var f = pyjs.exec_eval(
`
def f(idx, data):
    return create_task(client.send_websocket(idx, data))
f
`);
    const ret = f.py_call(id, request_body);
    await ret;
    const response = new Response('', {status: 200});
    f.delete();
    return response;
  }
  if (url.includes("/microverse/websocket/receive/")) {
    const id = url.slice(-32);
    const task = pyjs.exec_eval(`task = create_task(client.receive_websocket('${id}')); task`);
    const res = await task;
    if (res) {
      const response = new Response(res, {status: 200});
      return response;
    } else {
      const response = new Response(res, {status: 404});
      return response;
    }
  }
  var f = pyjs.exec_eval(
`
def f(method, url, body, headers):
    return create_task(client.send_request(method, url, body, headers))
f
`);
  const task = f.py_call(request.method, url, request_body, JSON.stringify(headers));
  const res = await task;
  const msg = JSON.parse(res);
  var response_body = null;
  if (msg.status !== 204) {
    response_body = (typeof msg.body === 'string') ? msg.body : JSON.stringify(msg.body);
    const isMain = url.includes("/microverse/static/lab/main.");
    if (isMain) {
      response_body = `
var baseUrl = '${baseUrl}';

WEBSOCKET
` + response_body;
    }
  }
  const response = new Response(response_body, {status: msg.status, headers: msg.headers});
  return response;
};

self.addEventListener("fetch", (event) => {
  event.respondWith(responseFromServer(event.request));
});

addEventListener("message", (event) => {
  const request = event.data;
  switch (request.type) {
    case "set base url":
      baseUrl = request.baseUrl;
      break;
    case "get version":
      event.waitUntil(
        (async () => {
          const client = event.source;
          client.postMessage({
            type: "version",
            version: "VERSION",
          });
        })(),
      );
      break;
    case "wait kernel-web-worker":
      event.waitUntil(
        (async () => {
          const msg = await waitForKernelWebWorkerRequest();
          if (msg.callback !== 0) {
            kernel_callbacks[msg.kernel_id] = msg.callback;
            delete msg.callback;
          }
          const client = event.source;
          client.postMessage({
            type: "kernel-web-worker",
            msg,
          });
        })(),
      );
      break;
    default:
      kernel_callbacks[request.kernel_id](request);
  }
});
