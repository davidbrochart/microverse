importScripts("pyjs_runtime_browser.js");

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
  pyjs.exec(`MAIN`);
  const serverReady = pyjs.exec_eval(`task = create_task(wait_server_ready()); task`);
  await serverReady;
};

self.addEventListener("install", (event) => {
  event.waitUntil(
    startServer(),
  );
});

const responseFromServer = async (request) => {
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

  if (request.url.includes("api/kernels") && request.url.includes("channels")) {
    const task = pyjs.exec_eval(`task = create_task(client.create_websocket('${request.url}')); task`);
    const ws_id = await task
    if (ws_id === "error") {
      const response = new Response(ws_id, {status: 404});
      return response;
    } else {
      const response = new Response(ws_id, {status: 200});
      return response;
    }
  }
  if (request.url.includes("/microverse/websocket/send/")) {
    const id = request.url.slice(-32);
    await pyjs.exec_eval(`task = create_task(client.send_websocket('${id}', '${request_body}')); task`);
    const response = new Response('', {status: 200});
    return response;
  }
  if (request.url.includes("/microverse/websocket/receive/")) {
    const id = request.url.slice(-32);
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
  const task = pyjs.exec_eval(`task = create_task(client.send_request({'method': '${request.method}', 'url': '${request.url}', 'body': '${request_body}', 'headers': '${JSON.stringify(headers)}'})); task`);
  const res = await task;
  const msg = JSON.parse(res);
  var response_body = null;
  if (msg.status !== 204) {
    response_body = (typeof msg.body === 'string') ? msg.body : JSON.stringify(msg.body);
    const isMain = request.url.includes("/microverse/static/lab/main.");
    if (isMain) {
      response_body = `WEBSOCKET` + response_body;
    }
  }
  const response = new Response(response_body, {status: msg.status, headers: msg.headers});
  return response;
};

self.addEventListener("fetch", (event) => {
  event.respondWith(responseFromServer(event.request));
});

addEventListener("message", (event) => {
  if (event.data === "get version") {
    event.waitUntil(
      (async () => {
        const client = event.source;
        client.postMessage({
          type: "type",
          version: "VERSION",
        });
      })(),
    );
  }
});
