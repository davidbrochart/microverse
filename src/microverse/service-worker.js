importScripts("pyjs_runtime_browser.js");

const startServer = async (self, event) => {
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
};

self.addEventListener("install", (event) => {
  event.waitUntil(
    startServer(self, event),
  );
});

const responseFromServer = async (request) => {
  const serverReady = pyjs.exec_eval(`task = create_task(wait_server_ready()); task`);
  await serverReady;
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
  const task = pyjs.exec_eval(`task = create_task(client.send_request({'method': '${request.method}', 'url': '${request.url}', 'body': '${request_body}', 'headers': '${JSON.stringify(headers)}'})); task`);
  const res = await task;
  const msg = JSON.parse(res);
  var response_body = null;
  if (msg.status !== 204) {
    response_body = (typeof msg.body === 'string') ? msg.body : JSON.stringify(msg.body);
  }
  const response = new Response(response_body, {status: msg.status, headers: msg.headers});
  return response;
};

self.addEventListener("fetch", (event) => {
  event.respondWith(responseFromServer(event.request));
});
