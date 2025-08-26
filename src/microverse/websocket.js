class WebSocket {
  constructor(url) {
    this._closed = false;
    var http_url;
    if (url.startsWith('wss')) {
      http_url = 'https' + url.slice(3);
    } else {
      http_url = 'http' + url.slice(2);
    }
    const i = baseUrl.length + 'microverse/'.length;
    http_url = baseUrl + 'microverse/microverse_websocket/open/' + http_url.slice(i);
    fetch(http_url).then((response) => {
      if (!response.ok) {
        this._closed = true;
      } else {
        response.text().then((data) => {
          this.id = data;
          if (this._onopen) {
            this._onopen();
          }
          this.receive();
        });
      }
    });
  }
  set onopen(handler) {
    this._onopen = handler;
    if (this.id) {
      handler();
    }
  }
  set onclose(handler) {
    this._onclose = handler;
    if (this._closed) {
      handler();
    }
  }
  set onerror(handler) {
    this._onerror = handler;
    if (this._closed) {
      handler();
    }
  }
  set onmessage(handler) {
    this._onmessage = handler;
  }
  set binaryType(value) {
  }
  get protocol() {
    return '';
  }
  send(data) {
    var url;
    if (typeof data === "string"  || data instanceof String) {
      url = baseUrl + 'microverse/microverse_websocket/send_text/' + this.id;
    } else {
      url = baseUrl + 'microverse/microverse_websocket/send_bytes/' + this.id;
    }
    fetch(url, {
      method: 'POST',
      body: data
    });
  }
  async receive() {
    while (!this._closed) {
      var response;
      try {
        response = await fetch(baseUrl + 'microverse/microverse_websocket/receive/' + this.id);
      } catch (error) {
        this._closed = true;
      }
      if (response && !response.ok) {
        this._closed = true;
      }
      if (!this._closed) {
        var data = await response.arrayBuffer();
        if (data.byteLength > 1) {
          if (new Uint8Array(data)[0] === 0) {  // binary
            this._onmessage({data: data.slice(1)});
          } else {  // text
            data = new TextDecoder().decode(data.slice(1));
            this._onmessage({data});
          }
        }
      }
    }
  }
  close() {
    fetch(baseUrl + 'microverse/microverse_websocket/close/' + this.id);
    this._closed = true;
  }
};
