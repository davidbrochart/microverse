class WebSocket {
  constructor(url) {
    this._closed = false;
    this._messageEventListeners = [];
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
  addEventListener(type, listener) {
    switch (type) {
      case 'message':
        if (!this._messageEventListeners.includes(listener)) {
          this._messageEventListeners.push(listener);
        }
        break;
      default:
    }
  }
  removeEventListener(type, listener) {
    switch (type) {
      case 'message':
        const index = this._messageEventListeners.indexOf(listener);
        if (index > -1) {
          this._messageEventListeners.splice(index, 1);
        }
        break;
      default:
    }
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
  _dispatchMessage(data) {
    const event = {data, type: 'message'};
    const listeners = this._messageEventListeners.slice();
    if (this._onmessage) {
      listeners.unshift(this._onmessage);
    }
    for (const listener of listeners) {
      try {
        listener.call(this, event);
      } catch (error) {
        console.error('WebSocket message listener failed:', error);
      }
    }
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
            this._dispatchMessage(data.slice(1));
          } else {  // text
            data = new TextDecoder().decode(data.slice(1));
            this._dispatchMessage(data);
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
