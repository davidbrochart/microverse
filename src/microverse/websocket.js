navigator.serviceWorker.addEventListener("message", (event) => {
  if (event.data.type === "version") {
    if (event.data.version === "VERSION") {
      window.location.assign("microverse/");
    } else {
      navigator.serviceWorker.getRegistrations().then(registrations => {
        for (const registration of registrations) {
          registration.unregister();
        }
        window.location.reload();
      });
    }
  }
});

class WebSocket {
  constructor(url) {
    this._closed = false;
    const http_url = 'http' + url.slice(2);
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
    fetch('http://127.0.0.1:8000/microverse/websocket/send/' + this.id, {
      method: 'POST',
      body: data
    });
  }
  async receive() {
    while (!this._closed) {
      const response = await fetch('http://127.0.0.1:8000/microverse/websocket/receive/' + this.id);
      if (!response.ok) {
        this._closed = true;
      } else {
        const data = await response.text();
        this._onmessage({data});
      }
    }
  }
  close() {
    this._close = true;
  }
};

