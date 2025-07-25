import base64
import json
import shutil
import subprocess
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

from cyclopts import App


app = App()

@app.default
def _main(*, environment: str = "environment", serve: bool = False):
    version = "0.1.2"
    here = Path(__file__).absolute().parent
    index_html = (here / "index.html").read_text()
    service_worker_js = (here / "service-worker.js").read_text()
    main = (here / "main.py").read_text()
    websocket = (here / "websocket.js").read_text()
    asgiwebsockettransport = (here / "asgi_websocket_transport.py").read_text()

    build_dir = Path("build").absolute()
    env_dir = Path("env").absolute()
    shutil.rmtree(build_dir, ignore_errors=True)
    shutil.rmtree(env_dir, ignore_errors=True)
    build_dir.mkdir()

    def get_dir_content(path: Path, contents: dict, dir: Path):
        for p in path.iterdir():
            if p.is_dir():
                (dir / p.name).mkdir()
                contents[p.name] = _content = {}
                get_dir_content(p, _content, dir / p.name)
            else:
                content_bytes = p.read_bytes()
                content_text = base64.b64encode(content_bytes).decode()
                (dir / p.name).write_text(content_text)
                contents[p.name] = None

    contents = {}
    contents_dir = build_dir / "contents"
    contents_dir.mkdir()
    get_dir_content(Path(environment) / "contents", contents, contents_dir)
    (build_dir / "contents.json").write_text(json.dumps(contents))

    def call(command: str):
        subprocess.run(command, check=True, shell=True)

    call(f'micromamba create -f {Path(environment) / "environment.yml"} --platform emscripten-wasm32 --prefix {env_dir} --relocate-prefix "/" --yes')
    for filename in (env_dir / "lib_js" / "pyjs").glob("*"):
        shutil.copy(filename, build_dir)
    call(f"empack pack env --env-prefix {env_dir} --outdir {build_dir} --no-use-cache")

    index = (
        index_html.replace("VERSION", version)
    )
    (build_dir / "index.html").write_text(index)

    main = (
        main
        .replace("ASGIWEBSOCKETTRANSPORT", asgiwebsockettransport)
    )

    service_worker = (
        service_worker_js
        .replace("MAIN", main)
        .replace("VERSION", version)
        .replace("WEBSOCKET", websocket)
    )
    (build_dir / "service-worker.js").write_text(service_worker)

    shutil.copy(here / "kernel-web-worker.js", build_dir)

    if serve:
        class StaticHandler(SimpleHTTPRequestHandler):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, directory=build_dir, **kwargs)

        print("Running server at http://127.0.0.1:8000")
        server = HTTPServer(("0.0.0.0", 8000), StaticHandler)
        server.serve_forever()


def main():
    app()


if __name__ == "__main__":
    app()
