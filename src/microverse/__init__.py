import shutil
import subprocess
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path


def main():
    version = "0.1.1"

    here = Path(__file__).absolute().parent
    main = (here / "main.py").read_text()
    websocket = (here / "websocket.js").read_text()
    asgiwebsockettransport = (here / "asgi_websocket_transport.py").read_text()
    fps_kernels = (here / "fps_kernels.py").read_text()
    fake_kernel = (here / "fake_kernel.py").read_text()

    build_dir = Path("build").absolute()
    env_dir = Path("env").absolute()
    shutil.rmtree(build_dir, ignore_errors=True)
    shutil.rmtree(env_dir, ignore_errors=True)
    build_dir.mkdir()

    def call(command: str):
        subprocess.run(command, check=True, shell=True)

    call(f'micromamba create -f {here / "environment.yml"} --platform emscripten-wasm32 --prefix {env_dir} --relocate-prefix "/" --yes')
    for filename in (env_dir / "lib_js" / "pyjs").glob("*"):
        shutil.copy(filename, build_dir)
    call(f"empack pack env --env-prefix {env_dir} --outdir {build_dir} --no-use-cache")

    index_html = (here / "index.html").read_text()
    index = (
        index_html.replace("VERSION", version)
    )
    (build_dir / "index.html").write_text(index)

    main = (
        main
        .replace("ASGIWEBSOCKETTRANSPORT", asgiwebsockettransport)
        .replace("FPS_KERNELS", fps_kernels)
        .replace("FAKE_KERNEL", fake_kernel)
    )

    service_worker_js = (here / "service-worker.js").read_text()
    service_worker = (
        service_worker_js
        .replace("MAIN", main)
        .replace("VERSION", version)
        .replace("WEBSOCKET", websocket)
    )
    (build_dir / "service-worker.js").write_text(service_worker)

    class StaticHandler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=build_dir, **kwargs)

    print("Running server at http://127.0.0.1:8000")
    server = HTTPServer(("0.0.0.0", 8000), StaticHandler)
    server.serve_forever()
