import shutil
import subprocess
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path


def main():
    here = Path(__file__).parent
    asynctestclient = (here / "asynctestclient.py").read_text()
    to_thread = (here / "to_thread.py").read_text()
    contents = (here / "contents.py").read_text()
    main = (here / "main.py").read_text()

    build_dir = Path("build").absolute()
    env_dir = Path("env").absolute()
    shutil.rmtree(build_dir, ignore_errors=True)
    shutil.rmtree(env_dir, ignore_errors=True)
    build_dir.mkdir()

    def call(command: str):
        subprocess.run(command, check=True, shell=True)

    call(f'micromamba create -f environment.yml --platform emscripten-wasm32 --prefix {env_dir} --relocate-prefix "/" --yes')
    for filename in (env_dir / "lib_js" / "pyjs").glob("*"):
        shutil.copy(filename, build_dir)
    call(f"empack pack env --env-prefix {env_dir} --outdir {build_dir} --no-use-cache")

    shutil.copyfile(here / "index.html", build_dir / "index.html")

    service_worker_js = (here / "service-worker.js").read_text()
    service_worker = (
        service_worker_js.replace("MAIN", main)
        .replace("ASYNCTESTCLIENT", asynctestclient)
        .replace("TO_THREAD", to_thread)
        .replace("CONTENTS", contents)
    )
    (build_dir / "service-worker.js").write_text(service_worker)

    class StaticHandler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=build_dir, **kwargs)

    print("Running server at http://127.0.0.1:8000")
    server = HTTPServer(("0.0.0.0", 8000), StaticHandler)
    server.serve_forever()
