import json
import shutil
import subprocess
import sys
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

import jupyterlab


def main():
    here = Path(__file__).parent
    asynctestclient = (here / "asynctestclient.py").read_text()
    to_thread = (here / "to_thread.py").read_text()
    contents = (here / "contents.py").read_text()
    main = (here / "main.py").read_text()

    prefix_dir = Path(sys.prefix)
    static_lab_dir = Path(jupyterlab.__file__).parent / "static"
    build_dir = Path("build").absolute()
    env_dir = Path("env").absolute()
    shutil.rmtree(build_dir, ignore_errors=True)
    shutil.rmtree(env_dir, ignore_errors=True)
    shutil.copytree(static_lab_dir, build_dir)

    def call(command: str):
        subprocess.run(command, check=True, shell=True)


    call(f'micromamba create -f environment.yml --platform emscripten-wasm32 --prefix {env_dir} --relocate-prefix "/" --yes')
    for filename in (env_dir / "lib_js" / "pyjs").glob("*"):
        shutil.copy(filename, build_dir)
    call(f"empack pack env --env-prefix {env_dir} --outdir {build_dir} --no-use-cache")

    main_id = None
    for path in static_lab_dir.glob("main.*.js"):
        main_id = path.name.split(".")[1]
        break
    assert main_id is not None

    vendor_id = None
    for path in static_lab_dir.glob("vendors-node_modules_whatwg-fetch_fetch_js.*.js"):
        vendor_id = path.name.split(".")[1]
        break

    base_url = "/"
    full_static_url = ""
    collaborative = False
    server_side_execution = False
    dev_mode = False
    disabled_extensions = []
    federated_extensions = []
    workspace = "default"

    page_config = {
                "appName": "JupyterLab",
                "appNamespace": "lab",
                "appUrl": "/lab",
                "appVersion": jupyterlab.__version__,
                "baseUrl": base_url,
                "cacheFiles": False,
                "collaborative": collaborative,
                "serverSideExecution": server_side_execution,
                "devMode": dev_mode,
                "disabledExtensions": disabled_extensions,
                "exposeAppInBrowser": False,
                "extraLabextensionsPath": [],
                "federated_extensions": federated_extensions,
                "fullAppUrl": f"{base_url}lab",
                "fullLabextensionsUrl": f"{base_url}lab/extensions",
                "fullLicensesUrl": f"{base_url}lab/api/licenses",
                "fullListingsUrl": f"{base_url}lab/api/listings",
                "fullMathjaxUrl": f"{base_url}static/notebook/components/MathJax/MathJax.js",
                "fullSettingsUrl": f"{base_url}lab/api/settings",
                "fullStaticUrl": full_static_url,
                "fullThemesUrl": f"{base_url}lab/api/themes",
                "fullTranslationsApiUrl": f"{base_url}lab/api/translations",
                "fullTreeUrl": f"{base_url}lab/tree",
                "fullWorkspacesApiUrl": f"{base_url}lab/api/workspaces",
                "ignorePlugins": [],
                "labextensionsUrl": "/lab/extensions",
                "licensesUrl": "/lab/api/licenses",
                "listingsUrl": "/lab/api/listings",
                "mathjaxConfig": "TeX-AMS-MML_HTMLorMML-full,Safe",
                "mode": "multiple-document",
                "notebookVersion": "[1, 9, 0]",
                "quitButton": True,
                "settingsUrl": "/lab/api/settings",
                "store_id": 0,
                "schemasDir": (
                    prefix_dir / "share" / "jupyter" / "lab" / "schemas"
                ).as_posix(),
                "terminalsAvailable": True,
                "themesDir": (prefix_dir / "share" / "jupyter" / "lab" / "themes").as_posix(),
                "themesUrl": "/lab/api/themes",
                "token": "4e2804532de366abc81e32ab0c6bf68a73716fafbdbb2098",
                "translationsApiUrl": "/lab/api/translations",
                "treePath": "",
                "workspace": workspace,
                "treeUrl": "/lab/tree",
                "workspacesApiUrl": "/lab/api/workspaces",
                "wsUrl": "",
            }

    index_html = (here / "index.html").read_text()
    service_worker_js = (here / "service-worker.js").read_text()

    vendors_node_modules = f'<script defer src="/static/lab/vendors-node_modules_whatwg-fetch_fetch_js.{vendor_id}.js"></script>' if vendor_id else ""
    index = (
        index_html.replace("PAGE_CONFIG", json.dumps(page_config))
        .replace("MAIN_ID", main_id)
        .replace("VENDORS_NODE_MODULES", vendors_node_modules)
        .replace("FULL_STATIC_URL", full_static_url)
        .replace("TO_THREAD", to_thread)
        .replace("CONTENTS", contents)
    )
    (build_dir / "index.html").write_text(index)

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
