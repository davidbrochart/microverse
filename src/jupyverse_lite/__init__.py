import json
import shutil
import sys
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

import jupyterlab


prefix_dir = Path(sys.prefix)
static_lab_dir = Path(jupyterlab.__file__).parent / "static"
build_dir = Path("build")
shutil.rmtree(build_dir,ignore_errors=True)
shutil.copytree(static_lab_dir, build_dir)

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

index_html = """\
<!doctype html><html lang="en"><head><meta charset="utf-8"><title>JupyterLab</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<script id="jupyter-config-data" type="application/json">PAGE_CONFIG</script>
VENDORS_NODE_MODULES
<script defer="defer" src="FULL_STATIC_URL/main.MAIN_ID.js?v=MAIN_ID"></script>
</head><body><script>/* Remove token from URL. */
  (function () {
    var location = window.location;
    var search = location.search;

    // If there is no query string, bail.
    if (search.length <= 1) {
      return;
    }

    // Rebuild the query string without the `token`.
    var query = '?' + search.slice(1).split('&')
      .filter(function (param) { return param.split('=')[0] !== 'token'; })
      .join('&');

    // Rebuild the URL with the new query string.
    var url = location.origin + location.pathname +
      (query !== '?' ? query : '') + location.hash;

    if (url === location.href) {
      return;
    }

    window.history.replaceState({ }, '', url);
  })();</script></body></html>
"""

vendors_node_modules = f'<script defer src="/static/lab/vendors-node_modules_whatwg-fetch_fetch_js.{vendor_id}.js"></script>' if vendor_id else ""
index = (
    index_html.replace("PAGE_CONFIG", json.dumps(page_config))
    .replace("MAIN_ID", main_id)
    .replace("VENDORS_NODE_MODULES", vendors_node_modules)
    .replace("FULL_STATIC_URL", full_static_url)
)

(build_dir / "index.html").write_text(index)

class StaticHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=build_dir, **kwargs)

server = HTTPServer(("0.0.0.0", 8000), StaticHandler)
server.serve_forever()
