"""Inject predictions.json into the site template.

Produces two flavors from site/template.html:
  * docs/index.html  - full standalone page (GitHub Pages ready)
  * docs/artifact.html - body-only fragment (for hosts that wrap it)
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TEMPLATE = ROOT / "site" / "template.html"
DATA = ROOT / "data" / "predictions.json"
DOCS = ROOT / "docs"

WRAPPER = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
{content}
</html>
"""


def build() -> None:
    payload = json.loads(DATA.read_text())
    # </script> inside JSON strings would terminate the script tag early.
    data_js = json.dumps(payload).replace("</", "<\\/")
    fragment = TEMPLATE.read_text().replace("__DATA_JSON__", data_js)

    DOCS.mkdir(exist_ok=True)
    (DOCS / "artifact.html").write_text(fragment)

    # For the standalone page, move <title>/<style> into a real <head>.
    head_end = fragment.index("</style>") + len("</style>")
    full = WRAPPER.format(
        content=fragment[:head_end] + "\n</head>\n<body>"
        + fragment[head_end:] + "</body>"
    )
    (DOCS / "index.html").write_text(full)
    print(f"built docs/index.html ({len(full)//1024} KiB) and docs/artifact.html")


if __name__ == "__main__":
    build()
