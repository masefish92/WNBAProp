"""One-command local runner: refresh data, rebuild the site, serve it.

    python run_local.py              # fetch latest data, rebuild, serve
    python run_local.py --no-fetch   # rebuild + serve from existing data
    python run_local.py --backtest   # also re-run the full prop backtest (~2 min)
    python run_local.py --port 9000

First-time setup:  pip install -r requirements.txt
"""

from __future__ import annotations

import argparse
import functools
import http.server
import subprocess
import sys
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PIPELINE = ROOT / "pipeline"
DOCS = ROOT / "docs"


def run(script: str, *args: str) -> None:
    cmd = [sys.executable, str(PIPELINE / script), *args]
    print(f"\n=== {script} {' '.join(args)}")
    res = subprocess.run(cmd, cwd=PIPELINE)
    if res.returncode != 0:
        sys.exit(f"{script} failed (exit {res.returncode})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-fetch", action="store_true",
                    help="skip downloading fresh data")
    ap.add_argument("--backtest", action="store_true",
                    help="re-run the full walk-forward backtest (slower)")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    try:
        import pandas  # noqa: F401
    except ImportError:
        sys.exit("Missing dependencies - run:  pip install -r requirements.txt")

    if not args.no_fetch:
        run("fetch_data.py")
    build_args = [] if args.backtest else ["--skip-backtest"]
    if not (ROOT / "data" / "predictions.json").exists():
        build_args = []  # first build has no cached backtest to reuse
    run("build_site.py", *build_args)

    url = f"http://localhost:{args.port}/"
    handler = functools.partial(
        http.server.SimpleHTTPRequestHandler, directory=str(DOCS))
    print(f"\nServing {DOCS / 'index.html'}\n  ->  {url}   (Ctrl+C to stop)")
    webbrowser.open(url)
    http.server.ThreadingHTTPServer(("127.0.0.1", args.port), handler).serve_forever()


if __name__ == "__main__":
    main()
