"""Build the site. Placeholder until the prop engine lands."""
import sys
from pathlib import Path

if not (Path(__file__).resolve().parent.parent / "site" / "template.html").exists():
    print("site template not present yet; skipping build")
    sys.exit(0)
