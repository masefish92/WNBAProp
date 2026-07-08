"""Download WNBA data from the sportsdataverse/wehoop data releases.

The wehoop project (the R package's data backend) publishes ESPN-sourced
parquet files as GitHub release assets on sportsdataverse/wehoop-wnba-data:
one file per season for schedules, player box scores, and team box scores.

This script is meant to run inside GitHub Actions (unrestricted network);
it tries the known release-tag URLs first and falls back to discovering
asset URLs through the GitHub releases API if the tags ever change.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO = "sportsdataverse/wehoop-wnba-data"
DATA = Path(__file__).resolve().parent.parent / "data"

# dataset -> (file name pattern, candidate release tags)
DATASETS = {
    "schedule": ("wnba_schedule_{s}.parquet",
                 ["espn_wnba_schedules", "wnba_schedules"]),
    "player_box": ("player_box_{s}.parquet",
                   ["espn_wnba_player_boxscores", "wnba_player_box"]),
    "team_box": ("team_box_{s}.parquet",
                 ["espn_wnba_team_boxscores", "wnba_team_box"]),
}

UA = {"User-Agent": "wnba-prop-analytics (github actions data refresh)"}


def _get(url: str, timeout: int = 120) -> bytes:
    req = urllib.request.Request(url, headers=UA)
    tok = os.environ.get("GITHUB_TOKEN")
    if tok and "api.github.com" in url:
        req.add_header("Authorization", f"Bearer {tok}")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def discover_assets() -> dict[str, str]:
    """Map every release-asset file name to its download URL."""
    assets: dict[str, str] = {}
    for page in range(1, 6):
        url = f"https://api.github.com/repos/{REPO}/releases?per_page=100&page={page}"
        try:
            releases = json.loads(_get(url, timeout=60))
        except (urllib.error.URLError, urllib.error.HTTPError) as e:
            print(f"  release discovery failed ({e}); relying on known tags")
            return assets
        if not releases:
            break
        for rel in releases:
            for a in rel.get("assets", []):
                assets.setdefault(a["name"], a["browser_download_url"])
    return assets


def fetch(dataset: str, season: int, discovered: dict[str, str]) -> bool:
    pattern, tags = DATASETS[dataset]
    name = pattern.format(s=season)
    urls = [f"https://github.com/{REPO}/releases/download/{t}/{name}" for t in tags]
    if name in discovered:
        urls.insert(0, discovered[name])
    dest = DATA / name
    for url in urls:
        for attempt in range(3):
            try:
                blob = _get(url)
                if not blob.startswith(b"PAR1"):
                    raise ValueError("not a parquet file")
                dest.write_bytes(blob)
                print(f"  {name}: {len(blob) // 1024} KiB")
                return True
            except (urllib.error.HTTPError, ValueError):
                break  # wrong URL / missing season: try next candidate
            except urllib.error.URLError:
                time.sleep(2 ** attempt)  # transient network error: retry
    print(f"  {name}: NOT FOUND")
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--schedule-from", type=int, default=2010)
    ap.add_argument("--box-from", type=int, default=2018)
    ap.add_argument("--to", type=int, default=0, help="last season (0 = current)")
    args = ap.parse_args()
    last = args.to or time.gmtime().tm_year

    DATA.mkdir(exist_ok=True)
    discovered = discover_assets()
    print(f"discovered {len(discovered)} release assets")

    ok, missing = 0, []
    for dataset, first in (("schedule", args.schedule_from),
                           ("player_box", args.box_from),
                           ("team_box", args.box_from)):
        print(f"{dataset}:")
        for season in range(first, last + 1):
            if fetch(dataset, season, discovered):
                ok += 1
            else:
                missing.append(f"{dataset}_{season}")

    print(f"\nfetched {ok} files; missing: {missing or 'none'}")
    # Current-season player box is the one file the props site cannot live
    # without; everything else may legitimately not exist yet (e.g. next
    # season's schedule before it is published).
    if not (DATA / f"player_box_{last}.parquet").exists() and \
            not (DATA / f"player_box_{last - 1}.parquet").exists():
        print("FATAL: no recent player box scores were fetched")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
