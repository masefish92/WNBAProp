"""Download WNBA data from the sportsdataverse/wehoop data repositories.

The wehoop project (the R package's data backend) maintains ESPN-sourced
parquet files, one per season, committed in the git trees of:

  * sportsdataverse/wehoop-wnba-raw   - schedules (wnba/schedules/parquet/)
  * sportsdataverse/wehoop-wnba-data  - player & team box scores
    (wnba/player_box/parquet/, wnba/team_box/parquet/)

Those trees are what wehoop's own build pipeline reads through
raw.githubusercontent.com, so they are kept current in-season.
"""

from __future__ import annotations

import argparse
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

RAW = "https://raw.githubusercontent.com"
DATA = Path(__file__).resolve().parent.parent / "data"

# dataset -> (file name pattern, repo path pattern)
DATASETS = {
    "schedule": ("wnba_schedule_{s}.parquet",
                 "sportsdataverse/wehoop-wnba-raw/main/wnba/schedules/parquet"),
    "player_box": ("player_box_{s}.parquet",
                   "sportsdataverse/wehoop-wnba-data/main/wnba/player_box/parquet"),
    "team_box": ("team_box_{s}.parquet",
                 "sportsdataverse/wehoop-wnba-data/main/wnba/team_box/parquet"),
}

UA = {"User-Agent": "wnba-prop-analytics (data refresh)"}


def fetch(dataset: str, season: int) -> bool:
    pattern, base = DATASETS[dataset]
    name = pattern.format(s=season)
    url = f"{RAW}/{base}/{name}"
    for attempt in range(4):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=180) as resp:
                blob = resp.read()
            if not blob.startswith(b"PAR1"):
                raise ValueError(f"{name}: not a parquet file")
            (DATA / name).write_bytes(blob)
            print(f"  {name}: {len(blob) // 1024} KiB")
            return True
        except urllib.error.HTTPError as e:
            if e.code == 404:  # season genuinely absent upstream
                break
            time.sleep(2 ** attempt)
        except (urllib.error.URLError, TimeoutError, ValueError):
            time.sleep(2 ** attempt)
    print(f"  {name}: NOT AVAILABLE")
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--schedule-from", type=int, default=2010)
    ap.add_argument("--box-from", type=int, default=2018)
    ap.add_argument("--to", type=int, default=0, help="last season (0 = current)")
    args = ap.parse_args()
    last = args.to or time.gmtime().tm_year

    DATA.mkdir(exist_ok=True)
    ok, missing = 0, []
    for dataset, first in (("schedule", args.schedule_from),
                           ("player_box", args.box_from),
                           ("team_box", args.box_from)):
        print(f"{dataset}:")
        for season in range(first, last + 1):
            if fetch(dataset, season):
                ok += 1
            else:
                missing.append(f"{dataset}_{season}")

    print(f"\nfetched {ok} files; missing: {missing or 'none'}")
    # The current-season player box is the one file the props site cannot
    # live without.
    if not any((DATA / f"player_box_{y}.parquet").exists() for y in (last, last - 1)):
        print("FATAL: no recent player box scores were fetched")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
