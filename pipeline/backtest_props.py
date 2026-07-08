"""Walk-forward backtest of the prop projection engine.

Replays player box scores chronologically (2018 warm-up, evaluated from
EVAL_FROM); every projection is made strictly before the game is fed to
the engine, so results are honest out-of-sample estimates.

Because historical sportsbook lines aren't freely available, evaluation
uses two complementary views:

  1. Projection accuracy: MAE / bias / correlation vs actuals, compared
     against the naive baselines a casual bettor would use (last-5 mean,
     last-10 mean, season-to-date mean). Beating those is the edge.
  2. Probability calibration: each projection is turned into P(over) for
     a synthetic line (last-10 mean floored to x.5, a crude stand-in for
     the market number); predicted probability buckets are compared with
     realized over-rates. Calibrated probabilities are what turn a
     projection into a bet you can size.

Usage: python pipeline/backtest_props.py [--from 2021] [--json out.json]
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from props import MARKETS, MARKET_LABELS, PropEngine, load_player_box, walk

EVAL_FROM = 2021
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def run(eval_from: int = EVAL_FROM) -> dict:
    df = load_player_box()
    records: list[dict] = []
    # pregame per-(player, market) actuals history for the naive baselines
    hist: dict = defaultdict(list)

    def emit(rows, projs):
        season = int(rows["season"].iloc[0])
        for r in rows.itertuples():
            pr_act = projs.get(r.athlete_id)
            if pr_act is None:
                continue
            proj, actuals = pr_act
            for m in MARKETS:
                past = hist[(r.athlete_id, m)]
                same_season = [v for s, v in past if s == season]
                rec = dict(
                    season=season, market=m, athlete=r.athlete_id,
                    mean=proj[m]["mean"], sd=proj[m]["sd"], act=actuals[m],
                    l5=np.mean(same_season[-5:]) if len(same_season) >= 3 else np.nan,
                    l10=np.mean(same_season[-10:]) if len(same_season) >= 3 else np.nan,
                    szn=np.mean(same_season) if len(same_season) >= 3 else np.nan,
                )
                records.append(rec)

    def observe_hist(rows):
        season = int(rows["season"].iloc[0])
        for r in rows.itertuples():
            if not r.played:
                continue
            a = PropEngine._actuals(r)
            for m in MARKETS:
                hist[(r.athlete_id, m)].append((season, a[m]))

    def emit_and_track(rows, projs):
        emit(rows, projs)

    engine = PropEngine()
    for _, rows in df.groupby("game_id", sort=False):
        if int(rows["season"].iloc[0]) >= eval_from:
            projs = {}
            for r in rows.itertuples():
                if not r.played:
                    continue
                pr = engine.project(r.athlete_id, r.opponent_team_id, r.home)
                if pr is not None:
                    projs[r.athlete_id] = (pr, engine._actuals(r))
            if projs:
                emit_and_track(rows, projs)
        observe_hist(rows)
        engine.observe(rows)

    rec = pd.DataFrame(records)
    return summarize(rec)


def summarize(rec: pd.DataFrame) -> dict:
    out = {"eval_seasons": sorted(rec["season"].unique().tolist()),
           "n_projections": int(len(rec)), "markets": {}, "calibration": {}}

    for m in MARKETS:
        r = rec[rec["market"] == m]
        rb = r.dropna(subset=["l10"])  # rows where baselines exist too
        def mae(col):
            return float((rb[col] - rb["act"]).abs().mean())
        by_season = {}
        for s, g in r.groupby("season"):
            gb = g.dropna(subset=["l10"])
            by_season[int(s)] = dict(
                n=int(len(g)),
                mae=float((g["mean"] - g["act"]).abs().mean()),
                mae_l10=float((gb["l10"] - gb["act"]).abs().mean()),
                bias=float((g["mean"] - g["act"]).mean()),
            )
        out["markets"][m] = dict(
            label=MARKET_LABELS[m],
            n=int(len(r)),
            mae=float((r["mean"] - r["act"]).abs().mean()),
            rmse=float(np.sqrt(((r["mean"] - r["act"]) ** 2).mean())),
            bias=float((r["mean"] - r["act"]).mean()),
            corr=float(r["mean"].corr(r["act"])),
            mae_l5=mae("l5"), mae_l10=mae("l10"), mae_season=mae("szn"),
            mae_model_on_baseline_rows=float((rb["mean"] - rb["act"]).abs().mean()),
            by_season=by_season,
        )

    # calibration vs synthetic line = floor(l10 mean) + 0.5
    cal_rows = rec.dropna(subset=["l10"]).copy()
    cal_rows["line"] = np.floor(cal_rows["l10"]) + 0.5
    cal_rows = cal_rows[cal_rows["line"] >= 0.5]
    cal_rows["p_over"] = [
        PropEngine.prob_over(m, s, l)
        for m, s, l in zip(cal_rows["mean"], cal_rows["sd"], cal_rows["line"])
    ]
    cal_rows["hit"] = (cal_rows["act"] > cal_rows["line"]).astype(float)

    for m in list(MARKETS) + ["all"]:
        g = cal_rows if m == "all" else cal_rows[cal_rows["market"] == m]
        buckets = []
        for lo in np.arange(0.05, 0.95, 0.05):
            b = g[(g["p_over"] >= lo) & (g["p_over"] < lo + 0.05)]
            if len(b) >= 30:
                buckets.append(dict(
                    p=float(lo + 0.025), n=int(len(b)),
                    predicted=float(b["p_over"].mean()),
                    actual=float(b["hit"].mean()),
                ))
        out["calibration"][m] = buckets

    # confidence-bucket win rates vs the synthetic line (both sides playable)
    conf = []
    g = cal_rows.copy()
    g["pick_over"] = g["p_over"] >= 0.5
    g["conf"] = (g["p_over"] - 0.5).abs()
    g["won"] = np.where(g["pick_over"], g["hit"], 1.0 - g["hit"])
    for lo, hi in [(0.0, .05), (.05, .10), (.10, .15), (.15, .25), (.25, .51)]:
        b = g[(g["conf"] >= lo) & (g["conf"] < hi)]
        if len(b):
            conf.append(dict(bucket=f"{50+lo*100:.0f}-{min(50+hi*100,100):.0f}%",
                             n=int(len(b)), win_rate=float(b["won"].mean())))
    out["confidence"] = conf
    return out


def print_report(res: dict) -> None:
    print(f"\nWalk-forward prop backtest - seasons {res['eval_seasons']}, "
          f"{res['n_projections']:,} projections\n")
    hdr = (f"{'market':<14}{'n':>7}{'MAE':>7}{'bias':>7}{'corr':>6} | "
           f"{'MAE L5':>7}{'MAE L10':>8}{'MAE szn':>8}{'model*':>7}")
    print(hdr)
    print("-" * len(hdr))
    for m, s in res["markets"].items():
        print(f"{MARKET_LABELS[m]:<14}{s['n']:>7,}{s['mae']:>7.2f}"
              f"{s['bias']:>7.2f}{s['corr']:>6.2f} | "
              f"{s['mae_l5']:>7.2f}{s['mae_l10']:>8.2f}{s['mae_season']:>8.2f}"
              f"{s['mae_model_on_baseline_rows']:>7.2f}")
    print("* model MAE restricted to rows where baselines exist\n")
    print("Calibration (all markets, synthetic x.5 line = floor(L10)+0.5):")
    for b in res["calibration"]["all"]:
        bar = "#" * int(b["actual"] * 40)
        print(f"  pred {b['predicted']:.2f}  actual {b['actual']:.2f} "
              f"(n={b['n']:>6,}) {bar}")
    print("\nWin rate of model picks vs synthetic line, by confidence:")
    for c in res["confidence"]:
        print(f"  {c['bucket']:>8}: {c['win_rate']:.1%}  (n={c['n']:,})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="eval_from", type=int, default=EVAL_FROM)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()
    res = run(args.eval_from)
    print_report(res)
    if args.json:
        args.json.write_text(json.dumps(res))
        print(f"\nwrote {args.json}")
