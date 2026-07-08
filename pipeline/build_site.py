"""Build the site: run both models and bake everything into docs/index.html.

Produces one self-contained page (no external JS/CSS) whose data payload is
injected into site/template.html at the __DATA_JSON__ marker. The payload:

  * elo      - team ratings, records, game predictions for upcoming games
  * slate    - upcoming games with per-player prop projections (mean +
               variance per market; the page turns those into P(over) for
               any line the user types)
  * logs     - recent game logs per player, for hit-rate charts and splits
  * defense  - opponent stat-allowed factors (matchup context)
  * backtest - walk-forward metrics for both models (the honesty tab)

Usage: python pipeline/build_site.py [--skip-backtest]
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

import backtest_props
from backtest import EVAL_FROM, evaluate
from model import EloModel, load_games, run_history
from props import (BASE_STATS, MARKETS, MARKET_LABELS, PropEngine,
                   load_player_box, walk)

ROOT = Path(__file__).resolve().parent.parent
TEMPLATE = ROOT / "site" / "template.html"
DOCS = ROOT / "docs"
DATA_OUT = ROOT / "data" / "predictions.json"

SLATE_DAYS = 10          # project props this many days ahead
LOG_GAMES = 40           # game-log depth per player shipped to the page
ROSTER_STALE_DAYS = 60   # drop players who haven't appeared in this long


def build_elo(games: pd.DataFrame) -> tuple[dict, EloModel]:
    """Game predictions payload (condensed version of the WNBA Edge site)."""
    current_season = int(games["season"].max())
    model = EloModel()
    rows = []
    recent = []
    for g in games[games["completed"]].itertuples():
        pre = model.update(g.home_id, g.away_id, g.date, int(g.season),
                           g.home_score, g.away_score, bool(g.neutral_site))
        if int(g.season) >= EVAL_FROM:
            rows.append(dict(
                season=int(g.season), win_prob_home=pre["win_prob_home"],
                spread_home=pre["spread_home"], total=pre["total"],
                margin=g.home_score - g.away_score,
                actual_total=g.home_score + g.away_score))
        if int(g.season) == current_season:
            recent.append(dict(
                date=g.date.strftime("%Y-%m-%d"),
                home=g.home_id, away=g.away_id,
                home_score=int(g.home_score), away_score=int(g.away_score),
                win_prob_home=round(pre["win_prob_home"], 3),
                spread_home=round(pre["spread_home"], 1)))
    preds_hist = pd.DataFrame(rows)
    metrics_all = {k: round(float(v), 4) for k, v in evaluate(preds_hist).items()}
    season_hist = preds_hist[preds_hist["season"] == current_season]
    metrics_season = ({k: round(float(v), 4) for k, v in evaluate(season_hist).items()}
                      if len(season_hist) else {})

    season_games = games[games["season"] == current_season]
    teams: dict[str, dict] = {}
    for g in season_games.itertuples():
        for side in ("home", "away"):
            tid = getattr(g, f"{side}_id")
            if tid not in teams:
                teams[tid] = dict(
                    id=tid,
                    name=getattr(g, f"{side}_display_name"),
                    abbr=getattr(g, f"{side}_abbreviation"),
                    logo=getattr(g, f"{side}_logo"),
                    color=(getattr(g, f"{side}_color") or "888888"),
                    wins=0, losses=0)
    for g in season_games[season_games["completed"]].itertuples():
        home_won = g.home_score > g.away_score
        teams[g.home_id]["wins" if home_won else "losses"] += 1
        teams[g.away_id]["losses" if home_won else "wins"] += 1
    for tid, t in teams.items():
        st = model.teams.get(tid)
        t["rating"] = round(st.rating, 1) if st else 1500.0
        t["off"] = round(st.pts_for, 1) if st else None
        t["deff"] = round(st.pts_against, 1) if st else None
    order = sorted(teams.values(), key=lambda t: -t["rating"])
    for i, t in enumerate(order, 1):
        t["rank"] = i

    now = pd.Timestamp.now(tz="UTC")
    upcoming = []
    for g in season_games[~season_games["completed"]].itertuples():
        if g.home_id not in teams or g.away_id not in teams:
            continue
        if not (now - timedelta(hours=12) <= g.date <= now + timedelta(days=SLATE_DAYS)):
            continue
        pre = model.pregame(g.home_id, g.away_id, g.date, current_season,
                            bool(g.neutral_site))
        upcoming.append(dict(
            game_id=str(g.game_id), date=g.date.isoformat(),
            home=g.home_id, away=g.away_id,
            venue=(g.venue_full_name or ""),
            win_prob_home=round(pre["win_prob_home"], 3),
            spread_home=round(pre["spread_home"], 1),
            total=round(pre["total"], 1),
            home_pts=round(pre["home_pts"], 1),
            away_pts=round(pre["away_pts"], 1)))
    upcoming.sort(key=lambda g: g["date"])

    payload = dict(
        season=current_season,
        teams=teams,
        rankings=[t["id"] for t in order],
        upcoming=upcoming,
        recent=recent[::-1][:30],
        metrics=dict(all=metrics_all, season=metrics_season, eval_from=EVAL_FROM),
    )
    return payload, model


def build_props(games: pd.DataFrame, elo: dict) -> dict:
    df = load_player_box()
    engine = walk(df)  # replay everything: engine now holds pregame state

    current_season = int(df["season"].max())
    now = pd.Timestamp.now(tz=None)

    # roster: player -> current team = team of their most recent appearance
    latest = (df[df["played"]].sort_values("game_date")
              .groupby("athlete_id").tail(1))
    roster: dict[str, list] = {}
    last_seen: dict[str, str] = {}
    for r in latest.itertuples():
        if (now - r.game_date).days > ROSTER_STALE_DAYS:
            continue
        roster.setdefault(str(r.team_id), []).append(r.athlete_id)
        last_seen[r.athlete_id] = r.game_date.strftime("%Y-%m-%d")

    team_abbr = {str(tid): t["abbr"] for tid, t in elo["teams"].items()}

    # prop projections for every upcoming game on the slate
    slate = []
    for g in elo["upcoming"]:
        entry = dict(game_id=g["game_id"], date=g["date"], home=str(g["home"]),
                     away=str(g["away"]), players=[])
        for tid, opp, is_home in ((g["home"], g["away"], True),
                                  (g["away"], g["home"], False)):
            for aid in roster.get(str(tid), []):
                proj = engine.project(aid, opp, is_home)
                if proj is None:
                    continue
                p = engine.players[aid]
                entry["players"].append(dict(
                    id=str(aid), name=p.name, pos=p.pos, team=str(tid),
                    headshot=p.headshot, last=last_seen.get(aid, ""),
                    minutes=round(proj["minutes"], 1),
                    mk={m: [round(proj[m]["mean"], 2),
                            round(proj[m]["sd"] ** 2, 2)] for m in MARKETS}))
        entry["players"].sort(key=lambda r: -r["mk"]["pts"][0])
        slate.append(entry)

    # game logs for every rostered player (charts, hit rates, splits)
    logs = {}
    played = df[df["played"]].sort_values("game_date")
    keep = {aid for ids in roster.values() for aid in ids}
    for aid, rows in played.groupby("athlete_id"):
        if aid not in keep:
            continue
        rows = rows.tail(LOG_GAMES)
        p = engine.players[aid]
        logs[str(aid)] = dict(
            name=p.name, pos=p.pos, team=str(p.team_id),
            headshot=p.headshot, last=last_seen.get(aid, ""),
            games=[[r.game_date.strftime("%y-%m-%d"), int(r.season),
                    team_abbr.get(str(r.opponent_team_id),
                                  str(r.opponent_team_abbreviation)),
                    1 if r.home else 0, round(float(r.minutes), 1),
                    int(r.pts), int(r.reb), int(r.ast),
                    int(r.fg3m), int(r.stl), int(r.blk)]
                   for r in rows.itertuples()])

    # matchup context: allowed factor per team/stat/position (1.0 = league avg)
    defense = {}
    for tid in elo["teams"]:
        d = {}
        for stat in BASE_STATS:
            d[stat] = {pos: round(engine.defense_factor(str(tid), stat, pos), 3)
                       for pos in "GFC"}
        defense[str(tid)] = d

    disp = {m: round(engine.dispersion(m), 3) for m in MARKETS}
    return dict(season=current_season, slate=slate, logs=logs,
                defense=defense, dispersion=disp,
                markets=MARKETS, labels=MARKET_LABELS)


def inject(payload: dict) -> None:
    data_js = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")
    html = TEMPLATE.read_text().replace("__DATA_JSON__", data_js)
    DOCS.mkdir(exist_ok=True)
    (DOCS / "index.html").write_text(html)
    print(f"built docs/index.html ({len(html) // 1024} KiB)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-backtest", action="store_true",
                    help="reuse backtest numbers already in data/predictions.json")
    args = ap.parse_args()

    games = load_games()
    elo, _ = build_elo(games)
    props = build_props(games, elo)

    bt_props = None
    if args.skip_backtest and DATA_OUT.exists():
        bt_props = json.loads(DATA_OUT.read_text()).get("backtest")
    if not bt_props:
        bt_props = backtest_props.run()

    payload = dict(
        generated=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        elo=elo, props=props, backtest=bt_props)

    DATA_OUT.write_text(json.dumps(payload))
    print(f"wrote {DATA_OUT} ({DATA_OUT.stat().st_size // 1024} KiB): "
          f"{len(props['slate'])} slate games, {len(props['logs'])} players")
    inject(payload)


if __name__ == "__main__":
    main()
