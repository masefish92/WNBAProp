"""Player prop projection engine.

Projects per-game player stat lines (points, rebounds, assists, threes,
steals, blocks and their combos) and converts them into over/under
probabilities for any betting line.

The engine is strictly walk-forward: it consumes completed games in
chronological order and can be asked for a projection at any point, which
uses only information available before tip-off. The same code path drives
both the historical backtest (backtest_props.py) and the live site build
(build_site.py), so backtest numbers are honest out-of-sample estimates
of live performance.

Projection recipe for player p, market m, against defense d:

    mean = projected_minutes(p)          # decayed EWMA of recent minutes
         * per_minute_rate(p, m)         # minutes-weighted decayed EWMA,
                                         #   shrunk toward the league rate
         * defense_factor(d, m, pos(p))  # opponent's allowed rate vs league
                                         #   for p's position group, shrunk
         * home_away_factor(m)           # small league-wide venue effect

    P(over line) from Normal(mean, sqrt(k_m * mean)) with a continuity
    correction; the dispersion k_m per market is itself estimated online
    from the walk's own out-of-sample errors, so probabilities stay
    calibrated without any fitting step.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Base stats read straight from the box score.
BASE_STATS = ["pts", "reb", "ast", "fg3m", "stl", "blk"]
# Combo markets and their components.
COMBOS = {
    "pra": ("pts", "reb", "ast"),
    "pr": ("pts", "reb"),
    "pa": ("pts", "ast"),
    "ra": ("reb", "ast"),
}
MARKETS = BASE_STATS + list(COMBOS)

MARKET_LABELS = {
    "pts": "Points", "reb": "Rebounds", "ast": "Assists",
    "fg3m": "3-Pointers Made", "stl": "Steals", "blk": "Blocks",
    "pra": "Pts+Reb+Ast", "pr": "Pts+Reb", "pa": "Pts+Ast", "ra": "Reb+Ast",
}

# --- tuning constants (validated by backtest_props.py) ---------------------
MIN_HALFLIFE = 5.0       # games; decay for the minutes projection
RATE_HALFLIFE = 18.0     # games; decay for per-minute scoring rates
SHRINK_MINUTES = 130.0   # effective minutes of league-prior in player rates
DEF_HALFLIFE = 15.0      # games; decay for team defensive factors
DEF_SHRINK_GAMES = 12.0  # games of neutral prior in defensive factors
DEF_POS_SHRINK = 20.0    # extra shrink of position split toward team factor
DEF_BETA = 0.7           # partial application of the defensive factor
SEASON_TURNOVER = 0.35   # carryover of decayed sums across seasons
MIN_GAMES = 5            # played games required before projecting a player
MIN_MINUTES = 8.0        # projected minutes required to emit props
DISP_PRIOR = {           # initial k in var = k * mean, refined online
    "pts": 2.1, "reb": 1.7, "ast": 1.5, "fg3m": 1.25, "stl": 1.1, "blk": 1.1,
    "pra": 2.6, "pr": 2.2, "pa": 2.3, "ra": 1.9,
}


def load_player_box(data_dir: Path = DATA_DIR, first_season: int = 2018) -> pd.DataFrame:
    """One row per player-game, chronological, real games only."""
    frames = []
    for path in sorted(data_dir.glob("player_box_*.parquet")):
        season = int(path.stem.split("_")[-1])
        if season < first_season:
            continue
        frames.append(pd.read_parquet(path))
    df = pd.concat(frames, ignore_index=True)
    df = df[df["season_type"].isin([2, 3])]  # regular season + playoffs

    df = df.rename(columns={
        "points": "pts", "rebounds": "reb", "assists": "ast",
        "three_point_field_goals_made": "fg3m", "steals": "stl",
        "blocks": "blk", "turnovers": "tov",
        "field_goals_made": "fgm", "field_goals_attempted": "fga",
        "free_throws_made": "ftm", "free_throws_attempted": "fta",
        "three_point_field_goals_attempted": "fg3a",
    })
    num_cols = ["minutes", "pts", "reb", "ast", "fg3m", "stl", "blk", "tov",
                "fgm", "fga", "ftm", "fta", "fg3a"]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["game_date"] = pd.to_datetime(df["game_date"])
    df["played"] = (
        (~df["did_not_play"].fillna(False).astype(bool))
        & df["minutes"].notna() & (df["minutes"] > 0)
    )
    df["home"] = df["home_away"].astype(str).str.lower().eq("home")
    df["pos"] = (
        df["athlete_position_abbreviation"].astype(str).str.upper().str[-1]
        .map({"G": "G", "F": "F", "C": "C"}).fillna("F")
    )
    df = df.sort_values(["game_date", "game_id", "team_id"]).reset_index(drop=True)
    return df


# --------------------------------------------------------------------------
# online state

@dataclass
class PlayerState:
    name: str = ""
    pos: str = "F"
    team_id: str = ""
    headshot: str = ""
    # decayed sums; age is measured in that player's own played games
    w_min: float = 0.0            # sum of decay weights (minutes projection)
    s_min: float = 0.0            # decayed sum of minutes
    w_rate: float = 0.0           # decayed sum of minutes (rate denominator)
    s_stat: dict = field(default_factory=lambda: defaultdict(float))
    games: int = 0
    last_date: pd.Timestamp | None = None
    season: int | None = None

    def decay_game(self) -> None:
        dm = 0.5 ** (1.0 / MIN_HALFLIFE)
        dr = 0.5 ** (1.0 / RATE_HALFLIFE)
        self.w_min *= dm
        self.s_min *= dm
        self.w_rate *= dr
        for k in self.s_stat:
            self.s_stat[k] *= dr


@dataclass
class DefenseState:
    """Decayed allowed-stat totals, split by opposing position group."""
    w_min: dict = field(default_factory=lambda: defaultdict(float))
    s_stat: dict = field(default_factory=lambda: defaultdict(float))
    games: float = 0.0
    season: int | None = None

    def decay_game(self) -> None:
        d = 0.5 ** (1.0 / DEF_HALFLIFE)
        self.games = self.games * d + 1.0
        for k in self.w_min:
            self.w_min[k] *= d
        for k in self.s_stat:
            self.s_stat[k] *= d


class PropEngine:
    def __init__(self) -> None:
        self.players: dict[str, PlayerState] = {}
        self.defense: dict[str, DefenseState] = {}
        # league per-minute rates by (stat, pos) and overall, slow decay
        self.lg_min: dict = defaultdict(float)
        self.lg_stat: dict = defaultdict(float)
        # league home/away per-minute rate sums per stat
        self.venue: dict = defaultdict(float)
        # online dispersion: decayed sums of (err^2 / mean) per market
        self.disp_n: dict = defaultdict(float)
        self.disp_s: dict = defaultdict(float)

    # -- helpers ------------------------------------------------------------

    def _season_rollover(self, obj, season: int) -> None:
        if obj.season is not None and season > obj.season:
            t = SEASON_TURNOVER
            if isinstance(obj, PlayerState):
                obj.w_min *= t; obj.s_min *= t; obj.w_rate *= t
                for k in obj.s_stat:
                    obj.s_stat[k] *= t
            else:
                obj.games *= t
                for k in obj.w_min:
                    obj.w_min[k] *= t
                for k in obj.s_stat:
                    obj.s_stat[k] *= t
        obj.season = season

    def league_rate(self, stat: str, pos: str | None = None) -> float:
        if pos and self.lg_min[("pos", pos)] > 5000:
            return self.lg_stat[(stat, pos)] / self.lg_min[("pos", pos)]
        total_min = self.lg_min["all"]
        return self.lg_stat[(stat, "all")] / total_min if total_min > 0 else 0.0

    def defense_factor(self, team_id: str, stat: str, pos: str) -> float:
        d = self.defense.get(team_id)
        if d is None or d.games < 2:
            return 1.0
        lg_pos = self.league_rate(stat, pos)
        lg_all = self.league_rate(stat)
        if lg_all <= 0:
            return 1.0
        # team-overall allowed factor, shrunk toward 1
        mins_all = sum(d.w_min[p] for p in "GFC")
        stat_all = sum(d.s_stat[(stat, p)] for p in "GFC")
        rate_all = stat_all / mins_all if mins_all > 0 else lg_all
        f_team = (rate_all / lg_all * d.games + DEF_SHRINK_GAMES) / (
            d.games + DEF_SHRINK_GAMES)
        # position split, shrunk toward the team factor
        if lg_pos > 0 and d.w_min[pos] > 0:
            rate_pos = d.s_stat[(stat, pos)] / d.w_min[pos]
            g_pos = d.games * d.w_min[pos] / max(mins_all, 1.0)
            f_pos = (rate_pos / lg_pos * g_pos + f_team * DEF_POS_SHRINK) / (
                g_pos + DEF_POS_SHRINK)
        else:
            f_pos = f_team
        return f_pos

    def venue_factor(self, stat: str, home: bool) -> float:
        h, a = self.venue[(stat, True)], self.venue[(stat, False)]
        mh, ma = self.venue[("min", True)], self.venue[("min", False)]
        if min(mh, ma) < 20000:  # not enough evidence yet
            return 1.0
        rh, ra = h / mh, a / ma
        avg = (rh + ra) / 2.0
        if avg <= 0:
            return 1.0
        return (rh if home else ra) / avg

    def dispersion(self, market: str) -> float:
        prior_n = 300.0
        n, s = self.disp_n[market], self.disp_s[market]
        return (s + DISP_PRIOR[market] * prior_n) / (n + prior_n)

    # -- projection ----------------------------------------------------------

    def projected_minutes(self, p: PlayerState) -> float:
        return p.s_min / p.w_min if p.w_min > 0 else 0.0

    def project(self, athlete_id: str, opponent_id: str, home: bool,
                season: int | None = None) -> dict | None:
        """Pregame projection for every market; None if not projectable."""
        p = self.players.get(athlete_id)
        if p is None or p.games < MIN_GAMES or p.w_rate <= 0:
            return None
        mins = self.projected_minutes(p)
        if mins < MIN_MINUTES:
            return None

        out = {"minutes": mins}
        means = {}
        for stat in BASE_STATS:
            lg = self.league_rate(stat, p.pos)
            raw = p.s_stat[stat] / p.w_rate
            rate = (p.s_stat[stat] + lg * SHRINK_MINUTES) / (
                p.w_rate + SHRINK_MINUTES)
            dfac = self.defense_factor(opponent_id, stat, p.pos) ** DEF_BETA
            vfac = self.venue_factor(stat, home)
            means[stat] = max(mins * rate * dfac * vfac, 0.0)
            _ = raw
        for combo, parts in COMBOS.items():
            means[combo] = sum(means[s] for s in parts)
        for m in MARKETS:
            out[m] = {"mean": means[m], "sd": math.sqrt(
                max(self.dispersion(m) * means[m], 0.05))}
        return out

    @staticmethod
    def prob_over(mean: float, sd: float, line: float) -> float:
        """P(stat > line) for an integer-valued, right-skewed stat.

        Uses a negative binomial with the given mean and variance (sd^2);
        box-score stats are overdispersed counts, and the NB's right skew
        is what keeps P(over) honest near the mean (a symmetric normal
        overstates it, because the median sits below the mean). Falls back
        to Poisson when variance <= mean and to a normal tail for large
        means where the discrete sum is unnecessary.
        """
        need = math.floor(line) + 1  # smallest count that clears the line
        if need <= 0:
            return 0.99
        var = sd * sd
        if mean <= 0.01:
            return 0.01
        if need > 400:
            z = (need - 0.5 - mean) / sd
            return min(max(0.5 * math.erfc(z / math.sqrt(2.0)), 0.01), 0.99)
        if var > mean * 1.02:
            # NB: pmf(0) = (r/(r+mu))^r, pmf(k+1) = pmf(k)*(k+r)/(k+1)*q
            r = mean * mean / (var - mean)
            q = mean / (mean + r)
            log_p0 = r * math.log(1.0 - q)
            pmf = math.exp(log_p0)
            cdf = pmf
            for k in range(need - 1):
                pmf *= (k + r) / (k + 1) * q
                cdf += pmf
        else:
            pmf = math.exp(-mean)
            cdf = pmf
            for k in range(need - 1):
                pmf *= mean / (k + 1)
                cdf += pmf
        return min(max(1.0 - cdf, 0.01), 0.99)

    # -- state updates -------------------------------------------------------

    def observe(self, rows: pd.DataFrame, update_dispersion: bool = True) -> None:
        """Feed one completed game (all player rows for both teams)."""
        by_team = dict(tuple(rows.groupby("team_id", sort=False)))
        team_ids = list(by_team)
        season = int(rows["season"].iloc[0])
        date = rows["game_date"].iloc[0]

        # dispersion updates use this game's pregame projections
        if update_dispersion:
            for r in rows.itertuples():
                if not r.played:
                    continue
                proj = self.project(r.athlete_id, r.opponent_team_id, r.home)
                if proj is None:
                    continue
                actuals = self._actuals(r)
                for m in MARKETS:
                    mean = proj[m]["mean"]
                    if mean > 0.3:
                        self.disp_n[m] = self.disp_n[m] * 0.9995 + 1.0
                        self.disp_s[m] = self.disp_s[m] * 0.9995 + \
                            (actuals[m] - mean) ** 2 / mean

        # defense + league + venue state
        for tid in team_ids:
            opp = [t for t in team_ids if t != tid]
            if not opp:
                continue
            opp_rows = by_team[opp[0]]
            d = self.defense.setdefault(tid, DefenseState())
            self._season_rollover(d, season)
            d.decay_game()
            for r in opp_rows.itertuples():
                if not r.played:
                    continue
                d.w_min[r.pos] += r.minutes
                for stat in BASE_STATS:
                    d.s_stat[(stat, r.pos)] += getattr(r, stat) or 0.0

        lg_decay = 0.9995
        for k in list(self.lg_min):
            self.lg_min[k] *= lg_decay
        for k in list(self.lg_stat):
            self.lg_stat[k] *= lg_decay
        for k in list(self.venue):
            self.venue[k] *= lg_decay

        for r in rows.itertuples():
            if not r.played:
                continue
            self.lg_min["all"] += r.minutes
            self.lg_min[("pos", r.pos)] += r.minutes
            self.venue[("min", r.home)] += r.minutes
            for stat in BASE_STATS:
                v = getattr(r, stat) or 0.0
                self.lg_stat[(stat, "all")] += v
                self.lg_stat[(stat, r.pos)] += v
                self.venue[(stat, r.home)] += v

        # player state
        for r in rows.itertuples():
            p = self.players.setdefault(r.athlete_id, PlayerState())
            p.name = r.athlete_display_name
            p.pos = r.pos
            p.team_id = r.team_id
            hs = getattr(r, "athlete_headshot_href", "")
            p.headshot = hs if isinstance(hs, str) else ""
            if not r.played:
                continue
            self._season_rollover(p, season)
            p.decay_game()
            p.w_min += 1.0
            p.s_min += r.minutes
            p.w_rate += r.minutes
            for stat in BASE_STATS:
                p.s_stat[stat] += getattr(r, stat) or 0.0
            p.games += 1
            p.last_date = date

    @staticmethod
    def _actuals(r) -> dict:
        a = {s: (getattr(r, s) or 0.0) for s in BASE_STATS}
        for combo, parts in COMBOS.items():
            a[combo] = sum(a[s] for s in parts)
        return a


def walk(df: pd.DataFrame, engine: PropEngine | None = None,
         emit=None, emit_from: int = 0) -> PropEngine:
    """Replay completed games chronologically through the engine.

    ``emit(game_rows, projections)`` is called per game before the engine
    sees the outcome, where projections is {athlete_id: (proj, actuals)}.
    """
    engine = engine or PropEngine()
    for _, rows in df.groupby("game_id", sort=False):
        if emit is not None and int(rows["season"].iloc[0]) >= emit_from:
            projs = {}
            for r in rows.itertuples():
                if not r.played:
                    continue
                pr = engine.project(r.athlete_id, r.opponent_team_id, r.home)
                if pr is not None:
                    projs[r.athlete_id] = (pr, engine._actuals(r))
            if projs:
                emit(rows, projs)
        engine.observe(rows)
    return engine
