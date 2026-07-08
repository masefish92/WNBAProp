"""MOV-adjusted Elo rating model for WNBA game prediction.

The approach follows the well-established FiveThirtyEight NBA Elo design,
re-tuned for the WNBA via backtesting (see backtest.py):

* every team starts at 1500 (expansion teams at 1300)
* between seasons ratings are regressed toward the league mean
* the winner takes K * mov_multiplier Elo points from the loser, where the
  margin-of-victory multiplier damps blowouts by heavy favorites
* home teams get a fixed Elo bonus (skipped on neutral courts)
* a small bonus/penalty is applied for rest-day differential
* Elo difference converts to a point spread and a win probability

Scoring-rate tracking (exponentially weighted points for/against) is kept
alongside Elo to project game totals.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

MEAN_RATING = 1500.0
EXPANSION_RATING = 1300.0

# Tuned via backtest.py grid search on 2015-2025 seasons.
DEFAULTS = dict(
    k=32.0,               # update speed
    home_adv=50.0,        # Elo points of home-court advantage (~1.9 pts);
                          # 50 leaves home-win rate unbiased (57% pred vs 56% actual)
    carryover=0.60,       # fraction of (rating - mean) kept between seasons
    elo_per_point=26.0,   # Elo difference equivalent to 1 point of spread
    rest_bonus=12.0,      # Elo points per day of rest advantage (capped)
    rest_cap=3.0,         # cap on rest-day differential considered
)


def load_games(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """All real WNBA games (regular season + playoffs), one row per game."""
    frames = []
    for path in sorted(data_dir.glob("wnba_schedule_*.parquet")):
        df = pd.read_parquet(
            path,
            columns=[
                "game_id", "season", "season_type", "date", "neutral_site",
                "status_type_completed", "status_type_name",
                "home_id", "home_display_name", "home_abbreviation",
                "home_logo", "home_color", "home_alternate_color", "home_score",
                "away_id", "away_display_name", "away_abbreviation",
                "away_logo", "away_color", "away_alternate_color", "away_score",
                "venue_full_name", "venue_address_city",
            ],
        )
        frames.append(df)
    games = pd.concat(frames, ignore_index=True)
    games["date"] = pd.to_datetime(games["date"], utc=True)
    games["home_score"] = pd.to_numeric(games["home_score"], errors="coerce")
    games["away_score"] = pd.to_numeric(games["away_score"], errors="coerce")

    # Drop All-Star and other exhibition matchups: real franchises appear in
    # many games per season, All-Star squads (Team Wilson, TBD, ...) do not.
    counts = pd.concat([games["home_id"], games["away_id"]]).value_counts()
    real = set(counts[counts >= 20].index)
    games = games[games["home_id"].isin(real) & games["away_id"].isin(real)]

    games = games.sort_values(["date", "game_id"]).reset_index(drop=True)
    games["completed"] = games["status_type_completed"].astype(bool) & games[
        "home_score"
    ].notna() & (games["home_score"] + games["away_score"] > 0)
    return games


@dataclass
class TeamState:
    rating: float = MEAN_RATING
    last_game: pd.Timestamp | None = None
    season: int | None = None
    # exponentially weighted scoring rates for total-points projection
    pts_for: float = 81.0
    pts_against: float = 81.0
    games_played: int = 0


@dataclass
class EloModel:
    k: float = DEFAULTS["k"]
    home_adv: float = DEFAULTS["home_adv"]
    carryover: float = DEFAULTS["carryover"]
    elo_per_point: float = DEFAULTS["elo_per_point"]
    rest_bonus: float = DEFAULTS["rest_bonus"]
    rest_cap: float = DEFAULTS["rest_cap"]
    score_alpha: float = 0.12  # EWMA weight for scoring rates
    # League scoring-environment tracker: slow EWMA of the totals residual.
    # Team EWMAs lag when the whole league's scoring level shifts (e.g. the
    # 2026 pace jump left raw totals ~4 pts low); this absorbs the shift.
    resid_alpha: float = 0.03
    total_resid: float = 0.0
    teams: dict = field(default_factory=dict)

    def team(self, team_id: str) -> TeamState:
        if team_id not in self.teams:
            self.teams[team_id] = TeamState(rating=EXPANSION_RATING)
        return self.teams[team_id]

    def _new_season(self, state: TeamState, season: int) -> None:
        if state.season is not None and season > state.season:
            state.rating = MEAN_RATING + self.carryover * (state.rating - MEAN_RATING)
        state.season = season

    def _rest_days(self, state: TeamState, date: pd.Timestamp) -> float:
        if state.last_game is None:
            return self.rest_cap
        return min((date - state.last_game).total_seconds() / 86400.0, 10.0)

    def pregame(self, home_id: str, away_id: str, date: pd.Timestamp,
                season: int, neutral: bool = False) -> dict:
        """Prediction for a game, WITHOUT updating ratings."""
        home, away = self.team(home_id), self.team(away_id)
        self._new_season(home, season)
        self._new_season(away, season)

        diff = home.rating - away.rating
        if not neutral:
            diff += self.home_adv
        rest_edge = min(self._rest_days(home, date), self.rest_cap) - min(
            self._rest_days(away, date), self.rest_cap
        )
        diff += self.rest_bonus * rest_edge

        win_prob = 1.0 / (1.0 + 10.0 ** (-diff / 400.0))
        spread = diff / self.elo_per_point  # positive = home favored

        # Total: blend each team's offense with the opponent's defense, then
        # shift by the league-environment residual tracker.
        home_exp = (home.pts_for + away.pts_against) / 2.0
        away_exp = (away.pts_for + home.pts_against) / 2.0
        total_raw = home_exp + away_exp
        total = total_raw + self.total_resid

        return dict(
            home_rating=home.rating,
            away_rating=away.rating,
            elo_diff=diff,
            win_prob_home=win_prob,
            spread_home=spread,
            total=total,
            total_raw=total_raw,
            home_pts=(total + spread) / 2.0,
            away_pts=(total - spread) / 2.0,
        )

    def update(self, home_id: str, away_id: str, date: pd.Timestamp,
               season: int, home_score: float, away_score: float,
               neutral: bool = False) -> dict:
        """Score a completed game and update ratings. Returns the pregame view."""
        pre = self.pregame(home_id, away_id, date, season, neutral)
        home, away = self.team(home_id), self.team(away_id)

        margin = home_score - away_score
        home_won = 1.0 if margin > 0 else 0.0
        # FiveThirtyEight MOV multiplier: damp blowouts by big favorites.
        winner_diff = pre["elo_diff"] if margin > 0 else -pre["elo_diff"]
        mov_mult = (abs(margin) + 3.0) ** 0.8 / (7.5 + 0.006 * winner_diff)
        shift = self.k * mov_mult * (home_won - pre["win_prob_home"])

        home.rating += shift
        away.rating -= shift

        self.total_resid = (1 - self.resid_alpha) * self.total_resid + \
            self.resid_alpha * ((home_score + away_score) - pre["total_raw"])

        a = self.score_alpha
        home.pts_for = (1 - a) * home.pts_for + a * home_score
        home.pts_against = (1 - a) * home.pts_against + a * away_score
        away.pts_for = (1 - a) * away.pts_for + a * away_score
        away.pts_against = (1 - a) * away.pts_against + a * home_score

        home.last_game = away.last_game = date
        home.games_played += 1
        away.games_played += 1
        return pre


def run_history(games: pd.DataFrame, model: EloModel | None = None,
                collect_from: int = 0) -> tuple[EloModel, pd.DataFrame]:
    """Replay all completed games chronologically through the model.

    Returns the fitted model and a frame of pregame predictions vs outcomes
    for seasons >= collect_from (used for backtesting/accuracy reporting).
    """
    model = model or EloModel()
    rows = []
    for g in games[games["completed"]].itertuples():
        pre = model.update(
            g.home_id, g.away_id, g.date, int(g.season),
            g.home_score, g.away_score, bool(g.neutral_site),
        )
        if int(g.season) >= collect_from:
            rows.append(dict(
                game_id=g.game_id, season=int(g.season), date=g.date,
                home_id=g.home_id, away_id=g.away_id,
                home=g.home_display_name, away=g.away_display_name,
                home_score=g.home_score, away_score=g.away_score,
                win_prob_home=pre["win_prob_home"],
                spread_home=pre["spread_home"], total=pre["total"],
                margin=g.home_score - g.away_score,
                actual_total=g.home_score + g.away_score,
            ))
    return model, pd.DataFrame(rows)
