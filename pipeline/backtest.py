"""Backtest & parameter tuning for the WNBA Elo model.

Replays history chronologically; every prediction is made strictly before
the game's result is fed to the model, so there is no lookahead. Evaluated
on seasons >= EVAL_FROM to give the model a warm-up period.

Metrics:
  * accuracy  - % of games where the pregame favorite won
  * brier     - mean squared error of the home win probability
  * log_loss  - negative log likelihood
  * spread_mae- mean absolute error of predicted vs actual margin
  * total_mae - mean absolute error of predicted vs actual game total
"""

from __future__ import annotations

import itertools
import sys

import numpy as np
import pandas as pd

from model import EloModel, load_games, run_history

EVAL_FROM = 2015


def evaluate(preds: pd.DataFrame) -> dict:
    p = preds["win_prob_home"].clip(1e-6, 1 - 1e-6)
    y = (preds["margin"] > 0).astype(float)
    picked_right = ((p > 0.5) == (y == 1)).mean()
    return dict(
        n=len(preds),
        accuracy=picked_right,
        brier=((p - y) ** 2).mean(),
        log_loss=-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean(),
        spread_mae=(preds["spread_home"] - preds["margin"]).abs().mean(),
        total_mae=(preds["total"] - preds["actual_total"]).abs().mean(),
    )


def run(games: pd.DataFrame, **params) -> dict:
    _, preds = run_history(games, EloModel(**params), collect_from=EVAL_FROM)
    return evaluate(preds)


def grid_search(games: pd.DataFrame) -> pd.DataFrame:
    grid = dict(
        k=[24, 28, 32, 40],
        home_adv=[60, 80, 100],
        carryover=[0.6, 0.7, 0.8],
        rest_bonus=[0, 12, 25],
    )
    rows = []
    for combo in itertools.product(*grid.values()):
        params = dict(zip(grid.keys(), combo))
        metrics = run(games, **params)
        rows.append({**params, **metrics})
        print(f"{params} -> brier={metrics['brier']:.4f} "
              f"acc={metrics['accuracy']:.3f} sMAE={metrics['spread_mae']:.2f}")
    return pd.DataFrame(rows).sort_values("log_loss")


def per_season(games: pd.DataFrame, **params) -> pd.DataFrame:
    _, preds = run_history(games, EloModel(**params), collect_from=EVAL_FROM)
    return preds.groupby("season").apply(
        lambda s: pd.Series(evaluate(s)), include_groups=False
    )


if __name__ == "__main__":
    games = load_games()
    if len(sys.argv) > 1 and sys.argv[1] == "grid":
        results = grid_search(games)
        print("\nTop 10 by log loss:")
        print(results.head(10).to_string(index=False))
    else:
        print("Defaults, per season:")
        print(per_season(games).round(3).to_string())
        print("\nOverall:")
        print({k: round(v, 4) for k, v in run(games).items()})
