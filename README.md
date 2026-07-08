# WNBA Prop Lab

Personal analytics site for WNBA player-prop betting: projections,
over/under probabilities, hit rates, matchup context, and honest
walk-forward backtests. Rebuilt automatically from
[wehoop](https://github.com/sportsdataverse/wehoop-wnba-data) (ESPN) data.

**The site is one static file: [`docs/index.html`](docs/index.html).**
Open it locally or serve it with GitHub Pages
(Settings → Pages → deploy from branch → `/docs`).

## What's inside

| View | What it does |
|---|---|
| **Slate** | Upcoming games with Elo spread / total / win prob and each team's top projected players |
| **Player** | Pick any player + market + line + odds → projection, P(over/under), EV per $1, ¼-Kelly stake, hit-rate splits (L5/L10/L20/season/home/away/vs-opponent), game-log chart vs the line, minutes trend, opponent position-defense rank |
| **Screener** | Every projected player on the slate, ranked by model edge for any market |
| **Games** | Elo power ratings and recent results vs model expectations |
| **Model** | The honesty tab: 5+ season walk-forward backtest vs naive baselines, probability calibration, confidence-bucket win rates |

## How the models work

**Props** — strictly walk-forward engine (`pipeline/props.py`):

```
projection = projected minutes            (recency-weighted EWMA)
           × per-minute rate              (minutes-weighted EWMA, shrunk to league)
           × opponent allowed-rate factor (per stat & position group, shrunk)
           × home/away factor
P(over)    = negative binomial(mean, dispersion fit online per market)
```

Markets: PTS, REB, AST, 3PM, STL, BLK, PRA, PR, PA, RA.

**Games** — MOV-adjusted Elo (`pipeline/model.py`), FiveThirtyEight-style,
retuned for the WNBA: rest bonus, home-court, season carryover, EWMA
scoring rates for totals.

**Backtests** — `pipeline/backtest_props.py` replays 2021–present (2018+
warm-up) making every projection before the game is scored: ~250K
projections. The projection beats L5/L10/season-average baselines on MAE
in every market, and predicted over-probabilities are calibrated to within
~1–2 points. `pipeline/backtest.py` does the same for Elo (66%+ winner
accuracy since 2015). Numbers are shown on the site's Model tab and
regenerate with every build — if the model degrades, the site will say so.

## Refreshing data / rebuilding

```bash
pip install -r requirements.txt
python pipeline/fetch_data.py        # pull latest wehoop parquet files
python pipeline/build_site.py        # rebuild docs/index.html (runs backtest)
python pipeline/backtest_props.py    # prop backtest report only
python pipeline/backtest.py          # Elo backtest report only
```

`.github/workflows/update.yml` does this automatically twice a day
(before and after the daily slate) and commits the result — enable
Actions on the repo and the site stays current by itself.

## Caveats (read before betting)

* The model doesn't know about **injuries, rest, or lineup news** — always
  check availability before placing anything.
* Backtest "win rates" are measured against **synthetic lines** (L10
  average, floored to x.5), not real closing lines. Real books are
  sharper. The calibration chart is the trustworthy signal.
* ESPN minutes/stat corrections occasionally revise a box score after the
  fact; numbers refresh with the next data pull.

Personal research tool — not betting advice.
