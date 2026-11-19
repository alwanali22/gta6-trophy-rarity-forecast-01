# GTA VI Trophy Rarity Forecast

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/alwanali22/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/alwanali22/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

A forecaster for **make.fun gta6-trophy-rarity-forecast-01**: for every
PlayStation 5 trophy of Grand Theft Auto VI, five quantiles (5%, 25%, 50%,
75%, 95%) of its Exophase earn rate at the case's target — one week after
launch for the final case.

## The model

Every observed earn rate is split into the two things that move it: the
count of players who earned the trophy, E(t), and the count of players
Exophase tracks, N(t). Each is fitted on log time, forecast to the target,
and the median rate is E_forecast / N_forecast.

- **Dilution is the story.** On launch night Exophase refreshed once,
  638 → 3,868 tracked players, and every rate but the 100% one fell. A
  model that extrapolates the rate itself cannot see a refresh coming; a
  model that forecasts E and N separately can. When the history is flat
  (no refresh seen yet) the model assumes one is pending and sizes N by
  that first refresh's ratio, 3868/638, instead of pretending N stays put.
- **Shrinkage.** Each trophy's log-E slope is shrunk toward its group's
  mean (story, strand, activity, collectible, platinum) — heavily when
  its earned count is small and noisy.
- **Story chain monotone.** Ch1 ≥ Ch2 ≥ Ch3 ≥ Ch4 ≥ final mission, by
  pool-adjacent-violators with minimal adjustment.
- **Honest spread.** Random walk + fit residual + a refresh term for
  Exophase's own denominator jumps; the platinum and 100%-completion
  trophies get a longer upside tail. Gone Fishing (100% of tracked
  players) is forecast at ~99.9%, not diluted — its earners grow with N.

## Run it

Python 3.13, standard library only, no network, deterministic:

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Reads every `*.in` in the input directory, writes one `*.out` per case:
`{"trophies": [{"id": n, "q": [q05, q25, q50, q75, q95]}]}`.

## Backtest (practice cases p00/p01, proxy answers)

| Forecast | Pinball loss |
|---|---|
| House baseline (persistence) | 1.69 |
| Median persistence + wide spread (σ=1.5) | 0.97 |
| **This model** | **0.25** |

Pseudo-final (p01 history → live 09:11 UTC rates): loss 0.29 vs baseline
1.42; median falls for 54/54 trophies (all but Gone Fishing). Deterministic
(two runs byte-identical), ~0.05 s per case. Full numbers in
`BUILD-REPORT-2026-11-19.md` (in the workdir, not shipped in this repo).

## License

MIT — see [LICENSE](LICENSE).
