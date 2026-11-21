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
- **Honest spread — branch mixture (final refit).** Exophase refreshes
  in jumps, so the predictive distribution is three branches centred
  at logit(median) ± J and logit(median), J the typical single-refresh
  |Δ ln N| (prior ln(3868/638) = 1.80), weights (0.10, 0.80, 0.10).
  The outer quantiles still reach the no-refresh / double-refresh
  branches ~1.8 logit apart, but the inner quantiles sit inside the
  dominant branch instead of being smeared across both. Within a
  branch the logit scale is 0.9 × sqrt(random-walk² + residual²).
  A 0.00% last rate is treated as censored (half an earner), not zero.
- **Extrapolation guard (final refit).** A log-E power law fitted on a
  steep growth phase can explode — on a final-like smoke history built
  from the live 21 Nov rates it sent Ch1 to 99.99%. The median rate is
  capped at min(3× the last rate, 60%); the upper quantiles still
  reach above the cap, so the progression-wave case stays covered.
- **Fresh data (final refit).** Live snapshot 21 Nov 04:28 UTC: Ch1
  18.02%, Ch2 3.61%, Ch3 0.07%, platinum and 100% now 0.00%, fishing
  16.73%. The story chain is spread out, not flat — per-trophy slopes
  now matter, and shrinkage (K=20) correctly bites only where E is
  small. Platinum/100% medians belong at the 0.01% floor (guides put
  the story at ~80 h, the platinum at 200–300 h, vs 168 h to target).
- **Stale-level correction (refit v2).** When the visible history ends in
  a long flat stretch, the last counts are stale-low: players kept earning
  while Exophase did not refresh (no change at all 09:11 → 12:11 UTC
  19 Nov). E is corrected upward by the trophy group's observed growth
  during the last frozen window (1.2–3.4× per 6.7 h), capped per group.

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
| **This model (final refit, 21 Nov)** | **0.15** |

Pseudo-final (p01 history → live 09:11 UTC rates): loss 0.17 vs baseline
1.42; median falls for 54/54 trophies (all but Gone Fishing). Long-horizon
synthetic (logistic-E, 108 h): loss 0.69 vs baseline 0.82. Deterministic
(two runs byte-identical), ~0.04 s per case. Full numbers in
`REFIT-REPORT-2026-11-21.md` (in the workdir, not shipped in this repo).

## License

MIT — see [LICENSE](LICENSE).
