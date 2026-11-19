#!/usr/bin/env python3
"""GTA VI trophy rarity forecast — Exophase earn-rate quantiles.

Entry for make.fun gta6-trophy-rarity-forecast-01. Python 3.13 stdlib only,
no network, deterministic: the same input always yields the same bytes.

Model (per case, per trophy), in one paragraph: split every observed earn
rate into the two things that move it — the count of players who earned the
trophy, E(t), and the count of players Exophase tracks, N(t) — fit each on
log time, forecast both to the target, and take the median rate as
E_forecast / N_forecast. Slopes are shrunk toward their trophy-group mean
(heavy shrinkage when E is small, where the counts are noisy); the story
chain is forced monotone (Ch1 >= Ch2 >= Ch3 >= Ch4 >= final); the spread is
an honest sum of a random-walk term, the fit's residual variance, and a
refresh term for Exophase's own denominator jumps.

Why split E and N: on launch night Exophase refreshed once, 638 -> 3,868
tracked players, and every rate but the 100% one fell (RULES.txt). A model
that extrapolates the rate itself cannot see a refresh coming; a model that
forecasts E and N separately can — when the history is flat (no refresh
seen yet) it assumes one is pending and sizes N by that first refresh's
ratio, 3868/638, instead of pretending N stays put.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone

RELEASE = datetime(2026, 11, 19, tzinfo=timezone.utc)
Z_SCORES = (-1.645, -0.674, 0.0, 0.674, 1.645)

# Story chain, in Exophase id order: Ch1, Ch2, Ch3, Ch4, final mission.
CHAIN_IDS = [3, 4, 5, 6, 7]

# Trophy groups for slope shrinkage: members share a difficulty shape.
# ids follow Exophase's list order (see the case's own trophies array).
GROUP_OF = {}
for _i in (3, 4, 5, 6, 7):
    GROUP_OF[_i] = "story"
for _i in (8, 9, 10, 11, 12, 13, 22, 23, 24, 26, 27):
    GROUP_OF[_i] = "strand"
for _i in (15, 16, 17, 18, 19, 20, 21, 28, 29, 32, 33, 34, 36, 37,
           38, 39, 40, 41, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53):
    GROUP_OF[_i] = "activity"
for _i in (30, 31, 35, 42, 54, 55):
    GROUP_OF[_i] = "collect"
GROUP_OF[1] = "platinum"
GROUP_OF[14] = "platinum"
GROUP_OF[2] = "opener"

# First Exophase refresh of launch night: 638 -> 3,868 tracked players
# (RULES.txt). Used as the pending-refresh scenario when the history shows
# no refresh yet (N flat): a flat page is a stale page, not a stable one.
DILUTION_RATIO = 3868.0 / 638.0

# Shrinkage strength: a trophy whose last earned count is E keeps a weight
# of E / (E + SHRINK_K) on its own slope; the rest goes to its group mean.
SHRINK_K = 20.0
# Recency half-life (hours) for the least-squares weights.
HALF_LIFE_H = 12.0
# Spread pieces, all on the logit scale (see forecast_spread).
REFRESH_SIGMA_FLAT = 1.1   # no refresh seen yet: one is pending, size ~ln(6)
REFRESH_SIGMA_SEEN = 0.45  # refreshes seen: cadence known, jumps smaller
SPREAD_FLOOR = 0.30
SPREAD_CAP = 2.50
# Platinum / 100%-completion: the upside tail is longer than the downside
# (a few hardcore players can finish far sooner than expected).
UPPER_TAIL_IDS = (1, 14)
UPPER_TAIL_MULT = 1.35
LOWER_TAIL_MULT = 0.90


def parse_time(text):
    """ISO-8601 UTC timestamp -> aware datetime."""
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def hours_since_release(moment):
    """Hours from the 2026-11-19T00:00Z release to a read/target."""
    delta = (moment - RELEASE).total_seconds() / 3600.0
    return max(delta, -0.9)


def logit_pct(rate):
    """logit of a percent rate, clamped to the scorer's [0.01, 99.99]."""
    p = min(max(rate, 0.01), 99.99) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit_pct(logit_value):
    """Inverse logit -> percent, clamped to the scorer's range."""
    if logit_value >= 20.0:
        return 99.99
    if logit_value <= -20.0:
        return 0.01
    return 100.0 / (1.0 + math.exp(-logit_value))


def weighted_line(xs, ys, ws):
    """Weighted least squares y ~ a + b*x. Returns (slope, intercept)."""
    sw = sum(ws)
    if sw <= 0.0 or len(xs) < 2:
        return 0.0, (ys[-1] if ys else 0.0)
    mx = sum(w * x for w, x in zip(ws, xs)) / sw
    my = sum(w * y for w, y in zip(ws, ys)) / sw
    sxx = sum(w * (x - mx) ** 2 for w, x in zip(ws, xs))
    if sxx <= 1e-12:
        return 0.0, my
    sxy = sum(w * (x - mx) * (y - my) for w, x, y in zip(ws, xs, ys))
    slope = sxy / sxx
    return slope, my - slope * mx


def median_of(values):
    """Median of a non-empty list of floats."""
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    if n % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def pool_adjacent_decreasing(values):
    """Isotonic regression (decreasing) by pool-adjacent-violators.

    Returns the closest non-increasing sequence in the least-squares sense,
    used to keep the story-chain medians monotone with minimal adjustment.
    """
    blocks = [[v, 1] for v in values]
    i = 0
    while i + 1 < len(blocks):
        if blocks[i][0] < blocks[i + 1][0]:
            merged = blocks[i][0] * blocks[i][1] + blocks[i + 1][0] * blocks[i + 1][1]
            count = blocks[i][1] + blocks[i + 1][1]
            blocks[i:i + 2] = [[merged / count, count]]
            if i:
                i -= 1
        else:
            i += 1
    out = []
    for value, count in blocks:
        out.extend([value] * count)
    return out


def read_series(case):
    """Build per-trophy time series from a case's history.

    Returns (times, n_series, per-trophy series) where times are hours
    since release, n_series the tracked-player counts, and each trophy's
    series a list of (time, rate) with null rates dropped.
    """
    history = case.get("history") or []
    times = []
    n_series = []
    raw = {}
    for read in history:
        try:
            moment = parse_time(read["at"])
        except (KeyError, TypeError, ValueError):
            continue
        times.append(hours_since_release(moment))
        n_series.append(float(read.get("players_tracked") or 0) or None)
        rates = read.get("rates") or []
        for pos, rate in enumerate(rates):
            if rate is None:
                continue
            raw.setdefault(pos, []).append((times[-1], float(rate)))
    return times, n_series, raw


def forecast_n(times, n_series, target_t):
    """Forecast tracked players at the target hour.

    Power law: log N ~ a + b * ln(1 + t), fitted on the last 24 h of reads
    (recency-weighted). When N never moved, a refresh is pending: scale by
    the first launch-night refresh instead of pretending N stays put.
    Returns (n_forecast, n_flat) with n_flat True when N never changed.
    """
    pairs = [(t, n) for t, n in zip(times, n_series) if n and n > 0]
    if not pairs:
        return 1.0, True
    distinct = {n for _, n in pairs}
    n_last = pairs[-1][1]
    if len(distinct) < 2:
        return n_last * DILUTION_RATIO, True
    cutoff = pairs[-1][0] - 24.0
    window = [(t, n) for t, n in pairs if t >= cutoff]
    if len({n for _, n in window}) < 2:
        window = pairs
    t_last = pairs[-1][0]
    xs = [math.log1p(max(t, -0.9)) for t, _ in window]
    ys = [math.log(n) for _, n in window]
    ws = [0.5 ** ((t_last - t) / HALF_LIFE_H) for t, _ in window]
    slope, intercept = weighted_line(xs, ys, ws)
    slope = min(max(slope, 0.0), 2.0)
    x_target = math.log1p(max(target_t, -0.9))
    n_forecast = math.exp(intercept + slope * x_target)
    n_forecast = min(max(n_forecast, n_last), n_last * 100.0)
    return n_forecast, False


def forecast_trophy_median(series, n_pairs, n_forecast, target_t,
                            group_slope):
    """Median earn rate for one trophy at the target, via E/N split.

    series: [(time, rate)] for this trophy. n_pairs: [(time, N)].
    Fits log E ~ a + b * ln(1 + t) (E = rate * N / 100), shrinks the slope
    toward the trophy's group mean, anchors the level at the last read,
    and divides the forecast count by the forecast N.
    """
    if not series:
        return None
    t_last, rate_last = series[-1]
    n_last = None
    for t, n in reversed(n_pairs):
        if n and n > 0:
            n_last = n
            break
    if n_last is None:
        return None
    e_last = max(rate_last * n_last / 100.0, 0.5)
    xs, ys, ws = [], [], []
    for t, rate in series:
        n_at = None
        for tt, nn in n_pairs:
            if abs(tt - t) < 1e-9:
                n_at = nn
                break
        if not n_at:
            continue
        e_at = max(rate * n_at / 100.0, 0.5)
        xs.append(math.log1p(max(t, -0.9)))
        ys.append(math.log(e_at))
        ws.append(0.5 ** ((t_last - t) / HALF_LIFE_H))
    slope = 0.0
    resid_var = 0.0
    if len(xs) >= 2:
        slope, intercept = weighted_line(xs, ys, ws)
        resid = [y - (intercept + slope * x) for x, y in zip(xs, ys)]
        resid_var = sum(w * r * r for w, r in zip(ws, resid)) / max(sum(ws), 1e-9)
        resid_var = min(max(resid_var, 0.0), 4.0)
    weight = e_last / (e_last + SHRINK_K)
    slope = weight * slope + (1.0 - weight) * group_slope
    slope = min(max(slope, -0.05), 2.0)
    x_target = math.log1p(max(target_t, -0.9))
    x_last = math.log1p(max(t_last, -0.9))
    e_forecast = e_last * math.exp(slope * (x_target - x_last))
    e_forecast = max(e_forecast, e_last * 0.9)
    median = 100.0 * e_forecast / n_forecast
    return median, resid_var


def forecast_spread(horizon_days, resid_var, n_flat):
    """Honest logit spread: random walk + fit residual + refresh risk."""
    random_walk = 0.5 * math.sqrt(max(horizon_days, 0.0))
    refresh = REFRESH_SIGMA_FLAT if n_flat else REFRESH_SIGMA_SEEN
    var = random_walk ** 2 + resid_var + refresh ** 2
    return min(max(math.sqrt(var), SPREAD_FLOOR), SPREAD_CAP)


def quantiles_for(median, spread, trophy_id):
    """Five quantiles at levels .05/.25/.5/.75/.95 from a logit median."""
    centre = logit_pct(median)
    if trophy_id in UPPER_TAIL_IDS:
        spread_up = spread * UPPER_TAIL_MULT
        spread_down = spread * LOWER_TAIL_MULT
    else:
        spread_up = spread_down = spread
    out = []
    for z in Z_SCORES:
        s = spread_up if z > 0 else (spread_down if z < 0 else spread)
        value = inv_logit_pct(centre + z * s)
        out.append(round(min(max(value, 0.01), 99.99), 4))
    for k in range(1, len(out)):
        out[k] = max(out[k], out[k - 1])
    return out


def persistence_row(rate, horizon_days, n_flat):
    """Fallback row: last rate with the baseline random-walk spread."""
    spread = forecast_spread(horizon_days, 0.0, n_flat)
    return quantiles_for(rate, spread, 0)


def forecast_case(case):
    """Forecast one case -> {"trophies": [{"id": n, "q": [...]}]}."""
    trophies = case.get("trophies") or []
    horizon_days = float(case.get("horizon_minutes") or 0) / 1440.0
    target_t = hours_since_release(parse_time(case["target"]))
    times, n_series, raw = read_series(case)
    n_pairs = [(t, n) for t, n in zip(times, n_series) if n and n > 0]
    n_forecast, n_flat = forecast_n(times, n_series, target_t)

    # Group slopes first (median of member slopes on log E), for shrinkage.
    member_slopes = {}
    for pos, series in raw.items():
        trophy = trophies[pos] if pos < len(trophies) else {}
        tid = trophy.get("id", pos + 1)
        group = GROUP_OF.get(tid, "activity")
        xs, ys = [], []
        for t, rate in series:
            n_at = next((nn for tt, nn in n_pairs if abs(tt - t) < 1e-9), None)
            if not n_at:
                continue
            xs.append(math.log1p(max(t, -0.9)))
            ys.append(math.log(max(rate * n_at / 100.0, 0.5)))
        if len(xs) >= 2:
            slope, _ = weighted_line(xs, ys, [1.0] * len(xs))
            member_slopes.setdefault(group, []).append(slope)
    group_slopes = {g: median_of(v) for g, v in member_slopes.items() if v}
    global_slope = median_of([s for v in member_slopes.values() for s in v]) \
        if member_slopes else 0.0

    medians = {}
    resid_vars = {}
    tier_medians = {}
    for pos, trophy in enumerate(trophies):
        tid = trophy.get("id", pos + 1)
        series = raw.get(pos, [])
        if not series:
            medians[tid] = None
            resid_vars[tid] = 0.0
            continue
        group = GROUP_OF.get(tid, "activity")
        g_slope = group_slopes.get(group, global_slope)
        got = forecast_trophy_median(series, n_pairs, n_forecast,
                                     target_t, g_slope)
        if got is None:
            medians[tid] = None
            resid_vars[tid] = 0.0
            continue
        median, resid_var = got
        medians[tid] = min(max(median, 0.01), 99.99)
        resid_vars[tid] = resid_var
        tier_medians.setdefault(trophy.get("tier", "bronze"), []).append(medians[tid])

    # Trophies with no usable read fall back to their tier's median.
    for tid, value in medians.items():
        if value is None:
            trophy = next((t for t in trophies if t.get("id") == tid), {})
            tier = trophy.get("tier", "bronze")
            pool = tier_medians.get(tier) or [v for v in medians.values()
                                              if v is not None]
            medians[tid] = median_of(pool) if pool else 0.1

    # Story chain monotone: Ch1 >= Ch2 >= Ch3 >= Ch4 >= final.
    chain_present = [tid for tid in CHAIN_IDS if tid in medians]
    if len(chain_present) >= 2:
        fixed = pool_adjacent_decreasing([medians[tid] for tid in chain_present])
        for tid, value in zip(chain_present, fixed):
            medians[tid] = value

    rows = []
    for pos, trophy in enumerate(trophies):
        tid = trophy.get("id")
        series = raw.get(pos, [])
        last_rate = series[-1][1] if series else 0.0
        if tid == 2 and last_rate >= 99.0:
            # Gone Fishing: every tracked player completes the opener, so
            # its earners grow with N instead of staying frozen — the rate
            # stays ~100% through a refresh. Median ~99.9, q05 ~97.
            rows.append({"id": tid, "q": quantiles_for(99.9, 2.0, tid)})
            continue
        spread = forecast_spread(horizon_days, resid_vars.get(tid, 0.0), n_flat)
        rows.append({"id": tid,
                     "q": quantiles_for(medians[tid], spread, tid)})
    rows.sort(key=lambda r: r["id"])
    return {"trophies": rows}


def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR") or "/input"
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output"
    os.makedirs(out_dir, exist_ok=True)
    names = sorted(f for f in os.listdir(in_dir) if f.endswith(".in"))
    for name in names:
        case_id = name[:-3]
        out_path = os.path.join(out_dir, case_id + ".out")
        try:
            with open(os.path.join(in_dir, name), "r", encoding="utf-8") as fh:
                case = json.load(fh)
            result = forecast_case(case)
        except Exception:
            # Never crash a case: fall back to persistence on the last read.
            try:
                with open(os.path.join(in_dir, name), "r",
                          encoding="utf-8") as fh:
                    case = json.load(fh)
                horizon_days = float(case.get("horizon_minutes") or 0) / 1440.0
                _, _, raw = read_series(case)
                trophies = case.get("trophies") or []
                rows = []
                for pos, trophy in enumerate(trophies):
                    series = raw.get(pos, [])
                    last = series[-1][1] if series else 0.1
                    rows.append({"id": trophy.get("id", pos + 1),
                                 "q": persistence_row(last, horizon_days, True)})
                result = {"trophies": sorted(rows, key=lambda r: r["id"])}
            except Exception:
                result = {"trophies": []}
        text = json.dumps(result, separators=(",", ":"))
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
