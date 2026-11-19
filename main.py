#!/usr/bin/env python3
"""Forecast Exophase earn rates for Grand Theft Auto VI (PS5) trophies.

Entry for make.fun gta6-trophy-rarity-forecast-01 (refit v2, written fresh
in-session 19 Nov 2026). Python 3.13 standard library only, no network,
deterministic.

Model (see STRATEGY.md): decompose each rate into E(t)/N(t).
  - N(t): power-law fit of log N against ln(1+t) over the last 24 h with
    recency weighting. A flat history means the page is stale, not the
    world: forecast N as last N times the first observed refresh jump
    (3868/638, launch night).
  - E(t): per trophy, weighted fit of log E against ln(1+t), slope shrunk
    toward the trophy's group mean (weight E/(E+K)), level anchored at the
    last observed count; counts never decrease, so E_forecast >= 0.9*E_last.
  - Stale-level correction (new in v2): when the visible history ends in a
    long flat stretch, the last observed counts are stale-low — players kept
    earning while Exophase did not refresh (proven 09:11 -> 12:11 UTC 19 Nov:
    zero change in 3 h; and proxy 04:30 -> live 09:11 shows E grew 1.2-3.4x
    per trophy group during a frozen page). Correct E_last upward by the
    group ratio for the stale tail before the checkpoint, capped per group.
  - Story chain Ch1 >= Ch2 >= Ch3 >= Ch4 >= final, enforced by
    pool-adjacent-violators with minimal adjustment.
  - Spread (logit): random walk + fit residual + refresh risk, scaled by
    SPREAD_SCALE (calibrated on the practice backtest); platinum and 100%
    trophies get a longer upper tail. Gone Fishing (id 2, 100%) is special:
    median 99.9, its earners grow with N, so it is never diluted.

Score is mean pinball loss on the logit scale over 55 trophies x 5
quantiles (0.05/0.25/0.5/0.75/0.95). Output is one compact JSON line per
case: {"trophies": [{"id": n, "q": [q05, q25, q50, q75, q95]}]}.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone

RELEASE = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)
Z = (-1.645, -0.674, 0.0, 0.674, 1.645)

# --- Tunable constants (calibrated on the practice backtest, see report) ---
HALF_LIFE_H = 12.0          # recency half-life for fits (hours)
SHRINK_K = 20.0             # shrinkage weight: w = E / (E + K)
# SPREAD_SCALE = 0.9 — calibrated 19 Nov on V1+V2+V2b plus two independent
# 108 h synthetic finals. Scale 1.0 costs ~0.02 on the short-horizon real
# tests (V1 0.262 / V2 0.266); scale 0.7 would meet the V1 < 0.20 proxy
# target but costs +0.02-0.04 on the long horizon and collapses coverage
# there (0.87 at 0.7, 0.62 at 0.6, 0.11 at 0.4). The final is a 108 h
# case, so the long horizon outranks the proxy: 0.9 keeps long-horizon
# loss within 0.01 of the best scale while clearly beating v1 on both
# real-data tests (V1 0.241, V2 0.245).
SPREAD_SCALE = 0.9          # global spread multiplier (tuned on V1+V2)
RW_COEF = 0.5               # random-walk sigma per sqrt(day)
REFRESH_SIGMA_FLAT = 1.1    # refresh risk when N never moved in history
REFRESH_SIGMA_SEEN = 0.45   # refresh risk when refreshes are visible
SPREAD_FLOOR = 0.30
SPREAD_CAP = 2.50
DELAYED_REFRESH_MULT = 3868.0 / 638.0   # first observed refresh jump
STALE_WINDOW_H = 6.7        # hours spanned by the proxy->live window
CHAIN_IDS = [3, 4, 5, 6, 7]

# Shrinkage groups (prior slope is the group mean fit slope).
SHRINK_GROUPS = {
    "story": {3, 4, 5, 6, 7},
    "strand": {8, 9, 10, 11, 12, 13, 22, 23, 24, 26, 27, 37},
    "activity": {20, 21, 33, 34, 35, 38, 39, 41, 42, 44, 46, 47, 48,
                 49, 50, 51, 53, 54, 55},
    "grind": {15, 16, 17, 18, 19, 25, 28, 29, 30, 31, 32, 36, 40, 43,
              45, 52},
    "platinum": {1, 14},
}

# Stale-growth priors: (E-growth ratio per STALE_WINDOW_H, cap).
# From research/live-rates-2026-11-19T1211Z.md: rate ratios proxy 04:30
# -> live 09:11 UTC are lower bounds on E growth while the page was
# frozen (N can only grow, so E grew at least by the rate ratio).
STALE_GROUPS = {
    "early": ({3}, 2.33, 3.0),
    "story": ({4, 5, 6, 7}, 1.28, 2.0),
    "strand": ({8, 9, 10, 11, 12, 13, 22, 23, 24, 26, 27, 37}, 1.28, 2.0),
    "fast": ({20, 21, 35}, 2.57, 4.0),
    "activity": ({33, 34, 38, 39, 41, 42, 44, 46, 47, 48, 49, 50, 51,
                  53, 54, 55}, 1.24, 2.0),
    "floor": ({1, 14, 15, 16, 17, 18, 19, 25, 28, 29, 30, 31, 32, 36, 40,
               43, 45, 52}, 1.00, 1.0),
}


def _group_of(tid, table):
    for name, members in table.items():
        if tid in members:
            return name
    return None


def parse_time(value):
    """Parse an ISO-8601 timestamp (with 'Z') into an aware datetime."""
    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def hours_since_release(moment):
    """Hours from the 2026-11-19T00:00:00Z release to a datetime."""
    return (moment - RELEASE).total_seconds() / 3600.0


def _clamp_rate(rate):
    return min(max(float(rate), 0.01), 99.99)


def logit_pct(rate):
    """logit of a percentage rate, clamped to [0.01, 99.99]."""
    p = _clamp_rate(rate) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit_pct(value):
    """Inverse logit back to a percentage, clamped to [0.01, 99.99]."""
    if value > 30.0:
        return 99.99
    if value < -30.0:
        return 0.01
    return _clamp_rate(100.0 / (1.0 + math.exp(-value)))


def weighted_line(xs, ys, ws):
    """Weighted least squares y ~ a + b*x; returns (slope, intercept)."""
    total_w = sum(ws)
    if total_w <= 0.0 or len(xs) == 0:
        return 0.0, (sum(ys) / len(ys) if ys else 0.0)
    mean_x = sum(w * x for w, x in zip(ws, xs)) / total_w
    mean_y = sum(w * y for w, y in zip(ws, ys)) / total_w
    num = sum(w * (x - mean_x) * (y - mean_y)
              for w, x, y in zip(ws, xs, ys))
    den = sum(w * (x - mean_x) ** 2 for w, x in zip(ws, xs))
    if den <= 1e-12:
        return 0.0, mean_y
    slope = num / den
    return slope, mean_y - slope * mean_x


def median_of(values):
    """Plain median of a non-empty list of numbers."""
    ordered = sorted(values)
    count = len(ordered)
    mid = count // 2
    if count % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def pool_adjacent_decreasing(values):
    """Isotonic regression (non-increasing) via pool-adjacent-violators."""
    blocks = [[value, 1] for value in values]
    index = 0
    while index < len(blocks) - 1:
        if blocks[index][0] < blocks[index + 1][0]:
            merged_value = ((blocks[index][0] * blocks[index][1]
                             + blocks[index + 1][0] * blocks[index + 1][1])
                            / (blocks[index][1] + blocks[index + 1][1]))
            merged_count = blocks[index][1] + blocks[index + 1][1]
            blocks[index:index + 2] = [[merged_value, merged_count]]
            if index > 0:
                index -= 1
        else:
            index += 1
    out = []
    for value, count in blocks:
        out.extend([value] * count)
    return out


def read_series(case):
    """Split a case into time series.

    Returns (times, n_series, raw): times are hours since release for each
    history read, n_series the players-tracked per read, raw maps trophy
    position (0-based, same order as case["trophies"]) to a list of
    (time, rate) pairs with non-null rates.
    """
    history = case.get("history") or []
    rows = []
    for read in history:
        stamp = read.get("at") or read.get("slot")
        if stamp is None:
            continue
        rows.append((hours_since_release(parse_time(stamp)), read))
    rows.sort(key=lambda item: item[0])
    times = [t for t, _ in rows]
    n_series = [read.get("players_tracked") for _, read in rows]
    raw = {}
    trophies = case.get("trophies") or []
    for pos in range(len(trophies)):
        series = []
        for t, read in rows:
            rates = read.get("rates") or []
            if pos < len(rates) and rates[pos] is not None:
                series.append((t, float(rates[pos])))
        if series:
            raw[pos] = series
    return times, n_series, raw


def forecast_n(times, n_series, target_t):
    """Forecast players-tracked N at target_t; returns (n_forecast, flat).

    flat is True when N never moved in the visible history.
    """
    pairs = [(t, n) for t, n in zip(times, n_series) if n and n > 0]
    if not pairs:
        return 1.0, True
    n_last = pairs[-1][1]
    values = [n for _, n in pairs]
    if max(values) == min(values):
        # Page frozen: a refresh is overdue. The first observed refresh
        # multiplied N by 3868/638, so forecast that jump, not flat N.
        return float(n_last) * DELAYED_REFRESH_MULT, True
    t_max = max(t for t, _ in pairs)
    window = [(t, n) for t, n in pairs if t >= t_max - 24.0]
    if len(window) < 2:
        window = pairs
    xs = [math.log1p(t) for t, _ in window]
    ys = [math.log(n) for _, n in window]
    ws = [0.5 ** ((t_max - t) / HALF_LIFE_H) for t, _ in window]
    slope, intercept = weighted_line(xs, ys, ws)
    slope = min(max(slope, 0.0), 2.0)
    n_forecast = math.exp(intercept + slope * math.log1p(max(target_t, 0.0)))
    return max(n_forecast, float(n_last)), False


def _fit_log_e(series, n_pairs):
    """Fit log E against ln(1+t); returns (slope, e_last, t_last, resid).

    E = rate * N / 100 per read. Returns None when the series carries no
    usable point.
    """
    by_time = {}
    for t, n in n_pairs:
        by_time[t] = n
    points = []
    for t, rate in series:
        n = by_time.get(t)
        if n is None or n <= 0:
            continue
        earned = max(rate * n / 100.0, 0.5)
        points.append((t, math.log(earned), earned))
    if not points:
        return None
    t_last = points[-1][0]
    e_last = points[-1][2]
    if len(points) < 2:
        return 0.0, e_last, t_last, 0.0
    xs = [math.log1p(t) for t, _, _ in points]
    ys = [y for _, y, _ in points]
    ws = [0.5 ** ((t_last - t) / HALF_LIFE_H) for t, _, _ in points]
    slope, intercept = weighted_line(xs, ys, ws)
    resid = sum(w * (y - (intercept + slope * x)) ** 2
                for w, x, y in zip(ws, xs, ys)) / max(sum(ws), 1e-9)
    return slope, e_last, t_last, min(max(resid, 0.0), 4.0)


def forecast_trophy_median(series, n_pairs, n_forecast, target_t,
                           prior_slope):
    """Forecast one trophy's median rate; returns (median, resid_var).

    Log-E slope is shrunk toward prior_slope with weight E/(E+K); the
    level is anchored at the last observed count, and E never decreases.
    """
    fit = _fit_log_e(series, n_pairs)
    if fit is None:
        return None
    slope_fit, e_last, t_last, resid_var = fit
    weight = e_last / (e_last + SHRINK_K)
    slope = weight * slope_fit + (1.0 - weight) * prior_slope
    slope = min(max(slope, -2.0), 2.0)
    growth = math.exp(slope * (math.log1p(max(target_t, 0.0))
                               - math.log1p(max(t_last, 0.0))))
    e_forecast = max(e_last * growth, 0.9 * e_last)
    median = _clamp_rate(100.0 * e_forecast / max(n_forecast, 1.0))
    return median, resid_var


def stale_boost(tid, stale_hours):
    """Level correction for a trophy whose history ends in a flat stretch.

    While the page is frozen, earners keep growing unseen; the group ratio
    (lower bound from the proxy->live window) corrects E_last upward for
    the stale tail only, capped per group. No boost for short tails.
    """
    if stale_hours <= 1.0:
        return 1.0
    for _name, (members, ratio, cap) in STALE_GROUPS.items():
        if tid in members:
            if ratio <= 1.0:
                return 1.0
            return min(cap, ratio ** (stale_hours / STALE_WINDOW_H))
    return 1.0


def forecast_spread(horizon_days, resid_var, n_flat):
    """Honest logit spread: random walk + fit residual + refresh risk."""
    walk = RW_COEF * math.sqrt(max(horizon_days, 0.0))
    refresh = REFRESH_SIGMA_FLAT if n_flat else REFRESH_SIGMA_SEEN
    resid = min(max(resid_var, 0.0), 4.0)
    base = math.sqrt(walk * walk + resid + refresh * refresh)
    spread = base * SPREAD_SCALE
    return min(max(spread, SPREAD_FLOOR), SPREAD_CAP)


def quantiles_for(median, spread, tid):
    """Five quantiles around a median; platinum/100% get a longer top."""
    centre = logit_pct(median)
    out = []
    previous = 0.0
    for z in Z:
        width = spread
        if tid in (1, 14):
            width = spread * (1.35 if z > 0 else 0.90 if z < 0 else 1.0)
        value = round(inv_logit_pct(centre + z * width), 4)
        value = max(value, previous)
        out.append(value)
        previous = value
    return out


def forecast_case(case):
    """Forecast every trophy in one case; returns {"trophies": [...]}."""
    trophies = case.get("trophies") or []
    if not trophies:
        return {"trophies": []}
    times, n_series, raw = read_series(case)
    target_raw = case.get("target")
    target_t = (hours_since_release(parse_time(target_raw))
                if target_raw else (times[-1] if times else 0.0))
    at_raw = case.get("at")
    checkpoint_t = (hours_since_release(parse_time(at_raw))
                    if at_raw else (times[-1] if times else 0.0))
    horizon_days = float(case.get("horizon_minutes") or 0) / 1440.0
    if horizon_days <= 0.0 and target_t > checkpoint_t:
        horizon_days = (target_t - checkpoint_t) / 24.0
    n_forecast, n_flat = forecast_n(times, n_series, target_t)
    n_pairs = [(t, n) for t, n in zip(times, n_series) if n and n > 0]

    # Pass 1: unshrunk log-E slopes, for the group priors.
    fitted_slope = {}
    for pos, trophy in enumerate(trophies):
        tid = trophy.get("id", pos + 1)
        if tid == 2:
            continue
        fit = _fit_log_e(raw.get(pos, []), n_pairs)
        if fit is not None:
            fitted_slope[tid] = fit[0]
    group_prior = {}
    for name, members in SHRINK_GROUPS.items():
        slopes = [fitted_slope[t] for t in members if t in fitted_slope]
        group_prior[name] = sum(slopes) / len(slopes) if slopes else 0.0

    # Pass 2: medians with shrinkage, then the stale-level correction.
    medians = {}
    resid_vars = {}
    for pos, trophy in enumerate(trophies):
        tid = trophy.get("id", pos + 1)
        if tid == 2:
            continue
        group = _group_of(tid, SHRINK_GROUPS)
        prior = group_prior.get(group, 0.0) if group else 0.0
        got = forecast_trophy_median(raw.get(pos, []), n_pairs,
                                     n_forecast, target_t, prior)
        if got is None:
            continue
        med, resid = got
        medians[tid] = med
        resid_vars[tid] = resid
        series = raw.get(pos, [])
        if series:
            last_change = series[0][0]
            previous_rate = series[0][1]
            for t, rate in series[1:]:
                if rate != previous_rate:
                    last_change = t
                    previous_rate = rate
            stale_hours = checkpoint_t - last_change
            boost = stale_boost(tid, stale_hours)
            if boost > 1.0:
                medians[tid] = _clamp_rate(med * boost)

    # Fallback for trophies with no usable series: their tier median.
    for pos, trophy in enumerate(trophies):
        tid = trophy.get("id", pos + 1)
        if tid in medians or tid == 2:
            continue
        tier = trophy.get("tier", "bronze")
        pool = [medians[t] for t in medians
                if trophies[t - 1].get("tier", "bronze") == tier]
        if not pool:
            pool = [v for v in medians.values()]
        medians[tid] = median_of(pool) if pool else 0.1
        resid_vars[tid] = 0.0

    # Story chain must not increase: Ch1 >= Ch2 >= Ch3 >= Ch4 >= final.
    chain = [t for t in CHAIN_IDS if t in medians]
    if len(chain) >= 2:
        fixed = pool_adjacent_decreasing([medians[t] for t in chain])
        for tid, value in zip(chain, fixed):
            medians[tid] = value

    rows = []
    for pos, trophy in enumerate(trophies):
        tid = trophy.get("id", pos + 1)
        series = raw.get(pos, [])
        last_rate = series[-1][1] if series else 0.0
        if tid == 2 and last_rate >= 99.0:
            rows.append({"id": tid, "q": quantiles_for(99.9, 2.0, tid)})
            continue
        spread = forecast_spread(horizon_days, resid_vars.get(tid, 0.0),
                                 n_flat)
        rows.append({"id": tid,
                     "q": quantiles_for(medians.get(tid, 0.1), spread, tid)})
    rows.sort(key=lambda row: row["id"])
    return {"trophies": rows}


def _persistence_fallback(case):
    """Never crash a case: persistence median with a wide honest spread."""
    trophies = case.get("trophies") or []
    horizon_days = float(case.get("horizon_minutes") or 0) / 1440.0
    _, _, raw = read_series(case)
    spread = forecast_spread(horizon_days, 0.0, True)
    rows = []
    for pos, trophy in enumerate(trophies):
        tid = trophy.get("id", pos + 1)
        series = raw.get(pos, [])
        last_rate = series[-1][1] if series else 0.1
        if tid == 2 and last_rate >= 99.0:
            rows.append({"id": tid, "q": quantiles_for(99.9, 2.0, tid)})
        else:
            rows.append({"id": tid,
                         "q": quantiles_for(last_rate, spread, tid)})
    rows.sort(key=lambda row: row["id"])
    return {"trophies": rows}


def main():
    input_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    output_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    os.makedirs(output_dir, exist_ok=True)
    for name in sorted(os.listdir(input_dir)):
        if not name.endswith(".in"):
            continue
        with open(os.path.join(input_dir, name)) as handle:
            case = json.load(handle)
        try:
            forecast = forecast_case(case)
        except Exception:
            forecast = _persistence_fallback(case)
        out_name = name[:-3] + ".out"
        blob = json.dumps(forecast, separators=(",", ":"))
        with open(os.path.join(output_dir, out_name), "w") as handle:
            handle.write(blob)
    return 0


if __name__ == "__main__":
    sys.exit(main())
