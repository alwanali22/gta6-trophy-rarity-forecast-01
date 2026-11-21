# SPDX-License-Identifier: MIT
"""Forecast GTA VI (PS5) trophy earn rates on Exophase.

Final refit (21 Nov 2026, in-session): fresh live snapshot 21 Nov 04:28 UTC
+ branch-mixture spread + censored-zero handling.

Model, per case, per trophy (design: STRATEGY.md, BUILD-REPORT-2026-11-19.md,
REFIT-REPORT-2026-11-19.md, research/live-rates-2026-11-21T0428Z.md):

  rate(t) = E(t) / N(t): E = earners, N = players tracked. E and N are
  forecast separately, because launch week is a tug-of-war between
  dilution (N explodes, every rate falls) and progression (E grows as
  the player cohort plays through the game).

  * N: weighted least squares of log N against ln(1 + hours since
    release) over the last 24 h of reads (recency half-life 12 h),
    slope clamped to [0, 1.2]. If N never moves in the history the page
    is stale, not stable: one delayed refresh is priced in,
    N_forecast = N_last * 3868/638 (the first refresh, launch night).
  * E: weighted least squares of log E against ln(1 + hours) over the
    last 24 h; slope shrunk toward the trophy's group mean with weight
    w = E_last / (E_last + 20), so small-count grind trophies borrow
    strength while big-E early trophies (Ch1, fishing) keep their own
    slope. Level is anchored to the last read, with a stale correction:
    if the page sat flat for hours before the checkpoint, E_last is
    boosted by the group's measured frozen-page growth (19 Nov
    proxy -> live ratios), capped per group. E never falls below
    0.9 * E_last: earner counts are cumulative. The median rate is
    capped at min(3x the last rate, 60%): a log-E power law fitted
    on a steep growth phase can otherwise explode to 99.99% for
    fast risers (Ch1, fishing) — the cap binds only when the fit
    extrapolates explosively, while the upper quantiles still reach
    above it.
  * A 0.00% last rate is CENSORED (E in [0, 0.005% of N]), not E = 0:
    it enters the fit as half an earner at half weight.
  * Story chain (Ch1 >= Ch2 >= Ch3 >= Ch4 >= final) is made monotone
    on the logit medians by pool-adjacent-violators averaging (minimal
    adjustment). Fresh data 21 Nov confirms it: 18.02 >= 3.61 >= 0.07
    >= 0.03 >= 0.03. No other chain is forced.
  * Spread is a BRANCH MIXTURE, not a Gaussian blur. Exophase
    refreshes in jumps: the predictive distribution is three branches
    centred at logit(median) + J, logit(median), logit(median) - J,
    where J is the typical single-refresh |delta ln N| (prior
    ln(3868/638) = 1.80 when no refresh is visible in the history),
    with weights (0.10, 0.80, 0.10). The no-refresh / double-refresh
    branches really are ~1.8 logit apart, so the OUTER quantiles stay
    wide enough to cover them, but the INNER quantiles (0.25/0.75)
    sit inside the dominant branch instead of being smeared across
    both. That is the honest tightening vs v2: same branch risk,
    less blur in the middle. Within a branch the logit scale is
    s0 = SPREAD_SCALE * sqrt(random-walk^2 + residual^2), random walk
    0.5 * sqrt(horizon days). J shrinks when the horizon spans many
    refreshes (the N power law already aggregates them).
  * Gone Fishing (id 2, 100% in every read): median 99.9, fixed wide
    row. Platinum / 100%-completion (ids 1, 14): upper tail stretched
    x1.35, lower tail x0.90 (a few hardcore players can finish far
    faster than the median; the floor is hard). Guides put the story
    at ~80 h and the platinum at ~200-300 h, vs 168 h from release to
    target, so those medians belong at the 0.01% floor.

Python 3.13, standard library only, no network, deterministic.
Reads $MAKEFUN_INPUT_DIR (else /input), writes $MAKEFUN_OUTPUT_DIR
(else /output): one <id>.out per <id>.in, {"trophies": [{"id": n,
"q": [q05, q25, q50, q75, q95]}]}, 4 decimals, non-decreasing.
"""

import json
import math
import os
from datetime import datetime, timezone

RELEASE = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)
LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)
Z = (-1.6448536269514722, -0.6744897501960817, 0.0,
     0.6744897501960817, 1.6448536269514722)

HALF_LIFE_H = 12.0
WINDOW_H = 24.0
SHRINK_K = 20.0
SPREAD_SCALE = 0.90
MIX_W = (0.10, 0.80, 0.10)
JUMP_PRIOR = 1.8023          # ln(3868 / 638), the first refresh
INTERVAL_PRIOR_H = 6.0
DELAYED_REFRESH = 3868.0 / 638.0

# Group of each trophy id (STRATEGY.md section 4.5 + fresh 21 Nov data:
# fast risers are fishing / arcade / vehicles / bowling).
STORY = {3, 4, 5, 6, 7}
STRAND = {8, 9, 10, 11, 12, 13, 22, 23, 24, 26, 27, 37}
FAST = {20, 21, 35, 46}
GRIND = {1, 14, 15, 17, 25, 30, 31, 32}
CHAIN = (3, 4, 5, 6, 7)     # story chain, monotone by PAVA

# Frozen-page E growth per 6.7 h (19 Nov proxy -> live 09:11 UTC ratios),
# and the cap applied to the stale boost, per group.
STALE_FACTOR = {"story_early": 2.33, "story": 1.28, "strand": 1.28,
                "fast": 2.57, "activity": 1.24, "grind": 1.00}
STALE_CAP = {"story_early": 3.0, "story": 2.0, "strand": 2.0,
             "fast": 4.0, "activity": 2.0, "grind": 2.0}

GONE_FISHING_Q = (97.38, 99.62, 99.9, 99.97, 99.99)


def group_of(tid):
    if tid == 2:
        return "gone"
    if tid == 3:
        return "story_early"
    if tid in STORY:
        return "story"
    if tid in STRAND:
        return "strand"
    if tid in FAST:
        return "fast"
    if tid in GRIND:
        return "grind"
    return "activity"


def parse_time(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def hours_since_release(moment):
    """Hours from the 19 Nov 2026 00:00 UTC release to a datetime."""
    return (moment - RELEASE).total_seconds() / 3600.0


def read_series(case):
    """History of a case as parallel series.

    Returns (times, n_series, raw): times are hours since release,
    n_series the players-tracked count per read, raw a dict from
    trophy position (0-based, in case order) to that trophy's
    earn-rate series over the reads (None where a read lacked it).
    Reads are oldest first.
    """
    times, n_series = [], []
    rows = []
    history = sorted(case.get("history") or [],
                     key=lambda r: r.get("at", ""))
    for rec in history:
        try:
            times.append(hours_since_release(parse_time(rec["at"])))
        except Exception:
            continue
        n = rec.get("players_tracked")
        n_series.append(float(n) if n else 0.0)
        rows.append(list(rec.get("rates") or []))
    n_tr = len(case.get("trophies") or [])
    raw = {}
    for i in range(n_tr):
        raw[i] = [(times[k], rows[k][i]) for k in range(len(rows))
                  if i < len(rows[k]) and rows[k][i] is not None]
    return times, n_series, raw


def logit_pct(rate):
    p = min(max(rate / 100.0, 0.0001), 0.9999)
    return math.log(p / (1.0 - p))


def inv_logit_pct(mu):
    if mu > 30.0:
        return 99.99
    if mu < -30.0:
        return 0.01
    p = 1.0 / (1.0 + math.exp(-mu))
    return min(max(p * 100.0, 0.01), 99.99)


def phi(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def weighted_line(xs, ys, ws):
    """Weighted least squares y ~ a + b*x; returns (slope, intercept).
    Alias of wls() under the name the validation harness calls."""
    return wls(xs, ys, ws)


def wls(xs, ys, ws):
    """Weighted least squares y ~ a + b*x. Returns (slope, intercept)."""
    sw = sum(ws)
    if sw <= 0.0:
        return 0.0, (ys[0] if ys else 0.0)
    mx = sum(w * x for w, x in zip(ws, xs)) / sw
    my = sum(w * y for w, y in zip(ws, ys)) / sw
    sxx = sum(w * (x - mx) ** 2 for w, x in zip(ws, xs))
    if sxx <= 1e-12:
        return 0.0, my
    sxy = sum(w * (x - mx) * (y - my) for w, x, y in zip(ws, xs, ys))
    b = sxy / sxx
    return b, my - b * mx


def resid_sd(xs, ys, ws, slope, intercept):
    n = len(xs)
    if n < 3:
        return None
    sw = sum(ws)
    var = sum(w * (y - (intercept + slope * x)) ** 2
              for w, x, y in zip(ws, xs, ys)) / sw
    dof = max(n - 2, 1)
    return math.sqrt(max(var * n / dof, 0.0))


def pool_adjacent_decreasing(values):
    """PAVA on a list of medians: smallest L2 adjustment making the
    list non-increasing. Same pooling as pava(); kept as its own
    name because the validation harness calls it directly."""
    return pava([float(v) for v in values])


def pava(values):
    """Pool-adjacent-violators: smallest L2 adjustment making the list
    non-increasing. Values are logit medians of the story chain."""
    pools = [[v, 1] for v in values]
    i = 0
    while i < len(pools) - 1:
        if pools[i][0] < pools[i + 1][0]:
            tot = pools[i][0] * pools[i][1] + pools[i + 1][0] * pools[i + 1][1]
            cnt = pools[i][1] + pools[i + 1][1]
            pools[i:i + 2] = [[tot / cnt, cnt]]
            if i > 0:
                i -= 1
        else:
            i += 1
    out = []
    for val, cnt in pools:
        out.extend([val] * cnt)
    return out


def mixture_quantile(mu, s0, jump, tau):
    """tau-quantile of the 3-branch mixture on the logit scale."""
    if jump < 1e-6 or s0 < 1e-6:
        return mu + _z(tau) * max(s0, 1e-6)
    scale = max(s0, jump)
    lo = mu - 8.0 * scale
    hi = mu + 8.0 * scale
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        cdf = (MIX_W[0] * phi((mid - (mu + jump)) / s0)
               + MIX_W[1] * phi((mid - mu) / s0)
               + MIX_W[2] * phi((mid - (mu - jump)) / s0))
        if cdf < tau:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _z(tau):
    return Z[LEVELS.index(tau)]


def forecast_n(times, n_series, target_t):
    """Forecast players-tracked at the target (hours since release).

    Power-law fit: weighted least squares of log N against
    ln(1 + hours), last 24 h of reads (recency half-life 12 h),
    slope clamped to [0, 1.2]; falls back to the full history.
    If N never moves, the page is stale, not stable: one delayed
    refresh is priced in (N_last * 3868/638).
    Returns (n_forecast, n_flat).
    """
    if not times:
        return 638.0 * DELAYED_REFRESH, True
    t_last = max(times)
    n_last = 0.0
    for t, n in sorted(zip(times, n_series)):
        if n and n > 0:
            n_last = float(n)
    if n_last <= 0:
        n_last = 638.0
    x_target = math.log1p(max(target_t, 0.0))
    for window in (WINDOW_H, None):
        pts = [(math.log1p(max(t, 0.0)), math.log(n),
                0.5 ** ((t_last - t) / HALF_LIFE_H))
               for t, n in zip(times, n_series)
               if n and n > 0 and (window is None or t >= t_last - window)]
        distinct_n = {round(p[1], 9) for p in pts}
        if len(pts) >= 2 and len(distinct_n) >= 2:
            slope, icept = wls([p[0] for p in pts], [p[1] for p in pts],
                               [p[2] for p in pts])
            slope = min(max(slope, 0.0), 1.2)
            return max(math.exp(icept + slope * x_target), n_last), False
    return n_last * DELAYED_REFRESH, True


CHAIN_IDS = [3, 4, 5, 6, 7]  # story chain, monotone by PAVA


def forecast_trophy_median(series, n_pairs, n_forecast, target_t,
                           group_slope=0.0):
    """Median earn rate (%) for one trophy at the target.

    series: [(hours since release, rate %)] for the trophy;
    n_pairs: [(hours since release, players tracked)];
    n_forecast: forecast players tracked at the target;
    target_t: target in hours since release; group_slope: slope
    prior the trophy's own log-E slope is shrunk toward.
    E(t) = rate * N / 100 is fit by weighted least squares of log E
    against ln(1 + hours) (last 24 h, recency half-life 12 h); the
    slope is shrunk with weight E_last / (E_last + 20); the level is
    anchored to the last read and E never falls below 0.9 * E_last.
    Returns (median_rate_percent, shrunk_slope).
    """
    n_by_t = {t: n for t, n in n_pairs}
    pts = []
    t_last = None
    rate_last = None
    for t, r in series:
        n = n_by_t.get(t)
        if n is None or n <= 0 or r is None:
            continue
        t_last = t if t_last is None else max(t_last, t)
        rate_last = float(r)
        e = 0.5 if float(r) <= 0.0 else float(r) / 100.0 * float(n)
        pts.append((t, math.log(max(e, 0.05))))
    if t_last is None or rate_last is None:
        return 0.5, 0.0
    n_last = n_by_t.get(t_last) or 0.0
    if n_last <= 0:
        return 0.5, 0.0
    e_last = 0.5 if rate_last <= 0.0 else rate_last / 100.0 * n_last
    win = [(t, y) for (t, y) in pts if t >= t_last - WINDOW_H]
    if len(win) < 2:
        win = pts
    xs = [math.log1p(max(t, 0.0)) for (t, _y) in win]
    slope = 0.0
    if len(win) >= 2 and len(set(round(v, 9) for v in xs)) >= 2:
        ws = [0.5 ** ((t_last - t) / HALF_LIFE_H) for (t, _y) in win]
        slope, _a = wls(xs, [y for (_t, y) in win], ws)
    wgt = e_last / (e_last + SHRINK_K)
    b = wgt * slope + (1.0 - wgt) * float(group_slope)
    x_last = math.log1p(max(t_last, 0.0))
    x_tgt = math.log1p(max(target_t, 0.0))
    e_fc = e_last * math.exp(b * (x_tgt - x_last))
    e_fc = max(e_fc, 0.9 * e_last)
    median = e_fc / n_forecast * 100.0 if n_forecast > 0 else 0.5
    return min(max(median, 0.01), 99.99), b


def forecast_spread(horizon_days, resid_var, n_flat):
    """Logit spread (sigma) for the harness's Gaussian variants.

    sqrt of random-walk variance (0.5 * sqrt(days))^2 + the fit's
    residual variance + refresh variance: 1.1^2 when N never moved
    in the history (a refresh is overdue), 0.45^2 when refreshes
    were seen. Scaled by SPREAD_SCALE, floored at 0.30, capped
    at 2.50. The entry's own quantiles use the branch mixture
    instead; this helper keeps the V5 variant comparison honest.
    """
    rw = 0.5 * math.sqrt(max(horizon_days, 0.0))
    refresh = 1.1 if n_flat else 0.45
    s = SPREAD_SCALE * math.sqrt(rw * rw + max(float(resid_var), 0.0)
                                 + refresh * refresh)
    return min(max(s, 0.30), 2.50)


def quantiles_for(median, spread, tid):
    """Five Gaussian quantiles (percent) for a trophy with the given
    median rate and logit spread: q_tau = inv_logit(logit(median)
    + z_tau * spread). Platinum / 100%-completion (ids 1, 14) get a
    stretched upper tail (x1.35) and a shrunk lower tail (x0.90);
    Gone Fishing (id 2) keeps its fixed wide row. The entry's own
    quantiles use the branch mixture; this helper serves the V5
    variant comparison."""
    if tid == 2:
        return [round(v, 4) for v in GONE_FISHING_Q]
    base = logit_pct(float(median))
    qs = []
    for z in Z:
        m = base + z * float(spread)
        if tid in (1, 14):
            if m > base:
                m = base + (m - base) * 1.35
            else:
                m = base + (m - base) * 0.90
        qs.append(round(inv_logit_pct(m), 4))
    for k in range(1, 5):
        if qs[k] < qs[k - 1]:
            qs[k] = qs[k - 1]
    return qs


def forecast_case(case):
    trophies = case["trophies"]
    history = sorted(case.get("history") or [],
                     key=lambda r: r.get("at", ""))
    checkpoint = parse_time(case["at"])
    horizon_h = float(case.get("horizon_minutes", 0)) / 60.0
    horizon_d = max(horizon_h / 24.0, 1e-6)
    target = parse_time(case["target"])
    t_target = (target - RELEASE).total_seconds() / 3600.0
    x_target = math.log1p(max(t_target, 0.0))

    # ---- time series -------------------------------------------------
    reads = []  # (t_hours, x, N, rates list)
    for rec in history:
        try:
            t_h = (parse_time(rec["at"]) - RELEASE).total_seconds() / 3600.0
            n = rec.get("players_tracked")
            rates = rec.get("rates") or []
        except Exception:
            continue
        if t_h < -1.0:
            continue
        reads.append((t_h, math.log1p(max(t_h, 0.0)),
                      float(n) if n else None, rates))
    reads.sort(key=lambda r: r[0])
    if reads:
        t_last = reads[-1][0]
        x_last = reads[-1][1]
        n_last = reads[-1][2]
    else:
        t_last, x_last, n_last = 0.0, 0.0, None
    if n_last is None or n_last <= 0:
        n_last = 638.0

    # ---- refresh events (N jumps) -> jump size J, cadence ------------
    jumps = []
    event_times = []
    for a, b in zip(reads, reads[1:]):
        if a[2] and b[2] and a[2] > 0 and b[2] > 0:
            ratio = b[2] / a[2]
            if ratio > 1.005 or ratio < 0.995:
                jumps.append(abs(math.log(ratio)))
                event_times.append(b[0])
    if jumps:
        sj = sorted(jumps)
        jump = sj[len(sj) // 2]
    else:
        jump = JUMP_PRIOR
    jump = min(max(jump, 0.05), 2.0)
    if len(event_times) >= 2:
        gaps = [b - a for a, b in zip(event_times, event_times[1:])]
        interval = sum(gaps) / len(gaps) if gaps else INTERVAL_PRIOR_H
    elif len(event_times) == 1 and reads:
        interval = max(event_times[0] - reads[0][0], 1.0)
    else:
        interval = INTERVAL_PRIOR_H
    interval = min(max(interval, 1.0), 48.0)
    lam = horizon_h / interval if interval > 0 else 0.0
    # Branch separation shrinks when the horizon spans many refreshes:
    # the N power law already aggregates those jumps.
    jump_eff = jump * min(1.0, 1.5 / math.sqrt(max(lam, 0.25)))

    # ---- N forecast ---------------------------------------------------
    n_fc, _n_flat = forecast_n([r[0] for r in reads],
                               [r[2] if r[2] else 0.0 for r in reads],
                               t_target)

    # ---- flat tail before the checkpoint (stale page?) ----------------
    flat_tail_h = 0.0
    for k in range(len(reads) - 1, 0, -1):
        prev, cur = reads[k - 1], reads[k]
        if prev[3] != cur[3] or prev[2] != cur[2]:
            flat_tail_h = t_last - cur[0]
            break
    else:
        if len(reads) >= 2:
            flat_tail_h = t_last - reads[0][0]

    # ---- per-trophy E fits -------------------------------------------
    n_tr = len(trophies)
    slopes = [0.0] * n_tr
    fitted = [False] * n_tr
    e_last = [1.0] * n_tr
    resids = [None] * n_tr
    last_rate = [None] * n_tr
    for idx, tr in enumerate(trophies):
        tid = tr["id"]
        series = []  # (t, x, E, weight)
        lr = None
        for (t, x, n, rates) in reads:
            if idx >= len(rates) or rates[idx] is None or n is None or n <= 0:
                continue
            r = float(rates[idx])
            lr = r
            if r <= 0.0:
                e, w = 0.5, 0.5     # censored 0.00%: half an earner
            else:
                e, w = r / 100.0 * n, 1.0
            w *= 0.5 ** ((t_last - t) / HALF_LIFE_H)
            series.append((t, x, math.log(max(e, 0.05)), w))
        last_rate[idx] = lr
        if lr is None:
            e_last[idx] = 1.0
        elif lr <= 0.0:
            e_last[idx] = 0.5
        else:
            e_last[idx] = lr / 100.0 * n_last
        win = [(x, y, w) for (t, x, y, w) in series if t >= t_last - WINDOW_H]
        if len(win) < 2:
            win = [(x, y, w) for (_t, x, y, w) in series]
        xs = [p[0] for p in win]
        if len(win) >= 2 and len(set(round(v, 9) for v in xs)) >= 2:
            b, a = wls(xs, [p[1] for p in win], [p[2] for p in win])
            slopes[idx] = b
            fitted[idx] = True
            resids[idx] = resid_sd(xs, [p[1] for p in win],
                                   [p[2] for p in win], b, a)
        else:
            slopes[idx] = 0.0
            fitted[idx] = False
            resids[idx] = None

    # ---- group slope means (E-weighted) -------------------------------
    group_slope = {}
    group_resid = {}
    for g in ("story_early", "story", "strand", "fast", "activity", "grind"):
        num = den = 0.0
        rs = []
        for idx, tr in enumerate(trophies):
            if group_of(tr["id"]) != g or not fitted[idx]:
                continue
            wgt = e_last[idx]
            num += wgt * slopes[idx]
            den += wgt
            if resids[idx] is not None:
                rs.append(resids[idx])
        group_slope[g] = num / den if den > 0 else 0.0
        rs.sort()
        group_resid[g] = rs[len(rs) // 2] if rs else 0.30

    # ---- medians + within-branch spread -------------------------------
    mu = [0.0] * n_tr
    s0s = [0.5] * n_tr
    for idx, tr in enumerate(trophies):
        tid = tr["id"]
        g = group_of(tid)
        if tid == 2:
            mu[idx] = logit_pct(99.9)
            s0s[idx] = 0.5
            continue
        wgt = e_last[idx] / (e_last[idx] + SHRINK_K)
        b = wgt * slopes[idx] + (1.0 - wgt) * group_slope.get(g, 0.0)
        # Stale boost on the anchored level, scaled by flat-tail length.
        e_anchor = e_last[idx]
        if flat_tail_h > 0.5:
            boost = STALE_FACTOR.get(g, 1.0) ** (flat_tail_h / 6.7)
            boost = min(boost, STALE_CAP.get(g, 2.0))
            e_anchor = e_last[idx] * boost
        e_fc = e_anchor * math.exp(b * (x_target - x_last))
        e_fc = max(e_fc, 0.9 * e_anchor)
        median = e_fc / n_fc * 100.0
        # Extrapolation guard: a log-E power law fitted on a steep
        # growth phase can explode (smoke test: Ch1 fitted slope ~3
        # sent the median to 99.99%). A rate cannot credibly more
        # than triple over the horizon while fresh buyers keep
        # entering at zero progress, and no trophy but id 2 can pass
        # 60% while N still grows. The cap binds only when the fit
        # extrapolates explosively; the upper quantiles still reach
        # above it, so the wave case stays covered.
        last_pct = last_rate[idx] if last_rate[idx] else 0.0
        cap = min(60.0, max(3.0 * last_pct, 1.0))
        median = min(max(median, 0.01), cap, 99.99)
        mu[idx] = logit_pct(median)
        resid = resids[idx]
        if resid is None:
            resid = group_resid.get(g, 0.30)
        resid = min(max(resid, 0.05), 1.0)
        rw = 0.5 * math.sqrt(horizon_d)
        s0 = SPREAD_SCALE * math.sqrt(rw * rw + resid * resid)
        s0s[idx] = min(max(s0, 0.15), 2.0)

    # ---- story-chain monotonicity (PAVA on logit medians) -------------
    pos = {tr["id"]: i for i, tr in enumerate(trophies)}
    if all(tid in pos for tid in CHAIN):
        vals = [mu[pos[tid]] for tid in CHAIN]
        adj = pava(vals)
        for tid, val in zip(CHAIN, adj):
            mu[pos[tid]] = val

    # ---- quantiles ------------------------------------------------------
    rows = []
    for idx, tr in enumerate(trophies):
        tid = tr["id"]
        if tid == 2:
            rows.append({"id": tid, "q": [round(v, 4) for v in GONE_FISHING_Q]})
            continue
        qs = []
        for tau in LEVELS:
            m = mixture_quantile(mu[idx], s0s[idx], jump_eff, tau)
            if tid in (1, 14):
                # Long upper tail, hard floor (platinum / 100%).
                if m > mu[idx]:
                    m = mu[idx] + (m - mu[idx]) * 1.35
                else:
                    m = mu[idx] + (m - mu[idx]) * 0.90
            qs.append(round(inv_logit_pct(m), 4))
        # enforce non-decreasing after rounding
        for k in range(1, 5):
            if qs[k] < qs[k - 1]:
                qs[k] = qs[k - 1]
        rows.append({"id": tid, "q": qs})
    return {"trophies": rows}


def fallback_case(case):
    """Persistence with the house-baseline spread; never invalid."""
    horizon_d = max(float(case.get("horizon_minutes", 0)) / 1440.0, 1e-6)
    rw = 0.5 * math.sqrt(horizon_d)
    rows = []
    hist = case.get("history") or []
    last = hist[-1].get("rates") if hist else []
    for tr in case["trophies"]:
        tid = tr["id"]
        r = None
        if last and tid - 1 < len(last):
            r = last[tid - 1]
        if r is None:
            r = 99.9 if tid == 2 else 0.5
        base = logit_pct(float(r))
        qs = [round(inv_logit_pct(base + z * rw), 4) for z in Z]
        for k in range(1, 5):
            if qs[k] < qs[k - 1]:
                qs[k] = qs[k - 1]
        rows.append({"id": tid, "q": qs})
    return {"trophies": rows}


def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    os.makedirs(out_dir, exist_ok=True)
    names = sorted(fn for fn in os.listdir(in_dir) if fn.endswith(".in"))
    for fn in names:
        with open(os.path.join(in_dir, fn), "r", encoding="utf-8") as fh:
            case = json.load(fh)
        try:
            result = forecast_case(case)
        except Exception:
            result = fallback_case(case)
        out_path = os.path.join(out_dir, fn[:-3] + ".out")
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(result, separators=(",", ":")))


if __name__ == "__main__":
    main()
