"""gta6-twitch-ccu-forecast-01 entry in Python 3.13 (standard library only).

Forecasts Twitch concurrent viewers in the "Grand Theft Auto VI" category
for hours 2..24 of release day, 2026-11-19 (UTC), from the case's hour-1
print and the release-day curves of past launches shipped in the case
(`comparables`).

Method
------
Work in log-ratios to hour 1: for a past launch i and hour h,
y_ih = log(v_ih / v_i1). Launch size (log hour-1 viewers) is the dominant
predictor of a release day's shape (corr -0.9 with the day's rise in the
kit's comparables): launches whose hour 1 is already huge unlocked
everywhere at once and stay flat; launches whose hour 1 is small still
have their big regional unlocks coming and rise steeply later. So for
each hour, take kernel-weighted quantiles of y_ih over the comparables,
with a Gaussian kernel on log hour-1 viewers (bandwidth 1.0 in log units),
downweighting tiny hour-1 launches (their ratios are noisy). The median
is the weighted median; the outer quantiles are scaled by 0.85 around it
(spread calibrated by leave-one-out cross-validation over the kit's 37
comparables: mean pinball loss 0.41 vs 0.93 for the persistence baseline,
0.21 vs 0.25 on launches with hour 1 >= 50k).

The forecast is scale-free (ratios only), so an hour-1 print beyond any
past launch still yields the biggest launches' shape, with no
extrapolation blow-up. If a case has no usable comparables, or anything
unexpected happens, it falls back to the house persistence baseline, so
the entry always writes a valid file.

How the grader runs it: `python3 -P -s /bot/main.py` (PYTHONHASHSEED=0),
standard library only, no network. It reads every <id>.in in the input
directory and writes <id>.out to the output directory:
{"hours": [{"hour": 2, "q": [q05, q25, q50, q75, q95]}, ...]}, exactly 23
rows, hours 2..24 each once, each q five finite numbers above 0 and at
most 1e9, non-decreasing. Runs are deterministic: no randomness, no time
or environment dependence, inputs processed in sorted order.

Outside the grader's container, MAKEFUN_INPUT_DIR and MAKEFUN_OUTPUT_DIR
stand in for /input and /output:
    mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
"""

import json
import math
import os
from datetime import datetime
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level.
Z = [-1.6448536269514722, -0.6744897501960817, 0.0, 0.6744897501960817, 1.6448536269514722]
# Gaussian kernel bandwidth on log hour-1 viewers (log units).
BW_SIZE = 1.0
# Spread of the outer quantiles around the median (1.0 = raw weighted quantiles).
SPREAD = 0.85
# Persistence fallback spread (the house baseline's sigma).
FALLBACK_SIGMA = 0.15
# Hour-1 prints below this many viewers get downweighted: their ratios are noisy.
TINY_H1 = 1000.0
TINY_WEIGHT = 0.3


def round2(x: float) -> float | int:
    """x rounded to 2 decimals, half up, as JavaScript's Math.round(x * 100) / 100 rounds it.

    A whole number comes back as an int, so json.dumps writes 1240 where JavaScript writes 1240 (not 1240.0).
    """
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value.is_integer() else value


def _ts(s) -> float:
    return datetime.fromisoformat(str(s).replace("Z", "+00:00")).timestamp()


def _wquantile(pairs, tau: float) -> float:
    """Weighted quantile of values. pairs: list of (value, weight), any order."""
    pairs = sorted(pairs)
    tot = sum(w for _, w in pairs)
    if tot <= 0:
        vals = sorted(v for v, _ in pairs)
        return vals[len(vals) // 2] if vals else 0.0
    cum = 0.0
    for v, w in pairs:
        cum += w
        if cum >= tau * tot:
            return v
    return pairs[-1][0]


def _day_hours(hourly, day_start_ts: float) -> dict:
    """hour (1..24) -> viewers on the UTC release day, from a list of points."""
    acc: dict[int, list[float]] = {}
    for p in hourly or []:
        try:
            t = _ts(p["from"])
        except Exception:
            continue
        v = p.get("viewers")
        if v is None:
            continue
        dh = (t - day_start_ts) / 3600.0
        if 0 <= dh < 24:
            acc.setdefault(int(dh) + 1, []).append(float(v))
    return {h: sum(v) / len(v) for h, v in acc.items()}


def _persistence(p1: float, h: int) -> list:
    """House baseline fallback: every hour stays at hour 1, spread widening like a random walk."""
    s = FALLBACK_SIGMA * math.sqrt(h - 1)
    qs = [round2(min(1e9, max(1.0, p1 * math.exp(z * s)))) for z in Z]
    for i in range(1, 5):
        if qs[i] < qs[i - 1]:
            qs[i] = qs[i - 1]
    return qs


def _training_set(case: dict) -> list:
    """[(log hour-1 viewers, {hour: log(v_h / v_1)})] for every usable comparable launch."""
    train = []
    for c in case.get("comparables") or []:
        try:
            day_start = c.get("day_start") or case.get("day_start")
            ds = _ts(day_start)
            hours = _day_hours(c.get("hourly") or [], ds)
        except Exception:
            continue
        if 1 not in hours or hours[1] <= 0:
            continue
        v1 = hours[1]
        ratios = {}
        for h in range(2, 25):
            if h in hours and hours[h] > 0:
                ratios[h] = math.log(hours[h] / v1)
        if ratios:
            train.append((math.log(v1), ratios))
    return train


def _forecast_hour(p1: float, h: int, train: list) -> list:
    log_p1 = math.log(p1)
    pairs = []
    for log_v1, ratios in train:
        if h not in ratios:
            continue
        d = (log_p1 - log_v1) / BW_SIZE
        w = math.exp(-0.5 * d * d)
        if log_v1 < math.log(TINY_H1):
            w *= TINY_WEIGHT
        pairs.append((ratios[h], w))
    if not pairs:
        return _persistence(p1, h)
    raw = [_wquantile(pairs, tau) for tau in LEVELS]
    med = raw[2]
    qs = []
    for qv in raw:
        v = p1 * math.exp(med + (qv - med) * SPREAD)
        qs.append(round2(min(1e9, max(1.0, v))))
    for i in range(1, 5):
        if qs[i] < qs[i - 1]:
            qs[i] = qs[i - 1]
    return qs


def forecast_case(case: dict) -> dict:
    """Forecast one case. Never raises: falls back to persistence per hour."""
    try:
        first = next((k for k in case.get("known") or [] if k.get("hour") == 1), None)
        p1 = max(1.0, float(first["viewers"])) if first is not None else 1.0
    except Exception:
        p1 = 1.0
    try:
        train = _training_set(case)
    except Exception:
        train = []
    try:
        hours = [int(r["hour"]) for r in case.get("forecast") or []]
    except Exception:
        hours = []
    if not hours:
        hours = list(range(2, 25))
    rows = []
    for h in sorted(set(hours)):
        if h < 2 or h > 24:
            continue
        try:
            q = _forecast_hour(p1, h, train)
        except Exception:
            q = _persistence(p1, h)
        rows.append({"hour": h, "q": q})
    # Exactly 23 rows, hours 2..24 each once, even if the case asked otherwise.
    have = {r["hour"] for r in rows}
    for h in range(2, 25):
        if h not in have:
            rows.append({"hour": h, "q": _persistence(p1, h)})
    rows.sort(key=lambda r: r["hour"])
    return {"hours": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        try:
            case = json.loads(path.read_text(encoding="utf-8"))
            forecast = forecast_case(case)
        except Exception:
            forecast = {"hours": [{"hour": h, "q": _persistence(1.0, h)} for h in range(2, 25)]}
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
