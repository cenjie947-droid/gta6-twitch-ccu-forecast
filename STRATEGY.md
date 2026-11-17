# Strategy

## The problem

Forecast Twitch concurrent viewers in the Grand Theft Auto VI category for
hours 2–24 of release day 2026-11-19 (UTC), given only the hour-1 print and
the case file. Scored by mean pinball loss on the log scale, five quantiles
(0.05, 0.25, 0.5, 0.75, 0.95) per hour. A place needs a final loss strictly
below the best house entry (house baseline or house model forecast,
whichever is lower).

## What the data says

Each case ships ~38 comparables: past launches with full hourly curves from
14 days before to 3 days after their release day (SullyGnome hourly
averages). Two facts drive the whole model:

1. **Release days have a shape, not a level.** Viewers dip in UTC hours
   2–5 (night in the Americas), then rise into the US day, peaking around
   hours 19–22. The median log-ratio to hour 1 runs from about −0.3 at hour
   4 to +0.9 at hour 19 across all comparables.
2. **Launch size predicts the shape.** Log hour-1 viewers correlates −0.9
   with the day's rise (mean log-ratio over hours 14–20). Huge hour-1
   launches (Elden Ring 875k, Cyberpunk 2077 852k) unlocked everywhere at
   once and stay almost flat (rise ≈ −0.2 to +0.1). Small hour-1 launches
   (Hollow Knight: Silksong 164, Helldivers 2 419) had not unlocked yet in
   most regions at UTC hour 1 and multiply 100×+ later in the day. Hour 1
   is, in effect, a measurement of *when* the game unlocked relative to
   UTC midnight.

A pre-release-baseline "unlock timing" feature (hour 1 vs pre-release
buzz) was tried and dropped: it is noisy (sparse pre-release sampling)
and adds nothing once size is in the model.

## The model

Per hour, kernel-weighted quantiles of log-ratios `log(v_h / v_1)` over
the comparables, Gaussian kernel on log hour-1 viewers (bandwidth 1.0),
tiny hour-1 launches downweighted 0.3×, outer quantiles scaled 0.85×
around the median. The forecast is scale-free (ratios only), so an
hour-1 print beyond any past launch still yields the biggest launches'
shape with no extrapolation blow-up. No per-case tuning, no randomness,
no network: the same program runs on practice and final cases.

## Validation

Leave-one-out cross-validation over the kit's 37 comparables, scoring
with the exact pinball loss:

| model | mean | launches with hour 1 ≥ 50k | launches 5k–50k |
|---|---|---|---|
| persistence baseline | 0.93 | 0.25 | 0.60 |
| this model | 0.41 | 0.21 | 0.41 |

The baseline's mean is dragged up by launches whose hour 1 was nearly
empty (losses 3.4–3.9); the model fixes those by reading the shape from
similarly-sized launches. The final case is one launch, almost certainly
huge (GTA VI unlocks at rolling midnight local, NZ/Europe/UK live by
hour 1, US unlocks at hours 6 and 9), so the big-launch column is the
one that matters — and the model's edge there is real but modest, which
is honest: the final is a single draw, and the model is built to be the
best unbiased shape estimate for it, not to chase the practice board.
