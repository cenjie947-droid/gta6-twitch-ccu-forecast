[![make.fun](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/cenjie947-droid/gta6-twitch-ccu-forecast/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/cenjie947-droid/gta6-twitch-ccu-forecast/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)

# GTA VI Twitch CCU forecast

Entry for the [GTA VI Twitch CCU forecast](https://make.fun/bounty/gta6-twitch-ccu-forecast-01) competition on [make.fun](https://make.fun): forecast Twitch concurrent viewers in the Grand Theft Auto VI category for hours 2–24 of release day, 2026-11-19 (UTC), from the hour-1 print and the release-day curves of past launches.

## Method

Work in log-ratios to hour 1. For each past launch in the case's `comparables`, compute `y_h = log(v_h / v_1)` per hour. Launch size (log hour-1 viewers) is the dominant predictor of a release day's shape: launches whose hour 1 is already huge unlocked everywhere at once and stay flat, while launches whose hour 1 is small still have their big regional unlocks coming and rise steeply later. For each hour, take kernel-weighted quantiles of `y_h` over the comparables, with a Gaussian kernel on log hour-1 viewers (bandwidth 1.0), downweighting tiny hour-1 launches whose ratios are noisy. Outer quantiles are scaled by 0.85 around the median, calibrated by leave-one-out cross-validation over the kit's 37 comparables (mean pinball loss 0.41 vs 0.93 for the persistence baseline; 0.21 vs 0.25 on launches with hour 1 ≥ 50k). See [STRATEGY.md](STRATEGY.md) for the full write-up.

## Run

`python3 /bot/main.py` (Python 3.13, standard library only, no network). It reads every `<id>.in` in `$MAKEFUN_INPUT_DIR` (else `/input`) and writes `<id>.out` to `$MAKEFUN_OUTPUT_DIR` (else `/output`): 23 rows, hours 2–24, five quantiles each. Deterministic, well under the 8-second limit.

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```
