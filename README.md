# FruitFly

A spiking neural simulation of the *Drosophila* MaleCNS connectome.

## Validation

The validation battery (`flybrain/analysis/battery.py`) runs the real connectome and
four shuffled-wiring null models (`weight_permutation`, `degree_preserving`,
`sign_permutation`, `type_preserving`) through an identical stimulus protocol
(drifting gratings in both directions, a looming disc, and a flat contrast
control), at five seeds each, and asks a question fixed in advance: does the
real wiring outperform its degree-preserving shuffle on both direction
selectivity (DSI) and looming discrimination, with non-overlapping 95%
bootstrap confidence intervals?

**Verdict:** `dsi_pass=False, looming_pass=False. The real wiring did NOT
measurably outperform its shuffle. Report this in the README; it is a
finding, not a failure.`

**overall = False.**

Every condition -- the real graph and all four nulls, at all five seeds --
produced a direction-selectivity index and a looming-discrimination score of
exactly `0.0`, so every confidence interval collapses to `[0.0, 0.0]`:

| metric | condition | mean | 95% CI |
|---|---|---|---|
| dsi | real | +0.0000 | [+0.0000, +0.0000] |
| dsi | weight_permutation | +0.0000 | [+0.0000, +0.0000] |
| dsi | degree_preserving | +0.0000 | [+0.0000, +0.0000] |
| dsi | sign_permutation | +0.0000 | [+0.0000, +0.0000] |
| dsi | type_preserving | +0.0000 | [+0.0000, +0.0000] |
| looming | real | +0.0000 | [+0.0000, +0.0000] |
| looming | weight_permutation | +0.0000 | [+0.0000, +0.0000] |
| looming | degree_preserving | +0.0000 | [+0.0000, +0.0000] |
| looming | sign_permutation | +0.0000 | [+0.0000, +0.0000] |
| looming | type_preserving | +0.0000 | [+0.0000, +0.0000] |

The pass criterion was committed in advance and was not adjusted after seeing
this result. The exact-zero pattern across every seed and every wiring
(rather than merely overlapping-but-nonzero intervals) suggests the watched
populations (T4/T5, and the DNp01/LPLC2/LC4 escape pathway) recorded no
spikes at all during the stimulus windows in this configuration, in both the
real graph and every shuffle -- a calibration/observability finding about
this run's tuning and stimulus regime, not evidence that wiring does or does
not matter. The battery, its pass criterion, and this result are recorded
in `out/battery.json` and `figures/battery.png`.

Battery wall-clock: 2669.5 s (44.49 min) for 25 conditions (5 seeds x
[real + 4 nulls]) on an RTX 2050, run 2026-09-20.
