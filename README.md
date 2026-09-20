# Fly Connectome Engine

Runs the MaleCNS v1.0 *Drosophila* connectome as a live spiking neural simulation driven by real
visual input, and tests whether the specific wiring does anything measurable against
shuffled-wiring null models.

- **166,700** annotated neurons, **14,806,510** synaptic edges (weight ≥ 2, both endpoints annotated)
- Leaky integrate-and-fire, exponential-Euler integration, event-driven propagation on GPU
- Signs derived from the connectome's own neurotransmitter predictions (98.1% coverage)
- Spec: `docs/superpowers/specs/2026-09-19-fly-connectome-engine-design.md`

## Validation

**The pre-committed pass criterion FAILED on both metrics.** It required the real connectome to
beat the degree-preserving null on direction selectivity *and* looming discrimination with
non-overlapping 95% confidence intervals. It did not. The criterion was fixed in advance
precisely so this outcome could not be renegotiated after seeing the data, and it has not been.

25 conditions (5 seeds × [real + 4 nulls]), 2,894 s on an RTX 2050. Every condition was
calibrated by the identical homeostatic procedure.

| Metric | real | weight_perm | degree_preserving | sign_perm | type_preserving |
|---|---|---|---|---|---|
| **DSI** | −0.00056 | +0.00014 | +0.00012 | +0.00118 | +0.00066 |
| **Looming** | −0.01219 | −0.00994 | +0.00003 | −0.00117 | −0.01877 |

The real connectome is deterministic, so its confidence interval is a point and the comparison is
a permutation test: does the real value fall outside each null's distribution?

### Direction selectivity: no signal anywhere

Every condition sits within ±0.002 of zero — the real wiring included. Measured per subtype, T4a
fires 1,691 spikes to a 0° grating and 1,692 to 180°.

This is a limitation of the model, **not evidence about the wiring**. T4/T5 direction selectivity
works by comparing a *delayed* input (Mi9, Mi4/CT1) against a *non-delayed* one (Mi1, Tm3) from
neighbouring columns. Every neuron here shares τ_m = 20 ms and τ_syn = 5 ms, so there is no
differential delay and the correlation that produces direction selectivity cannot form. This is
why Lappalainen et al. (flyvis) *train* 734 parameters — per-cell-type time constants, resting
potentials and synaptic strengths. **Connectome topology with uniform biophysics appears
insufficient for direction selectivity.** Reproducing it would require fitting time constants,
which is a different and much larger project.

### Looming: real wiring differs from the nulls, but in the wrong direction

The real network's escape pathway (DNp01/02/03 gated on LC4 + LPLC2) fires *less* to an expanding
dark disc than to a flat control — the opposite of the biology, where looming drives Giant Fiber
escape. So the criterion fails on sign, not only on magnitude.

There is nevertheless real structure in which nulls destroy the effect:

| Null | What it preserves | Looming | Real distinguishable? |
|---|---|---|---|
| weight_permutation | topology + signs; shuffles magnitudes | −0.00994 | **no** |
| type_preserving | type-to-type counts; shuffles partners | −0.01877 | yes |
| sign_permutation | topology; shuffles which neurons inhibit | −0.00117 | yes |
| degree_preserving | degree sequences only; destroys topology | +0.00003 | yes |

Destroying topology abolishes the effect entirely. Shuffling *which* neurons are inhibitory
mostly abolishes it. Shuffling synaptic magnitudes while keeping topology intact does not change
it measurably. Read narrowly, the effect depends on topology and on the placement of inhibition,
but not on precise synaptic weights.

That reading should be treated with caution, because the effect's sign is biologically backwards.
The most likely explanation is that the model's excitation/inhibition balance is wrong: the
looming stimulus delivers net inhibition to the escape circuit rather than net excitation, and
baseline activity is dominated by the tonic drive (see Caveats).

## Caveats

- **`i_tonic` is a free parameter I introduced**, not a literature value. The fly's ON pathway
  signals by disinhibition, which cannot modulate a silent neuron, so without a standing baseline
  the pathway is mute — T4 fired exactly zero times across an entire battery before it was added.
  It is applied identically to real and null conditions, so the comparison stays fair, but it
  weakens any claim that the model uses only published parameters.
- **Uniform time constants** across all 166,700 neurons. This is the limitation that makes the
  direction-selectivity arm uninformative.
- **Modulators (dopamine, octopamine, serotonin) are zeroed** — they do not fit a two-sign LIF
  model. This is why no `valence` signal is emitted.
- **Capture is 30 fps against a ~200 Hz flicker-fusion visual system**, bounding what the motion
  detectors can see.
- The real condition's five seeds are identical because the simulation is deterministic; its
  interval is a point, not a sampled distribution.

## What is verified

| Property | Result |
|---|---|
| Membrane dynamics vs closed-form solution | max error **1.42e-14** |
| GPU vs CPU spike-raster agreement | **100.000000%**, max \|Δv\| 2.8e-14 |
| Monosynaptic latency (LPLC2 → partners) | median **1.00 ms**, 201/235 responded |
| Dale's law across the compiled graph | holds for every sampled neuron |
| Artifact hash reproducibility | identical across rebuilds |
| Event-driven scaling (quiet vs busy) | **3.89×** (dense would be ~1.0) |

## Reproducing

```bash
python -m flybrain.ingest.build_graph 2   # compile the graph artifact (~45 s)
pytest tests/ -q                          # 118 tests
python -u run_battery.py                  # full battery (~48 min)
```
