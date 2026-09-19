# Fly Connectome Engine — Design Spec

**Date:** 2026-09-19
**Status:** Approved for planning
**Scope:** v1 — the simulation engine and its avatar output. Silent (no speech).

---

## 1. Goal

Run the MaleCNS v1.0 *Drosophila* connectome as a live spiking neural simulation, driven by real visual input, and use its population activity to drive a VTuber avatar's reactions.

The engine must be scientifically honest: every signal it emits names the neurons it comes from, and the claim "the wiring matters" is tested against shuffled-wiring null models rather than asserted.

### Non-goals for v1

Speech, LLM integration, memory, moderation, streaming, and platform compliance are **out of scope**. They attach later as subscribers to the signal bus (see §9) without modifying the engine. This boundary is the primary defense against scope creep.

---

## 2. Verified data facts

All figures below were measured directly from the local files, not taken from publications.

| Fact | Value |
|---|---|
| Total synaptic edges (minconf 0.5) | 151,856,684 |
| Edges with weight >= 2 | 57,670,765 (38.0%) |
| ...of those, both endpoints annotated | 15,283,237 (26.5% of w>=2) |
| **Edges in the compiled graph (T=2)** | **14,806,510** after dropping sign-0 sources |
| Edges with weight >= 3 | 23,014,406 (15.2%) |
| Max edge weight | 2,591 |
| Annotated neurons (with superclass) | 166,700 |
| Neurons with definite neurotransmitter | 163,523 (98.1%) |
| Optic-lobe hex columns | 892 per lobe |
| Descending neurons | 1,314 |
| VNC motor neurons | 708 |
| Photoreceptors (ol_sensory) | 6,098 |

Key cell-type counts: L1 1,776 · L2 1,779 · Mi1 1,773 · T4a/b ~1,687 each · T5a/b ~1,690 each · LPLC2 185 · LC4 126 · DNp01 (Giant Fiber) 2.

Full retinotopic tiling is available: L1, L2, L5, Mi1, Tm1, Tm2, Tm9, Mi4, Mi9, C3 and T1 each carry `assignedOlHex1`/`assignedOlHex2` coordinates across all 892 columns. Photoreceptors themselves carry **no** column assignment, which determines the input path (§6).

---

## 3. Architecture

Seven modules, no circular dependencies:

```
flybrain/
  ingest/      feather -> compiled graph artifact   (offline, run once)
  core/        LIF simulation over a compiled graph (pure, no I/O)
  encode/      pixels -> per-column input currents
  readout/     spike trains -> named signals
  nulls/       graph rewiring for control models
  drivers/
    batch      stimulus script -> recorded activity -> analysis
    live       real-time loop, own process, fixed clock
  bridge/      signal bus -> VTube Studio
```

**Hard rule:** `core` imports nothing from `encode`, `readout` or `drivers`. It accepts an input-current vector, advances one timestep, exposes a spike mask. Because both drivers sit on the identical core, validation results transfer to what appears on screen.

---

## 4. Ingest — compiled graph artifact

Run once, offline. Produces a versioned artifact the simulator memory-maps.

1. **Stream and threshold.** Iterate the 2,318 Arrow record batches, keep `weight >= T`. **Default T = 2 -> 57,670,765 edges.** Never materialize all 151.9M rows.
2. **Restrict to real neurons.** Inner-join both endpoints against the 166,700 bodies carrying a `superclass`. Drops orphan fragments.
3. **Assign signs from neurotransmitter** (presynaptic property, applied to all out-edges — Dale's law):

   | Neurotransmitter | Sign | Note |
   |---|---|---|
   | acetylcholine | +1 | excitatory |
   | GABA | -1 | inhibitory |
   | glutamate | -1 | GluCl-alpha is inhibitory in *Drosophila* |
   | histamine | -1 | photoreceptor -> LMC sign inversion |
   | dopamine / octopamine / serotonin | 0 | modulators; v1 limitation, see §7 |
   | unclear / missing (1.9%) | fallback | cell-type consensus, then excluded |

   The sign map is **configuration, not code**, so glutamate's polarity can be flipped and re-run as a sensitivity check.
4. **Reindex** `bodyId` -> dense `0..N-1`; build CSR; weights fp32, indices int32.
5. **Emit** `graph.npz` + `manifest.json` recording threshold, edge/neuron counts, NT policy, dropped-edge tallies, and a content hash.

The manifest hash is what makes null comparison meaningful: real and shuffled graphs come from the same pipeline and differ only in the rewiring step. **The simulator refuses to start on a hash mismatch** — every result must be traceable to an exact build.

---

## 5. Simulation core

### Model

Leaky integrate-and-fire, exponential synapses, one compartment per neuron. Same model class as Shiu et al. (*Nature* 2024), which applied synapse-count-scaled LIF to the FlyWire whole-brain connectome and predicted real sensorimotor responses. Defaults (all configurable):

- resting -52 mV, threshold -45 mV, reset to rest
- refractory ~2.2 ms, membrane tau ~20 ms, synaptic tau ~5 ms
- weight ~0.275 mV x synapse count x sign x per-type gain

### The hot loop is event-driven — mandatory, not an optimization

Dense sparse-matrix x spike-vector touches every edge each step. Measured at build time, the compiled graph holds **14,806,510** edges -- not the 57.7M that survive thresholding, because 73.5% of those have a postsynaptic partner never traced to an annotated neuron, and an unidentifiable body cannot be simulated. That is ~118 MB of traffic per dense step rather than the ~460 MB first estimated, putting the dense ceiling near **950 steps/s** -- still at or below real time, and still the reason to stay event-driven.

Event-driven propagation touches only the out-edges of neurons that actually fired.

**Measured on the target RTX 2050 with the real 14.8M-edge graph, a full step runs far below the ~10,000 steps/s this analysis originally predicted.** The prediction counted memory traffic and ignored kernel-launch latency, which dominates: at 166,700 neurons each elementwise operation moves only 0.67 MB, so a step is a sequence of ~15 short kernels rather than a bandwidth-bound job. Measured full-step throughput:

| Implementation | steps/s | vs real time |
|---|---|---|
| Event-driven gather, with per-step CPU syncs | 230 | 0.23x at dt=1ms |
| Event-driven gather, sync-free | 271 | 0.27x at dt=1ms |
| Fixed-shape dense spMV | 435 | 0.87x at dt=2ms |

Two conclusions follow, and both correct claims made earlier in this document. First, **dense propagation is not a build failure at this graph size** -- it is the faster of the two, because at ~2.5% spiking with average out-degree 89 the gather's index arithmetic costs more than streaming 59 MB of weights. Event-driven remains the better choice at low activity and is kept, but the claim that dense caps at 240 steps/s was computed against an edge count 4x too high. Second, **real-time simulation at dt=1ms is not achievable on this hardware**; dt=2ms is close and dt=4ms clears it. `torch.compile`/CUDA graphs, which would attack the launch overhead directly, are unavailable (no Triton on Windows).

Per-step CPU-GPU synchronisation is therefore forbidden in the hot loop: no `int(x.sum())`, no `bool(x.any())`. Those alone cost ~18%.

Per step: `spikes.nonzero()` -> build a flat gather index into those rows' CSR ranges via `repeat_interleave` -> one `index_add_` scattering signed weights into synaptic current.

**Runaway detector:** step cost rises with spike count, so a spike-fraction ceiling sustained over N steps triggers protection (see §10).

### Calibration — homeostatic tuning

Per-cell-type gains adjusted until median firing rates sit in plausible bands (~1–20 Hz): run a standard stimulus, measure per-type rates, scale each gain by `(target/observed)^eta`, repeat ~20 rounds, freeze, record in the manifest.

**Critical rule: the identical tuning procedure runs on every null model.** Tuning the real network into working while leaving controls to fail would invalidate the entire comparison.

---

## 6. Sensory encoding

A `FrameSource` interface with three implementations — webcam, screen capture, and a synthetic generator for the battery — all emitting timestamped grayscale float arrays. Source is a runtime switch.

**Pixels -> columns.** The hex lattice is precomputed as a sparse sampling matrix; each of the 892 columns per eye draws a Gaussian-weighted patch, so a frame becomes column luminances in one sparse multiply. Both eyes sample the same frame with a horizontal offset, approximating binocular overlap. The fly's real ~270° field far exceeds a webcam's ~65°; v1 maps the frame onto the central field and documents that the periphery is dark.

**Luminance -> lamina current.** Photoreceptors have no column assignment, so input enters one synapse downstream at the lamina. This is faithful rather than a compromise: the photoreceptor→LMC synapse is histaminergic and inhibitory, a sign inversion already present in the NT data. The encoder applies divisive adaptation against a running luminance mean (without which any light change saturates the model), then drives L1 and L2 — the ON and OFF pathway inputs — with opposite contrast polarity.

**Documented limitation:** 30 fps capture against a ~200 Hz flicker-fusion visual system means the motion detectors see an undersampled world. 60 fps capture helps; it never fully closes. Recorded in the manifest because it bounds what T4/T5 can possibly report.

---

## 7. Readout and the honesty contract

**Every emitted signal names the exact neurons it derives from, in a versioned schema, and the visualization can display that provenance live.** This is structural, not a promise.

| Signal | Source neurons | Basis |
|---|---|---|
| `startle` (event + magnitude) | DNp01/02/03, gated on LC4 + LPLC2 | The real escape circuit |
| `flow` (2D vector per eye) | T4a/b/c/d, T5a/b/c/d population vector | Genuine direction selectivity |
| `salience` (visual-field position) | LC4, LC6, LPLC1/2 object detectors | Where something notable is |
| `arousal` (slow scalar) | Central-brain population rate, leaky integrator | Persistence + scalability |

### `valence` is deliberately absent

Fly valence runs substantially through dopamine and octopamine, which v1 sets to **zero** because modulators do not fit a two-sign LIF model. Emitting a "connectome-derived valence" would invent precisely what the data cannot support. v1 ships `arousal`, which it can justify, and records why `valence` is missing. It arrives when modulators get proper treatment.

---

## 8. Validation and null models

### Nulls, increasing severity

1. **Weight permutation** — shuffle weights across existing edges. Does strength matter?
2. **Degree-preserving rewire** — configuration model preserving in/out degree. The standard null.
3. **Sign permutation** — shuffle NT labels. Does E/I placement matter?
4. **Type-preserving rewire** — preserve cell-type-to-cell-type statistics, randomize individual partners. Strictest: does the *specific* wiring matter beyond the type-level summary?

All built by the same pipeline; only the rewiring step differs.

### Battery

1. **Engine test (encoder-free).** Inject current directly into an identified presynaptic population; verify known postsynaptic partners respond with correct sign and ~1–3 ms per-synapse latency, checked against the connectivity data itself. This isolates engine faults from encoder faults: if this passes and the visual tests fail, the encoder is the problem.
2. **Drifting gratings** -> T4/T5 direction-selectivity index.
3. **Expanding dark discs**, varied r/v ratios -> LPLC2 tuning and Giant Fiber spike timing. The looming → LPLC2/LC4 → DNp01 escape pathway is among the best-characterised circuits in neuroscience, giving published response properties to compare against.
4. **Contrast steps** -> L1/L2 ON/OFF polarity.

Each runs on the real graph and >= 5 seeds per null, all homeostatically tuned by the identical procedure, reported with effect sizes and confidence intervals.

### Pass criterion — committed in advance

The real connectome must beat degree-preserving nulls on direction selectivity and looming discrimination with **non-overlapping confidence intervals**.

If it does not, that is the finding. It goes in the README. The character still works — it reacts to a network whose specific wiring did not measurably outperform its shuffle, and the project will be one of the few that actually checked.

---

## 9. Live driver, bus, and VTube Studio bridge

Three processes:

```
[capture + encode] --shared mem ring--> [sim core, fixed clock] --WebSocket--> [VTS bridge] --> VTube Studio
                                                                      \-----> [debug viewer in browser]
```

**WebSocket bus.** VTube Studio's plugin API is already a WebSocket (port 8001) and a browser debug viewer is WebSocket-native, so bridge and visualizer are both just clients. Later consumers — LLM layer, OBS overlay, recorder — subscribe without touching the engine. JSON packets at 60 Hz carrying a handful of floats are negligible.

**Clock discipline.** The core runs ~10,000 steps/s but needs 1,000 for real time, so it throttles against the wall clock with a configurable speed multiplier and logs drift. Input is read non-blocking; with no new frame the last persists. A camera stall means the fly sees a static scene, never a stuttering simulation.

**Backpressure never reaches the core.** Slow subscribers get packets dropped, not honored.

**Avatar:** VTube Studio (free) with its bundled Live2D models (Hiyori/Mao/Mark/Akari), which already expose standard parameters; custom parameters are one API call. A commissioned model can replace it later without engine changes.

**Signal -> avatar mapping lives in YAML**, not code: signal, VTS parameter, scale, smoothing time constant, decay. Tuning how twitchy the character feels is editing a config file.

**All smoothing and easing happens in the bridge, never in the core.** The core emits the raw truth that validation recorded; the bridge shapes it for presentation. If they ever disagree, it is provable which is which.

---

## 10. Testing and failure behavior

### Testing philosophy

Test-first. **Every phase asserts specific known values — no smoke tests.** Each phase also ends with a manual verification the human performs directly.

- **Core** — hand-built 3-neuron graphs with analytically known LIF solutions; membrane decay verified against closed-form exponential to 1e-6; threshold, refractory and sign propagation exact. Property-based tests via hypothesis.
- **Ingest** — small-slice golden manifest; explicit table test for the NT→sign map, since it is configuration that silently changes every downstream result.
- **Encoder** — known image patterns -> known column values; adaptation step response.
- **Readout** — synthetic spike trains -> expected signals, including silent-population cases.
- **Integration** — engine test (fast, CI); full battery marked slow, run deliberately.

### Phase gates

| Phase | Automated test asserting known values | Manual verification |
|---|---|---|
| 0 · Env + data guard | Exactly 151,856,684 edges / 166,700 neurons; CUDA present | Numbers print |
| 1 · Ingest -> graph | Edge count at T=2 exactly 57,670,765; Dale's law holds; LPLC2→DNp01 present; manifest hash reproducible | Read `manifest.json` |
| 2 · LIF core, CPU | Membrane decay matches closed form to 1e-6; refractory exact | Plot V(t) vs analytic curve |
| 3 · GPU, full graph | GPU matches CPU within fp tolerance over 1,000 steps; asserts >= 1,000 steps/s | Watch steps/s and VRAM |
| 4 · Engine test | Latency 1–3 ms/synapse, correct sign, partners match connectome | Latency histogram |
| 5 · Encoder | Known patterns -> known column values; adaptation response | Webcam frame beside hex rendering |
| 6 · Battery + nulls | DSI real vs 5 null seeds; CIs computed | The comparison figure |
| 7 · Readout + bus | Synthetic spikes -> expected signals; WebSocket schema contract | Browser debug viewer |
| 8 · VTS bridge | Mapping YAML -> injection against mock VTS server; reconnect under failure | Wave at webcam, avatar reacts |

### Failure behavior

| Failure | Response |
|---|---|
| Camera/screen source lost | Encoder holds last frame, sets `stale`; sim continues |
| VTube Studio disconnects | Bridge reconnects with backoff; sim unaffected |
| Runaway activity | **Live:** clamp gain, log, emit `degraded`. **Batch:** halt, record failure |
| Subscriber too slow | Drop packets |
| Graph artifact hash mismatch | Refuse to start |

Live and batch fail differently on purpose: a frozen stream is dead, but an experiment that silently clamps gain and keeps recording produces contaminated data.

### Performance budget (RTX 2050, 4 GB)

Graph ~118 MB VRAM (measured, 14.8M edges) · neuron state < 10 MB · encode ~1 ms/frame · sim step ~3.7 ms measured at nominal activity (not the 0.1 ms first estimated) · **end-to-end photon-to-parameter latency target < 50 ms.**

---

## 11. Environment

Python 3.11.5. Already present: torch 2.13.0+cu126 (CUDA available), numpy 2.4.6, pyarrow 25.0.1, pandas 2.3.3, opencv-python 5.0.0.93, scipy 1.17.1, numba 0.67.0, networkx 3.6.1, websockets 15.0.1, pytest 9.1.1, pytest-asyncio 1.4.0, hypothesis 6.165.10, matplotlib 3.11.1, pillow 12.3.0.

To add: `mss` (screen capture), `pyyaml` (mapping config).

Data files stay on disk and out of git (1.05 GB); `.gitignore` excludes `*.feather`.

---

## 12. Principal risks

1. **The network may not do anything interesting.** Connectome LIF models are highly sensitive to gain; the plausible-rate band may be narrow or empty. Mitigated by homeostatic tuning, and the Phase 4 engine test catches it before the battery.
2. **Nulls may match the real wiring.** Genuinely possible — it is what honest prior projects found. Treated as a reportable result, not a failure.
3. **30 fps into a ~200 Hz visual system** bounds what T4/T5 can report. Documented; partially mitigated by 60 fps capture.
4. **Modulators are zeroed**, which is exactly the system underlying fly arousal. This is why `valence` is absent from v1.
5. **Scope creep toward the talking VTuber.** The bus contract is the defense: later subsystems attach without modifying the engine.

---

## 13. Deferred (not part of v1)

Conversational stack (LLM + STT + TTS + memory + moderation), streaming and platform integration, and legal compliance — AI-disclosure under CA SB 243 and EU AI Act Article 50, self-harm crisis protocol, YouTube synthetic-content labelling. These become mandatory the moment the character talks to the public; they are out of scope only because v1 is silent and private.
