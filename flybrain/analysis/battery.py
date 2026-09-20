"""The validation battery and its pre-committed pass criterion.

Real and null graphs run through an identical pipeline: same stimuli, same
tuning procedure, same number of rounds. Any divergence would mean the
comparison measures the procedure rather than the wiring.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from flybrain.analysis.metrics import (
    bootstrap_ci,
    direction_selectivity_index,
    intervals_overlap,
    looming_discrimination,
)
from flybrain.core.lif import LIFNetwork, LIFParams
from flybrain.core.tuning import homeostatic_tune
from flybrain.drivers.batch import run_batch
from flybrain.encode.encoder import LaminaEncoder
from flybrain.encode.source import SyntheticSource
from flybrain.nulls.rewire import NULL_MODELS
from flybrain.stimuli.patterns import contrast_step, drifting_grating, looming_disc

HEIGHT, WIDTH = 120, 160
N_FRAMES = 90


def _watch_groups(graph) -> dict[str, np.ndarray]:
    return {
        "T4": graph.types_matching("T4"),
        "T5": graph.types_matching("T5"),
        "escape": np.concatenate(
            [graph.type_index("DNp01"), graph.type_index("LPLC2"), graph.type_index("LC4")]
        ),
        "L1": graph.type_index("L1"),
        "L2": graph.type_index("L2"),
    }


def _tune_groups(graph) -> dict[str, np.ndarray]:
    groups = {}
    for name in ("ol_intrinsic", "cb_intrinsic", "vnc_intrinsic", "visual_projection"):
        idx = graph.superclass_index(name)
        if len(idx):
            groups[name] = idx
    return groups


def _tuning_drive(net, seed: int = 12345, n_seed_neurons: int = 2000):
    """A fixed, reproducible calibration stimulus.

    Two requirements pull in opposite directions and both must be met.

    Tuning against zero input is meaningless: with nothing firing, the update
    rule can only ratchet gains upward forever. But a uniform drive strong
    enough to make neurons fire BY ITSELF is just as bad, and less obviously
    so. `gain` scales only synaptic contributions, so any neuron driven over
    threshold by the external current alone fires at a rate no gain setting can
    change -- calibration then silently does nothing.

    The synaptic filter amplifies a constant drive by 1/(1 - exp(-dt/tau_syn))
    = 5.52x at the defaults, and threshold sits 7 mV above rest, so any uniform
    drive above ~1.27 mV/step is self-igniting. (An earlier version of this
    function used rand*2.0 and was mostly above that line.)

    So: a subthreshold background everywhere, plus a sparse set of seed neurons
    driven hard enough to fire and start recurrent activity. Population rates
    are then genuinely a function of gain, which is what calibration adjusts.
    Deterministic, and identical for real and null graphs.
    """
    generator = torch.Generator(device="cpu").manual_seed(seed)
    drive = torch.full((net.n_neurons,), 0.3)          # 0.3 * 5.52 = 1.66 mV, subthreshold
    seeds = torch.randperm(net.n_neurons, generator=generator)[:n_seed_neurons]
    drive[seeds] = 14.0                                 # comfortably suprathreshold
    drive = drive.to(net.device, net.i_syn.dtype)
    return lambda _: drive


def _one_condition(graph, indptr, indices, weights, device, seed) -> dict:
    net = LIFNetwork(indptr, indices, weights, LIFParams(), device=device)
    encoder = LaminaEncoder(graph, HEIGHT, WIDTH)
    watch = _watch_groups(graph)

    # Identical tuning for real and null -- this is the fairness guarantee.
    homeostatic_tune(
        net,
        _tune_groups(graph),
        _tuning_drive(net),
        target_hz=5.0,
        rounds=5,
        steps_per_round=200,
    )

    def sweep(generator):
        source = SyntheticSource(generator, HEIGHT, WIDTH)
        return run_batch(net, encoder, source, watch, n_frames=N_FRAMES, steps_per_frame=33)

    preferred = sweep(drifting_grating(HEIGHT, WIDTH, 0.0))
    opposite = sweep(drifting_grating(HEIGHT, WIDTH, 180.0))
    loom = sweep(looming_disc(HEIGHT, WIDTH, n_frames=N_FRAMES))
    flat = sweep(contrast_step(HEIGHT, WIDTH, low=0.5, high=0.5, switch_frame=10_000))

    dsi = direction_selectivity_index(
        float(preferred["T4"].sum() + preferred["T5"].sum()),
        float(opposite["T4"].sum() + opposite["T5"].sum()),
    )
    loom_score = looming_discrimination(loom["escape"], flat["escape"])

    # Free the graph before the next condition allocates its own. Twenty-five
    # conditions at ~460 MB each will not coexist on a 4 GB card.
    del net, encoder
    if device == "cuda":
        torch.cuda.empty_cache()

    return {"seed": seed, "dsi": dsi, "looming": loom_score}


def run_battery(
    graph, device: str = "cuda", seeds=(0, 1, 2, 3, 4), out_dir: Path = Path("out")
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    conditions: dict[str, list[dict]] = {"real": []}

    for seed in seeds:
        print(f"[battery] real seed={seed} starting", flush=True)
        conditions["real"].append(
            _one_condition(graph, graph.indptr, graph.indices, graph.weights, device, seed)
        )
        print(f"[battery] real seed={seed} done: {conditions['real'][-1]}", flush=True)

    for name, fn in NULL_MODELS.items():
        conditions[name] = []
        for seed in seeds:
            print(f"[battery] {name} seed={seed} starting", flush=True)
            args = (
                (graph.indptr, graph.indices, graph.weights, graph.types, seed)
                if name == "type_preserving"
                else (graph.indptr, graph.indices, graph.weights, seed)
            )
            p, i, w = fn(*args)
            conditions[name].append(_one_condition(graph, p, i, w, device, seed))
            print(f"[battery] {name} seed={seed} done: {conditions[name][-1]}", flush=True)

    results: dict = {"dsi": {}, "looming": {}, "raw": conditions}
    for metric in ("dsi", "looming"):
        for name, runs in conditions.items():
            values = np.array([r[metric] for r in runs], dtype=float)
            results[metric][name] = {
                "mean": float(values.mean()),
                "values": values.tolist(),
                "ci": list(bootstrap_ci(values, seed=0)),
            }

    results["verdict"] = evaluate_pass_criterion(results)
    (out_dir / "battery.json").write_text(json.dumps(results, indent=2))
    return results


def evaluate_pass_criterion(results: dict) -> dict:
    """Committed in advance: the real connectome must beat the
    degree-preserving null on BOTH metrics with non-overlapping 95% CIs."""
    verdict = {}
    for metric in ("dsi", "looming"):
        real = tuple(results[metric]["real"]["ci"])
        null = tuple(results[metric]["degree_preserving"]["ci"])
        verdict[f"{metric}_pass"] = (not intervals_overlap(real, null)) and real[0] > null[1]

    verdict["overall"] = bool(verdict["dsi_pass"] and verdict["looming_pass"])
    verdict["detail"] = (
        f"dsi_pass={verdict['dsi_pass']}, looming_pass={verdict['looming_pass']}. "
        + (
            "The real wiring outperformed its degree-preserving shuffle."
            if verdict["overall"]
            else "The real wiring did NOT measurably outperform its shuffle. "
            "Report this in the README; it is a finding, not a failure."
        )
    )
    return verdict
