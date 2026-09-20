"""Run the full validation battery: real connectome vs four null models, 5 seeds each."""
import time
from pathlib import Path
from flybrain.analysis.battery import run_battery
from flybrain.ingest.graph import load_graph
from flybrain.paths import ARTIFACTS

t0 = time.perf_counter()
g = load_graph(ARTIFACTS / "graph_t2.npz")
print(f"graph: {g.n_neurons:,} neurons, {g.n_edges:,} edges", flush=True)
r = run_battery(g, device="cuda", seeds=(0, 1, 2, 3, 4), out_dir=Path("out"))
print(f"\nwall clock: {time.perf_counter() - t0:.0f}s", flush=True)
print(r["verdict"]["detail"], flush=True)
for m in ("dsi", "looming"):
    for name, v in r[m].items():
        print(f'{m:8s} {name:20s} mean={v["mean"]:+.5f}  '
              f'CI=[{v["ci"][0]:+.5f}, {v["ci"][1]:+.5f}]', flush=True)
