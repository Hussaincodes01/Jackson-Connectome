# Fly Connectome Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the MaleCNS v1.0 *Drosophila* connectome as a live spiking simulation driven by real visual input, emitting named signals that drive a VTube Studio avatar.

**Architecture:** An offline ingest stage compiles 1 GB of Arrow edge data into a signed CSR graph artifact. A pure simulation core runs event-driven leaky integrate-and-fire dynamics over that graph on the GPU, importing nothing from I/O modules. Two drivers sit on the identical core — a batch driver for validation against shuffled-wiring null models, and a live driver that publishes signals over a WebSocket bus which a VTube Studio bridge consumes.

**Tech Stack:** Python 3.11.5, PyTorch 2.13.0+cu126 (CUDA), NumPy 2.4.6, PyArrow 25.0.1, SciPy 1.17.1, OpenCV 5.0.0.93, mss 10.2.0, websockets 15.0.1, PyYAML 6.0.3, pytest 9.1.1, pytest-asyncio 1.4.0, hypothesis 6.165.10, matplotlib 3.11.1.

**Spec:** `docs/superpowers/specs/2026-09-19-fly-connectome-engine-design.md`

## Global Constraints

- **Edge threshold default `T = 2`**, yielding exactly **57,670,765** edges. Configurable; `T=3` gives 23,014,406 and `T=5` gives 7,622,864.
- **Total edges in source file: 151,856,684.** Total annotated neurons: **166,700**.
- **NT sign map:** acetylcholine `+1`; GABA, glutamate, histamine `-1`; dopamine, octopamine, serotonin `0`; unclear/missing falls back to cell-type consensus, then `0`. Sign `0` means the neuron's outgoing edges are dropped at build time.
- **Sign is a presynaptic property** applied to all of a neuron's out-edges (Dale's law). Never per-edge.
- **The hot loop must be event-driven.** Dense propagation over all 57.7M edges caps at ~240 steps/s on an RTX 2050 and is a build failure, not a slow path.
- **`flybrain/core/` imports nothing from `encode`, `readout`, `drivers` or `bridge`.** Enforced by a test.
- **No smoke tests.** Every test asserts a specific known value or a closed-form result.
- **Homeostatic tuning runs identically on real and null graphs.** Any divergence invalidates the comparison.
- **Live driver clamps on runaway; batch driver halts on runaway.**
- **GPU target:** RTX 2050, 4 GB VRAM. Graph must stay under ~500 MB.
- Data files stay out of git (`.gitignore` excludes `*.feather`).

---

## File Structure

| File | Responsibility |
|---|---|
| `flybrain/paths.py` | Canonical data file locations |
| `flybrain/ingest/signs.py` | NT → sign resolution |
| `flybrain/ingest/build_graph.py` | Arrow stream → CSR artifact + manifest |
| `flybrain/ingest/graph.py` | `Graph` dataclass, loader, hash verification, cell-type index |
| `flybrain/core/lif.py` | `LIFParams`, `LIFNetwork` — event-driven LIF, device-agnostic |
| `flybrain/core/runaway.py` | Runaway detection policy |
| `flybrain/core/tuning.py` | Homeostatic gain calibration |
| `flybrain/encode/source.py` | `FrameSource` ABC + webcam/screen/synthetic |
| `flybrain/encode/lattice.py` | Hex lattice → sparse pixel sampling matrix |
| `flybrain/encode/encoder.py` | Luminance → adapted L1/L2 lamina currents |
| `flybrain/readout/signals.py` | Spike trains → named signals with provenance |
| `flybrain/nulls/rewire.py` | Four null-model rewirings |
| `flybrain/stimuli/patterns.py` | Gratings, looming discs, contrast steps |
| `flybrain/drivers/batch.py` | Stimulus → recorded activity → analysis |
| `flybrain/drivers/live.py` | Real-time loop, clock discipline, WebSocket publisher |
| `flybrain/analysis/metrics.py` | DSI, latency, CIs |
| `flybrain/bridge/vts.py` | Signal bus → VTube Studio parameter injection |
| `config/nt_signs.yaml` | Sign map (configuration, not code) |
| `config/avatar_mapping.yaml` | Signal → VTS parameter mapping |

---

# Phase 0 — Environment and data guard

### Task 1: Data integrity guard

**Files:**
- Create: `flybrain/__init__.py`, `flybrain/paths.py`
- Test: `tests/test_data_integrity.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `WEIGHTS`, `ANNOTATIONS`, `NEUROTRANSMITTERS`, `ARTIFACTS` as `pathlib.Path` constants; `count_edges() -> int`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_data_integrity.py
import pytest
import torch
import pyarrow.feather as feather
from flybrain.paths import WEIGHTS, ANNOTATIONS, NEUROTRANSMITTERS, count_edges

EXPECTED_EDGES = 151_856_684
EXPECTED_NEURONS = 166_700


def test_data_files_present():
    for p in (WEIGHTS, ANNOTATIONS, NEUROTRANSMITTERS):
        assert p.exists(), f"missing data file: {p}"


def test_cuda_available():
    assert torch.cuda.is_available(), "CUDA required; event-driven core targets GPU"


@pytest.mark.slow
def test_edge_count_exact():
    assert count_edges() == EXPECTED_EDGES


def test_annotated_neuron_count_exact():
    tbl = feather.read_table(ANNOTATIONS, columns=["superclass"])
    n = sum(1 for s in tbl.column("superclass").to_pylist() if s is not None)
    assert n == EXPECTED_NEURONS
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_data_integrity.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'flybrain'`

- [ ] **Step 3: Write minimal implementation**

```python
# flybrain/__init__.py
```

```python
# flybrain/paths.py
"""Canonical locations for the MaleCNS v1.0 data files."""
from pathlib import Path

import pyarrow as pa
import pyarrow.ipc as ipc

ROOT = Path(__file__).resolve().parent.parent
WEIGHTS = ROOT / "connectome-weights-male-cns-v1.0-minconf-0.5.feather"
ANNOTATIONS = ROOT / "body-annotations-male-cns-v1.0-minconf-0.5.feather"
NEUROTRANSMITTERS = ROOT / "body-neurotransmitters-male-cns-v1.0.feather"
ARTIFACTS = ROOT / "artifacts"


def count_edges() -> int:
    """Stream every record batch and count rows without materialising them."""
    total = 0
    with pa.memory_map(str(WEIGHTS), "rb") as src:
        reader = ipc.open_file(src)
        for i in range(reader.num_record_batches):
            total += reader.get_batch(i).num_rows
    return total
```

- [ ] **Step 4: Register the `slow` marker**

```ini
# pytest.ini
[pytest]
markers =
    slow: long-running tests (full-file scans, full-graph runs, battery)
testpaths = tests
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_data_integrity.py -v -m "not slow"`
Expected: PASS (3 tests)

Run: `pytest tests/test_data_integrity.py -v -m slow`
Expected: PASS in roughly 10 s

- [ ] **Step 6: MANUAL VERIFICATION**

Run: `python -c "from flybrain.paths import count_edges; print(f'{count_edges():,}')"`
Confirm with your own eyes it prints `151,856,684`.

- [ ] **Step 7: Commit**

```bash
git add flybrain/ tests/ pytest.ini
git commit -m "feat: data integrity guard asserting exact edge and neuron counts"
```

---

# Phase 1 — Ingest

### Task 2: Neurotransmitter sign resolution

**Files:**
- Create: `config/nt_signs.yaml`, `flybrain/ingest/__init__.py`, `flybrain/ingest/signs.py`
- Test: `tests/ingest/test_signs.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - `load_sign_map(path: Path) -> dict[str, int]`
  - `resolve_signs(consensus_nt: np.ndarray, celltype_nt: np.ndarray, sign_map: dict[str, int]) -> np.ndarray` returning `int8` array, same length as inputs. Both inputs are object/string arrays where missing is `None`.

- [ ] **Step 1: Write the failing test**

```python
# tests/ingest/test_signs.py
import numpy as np
import pytest
from flybrain.ingest.signs import load_sign_map, resolve_signs
from flybrain.paths import ROOT


@pytest.fixture
def sign_map():
    return load_sign_map(ROOT / "config" / "nt_signs.yaml")


def test_sign_map_exact_values(sign_map):
    assert sign_map["acetylcholine"] == 1
    assert sign_map["gaba"] == -1
    assert sign_map["glutamate"] == -1
    assert sign_map["histamine"] == -1
    assert sign_map["dopamine"] == 0
    assert sign_map["octopamine"] == 0
    assert sign_map["serotonin"] == 0


def test_each_transmitter_resolves_to_its_sign(sign_map):
    nts = np.array(
        ["acetylcholine", "gaba", "glutamate", "histamine",
         "dopamine", "octopamine", "serotonin"],
        dtype=object,
    )
    empty = np.array([None] * 7, dtype=object)
    assert resolve_signs(nts, empty, sign_map).tolist() == [1, -1, -1, -1, 0, 0, 0]


def test_unclear_falls_back_to_celltype_consensus(sign_map):
    consensus = np.array(["unclear", "unclear"], dtype=object)
    celltype = np.array(["gaba", "acetylcholine"], dtype=object)
    assert resolve_signs(consensus, celltype, sign_map).tolist() == [-1, 1]


def test_missing_everywhere_resolves_to_zero(sign_map):
    consensus = np.array([None, "unclear"], dtype=object)
    celltype = np.array([None, "unclear"], dtype=object)
    assert resolve_signs(consensus, celltype, sign_map).tolist() == [0, 0]


def test_unknown_transmitter_name_resolves_to_zero(sign_map):
    consensus = np.array(["kryptonite"], dtype=object)
    celltype = np.array([None], dtype=object)
    assert resolve_signs(consensus, celltype, sign_map).tolist() == [0]


def test_returns_int8(sign_map):
    out = resolve_signs(
        np.array(["acetylcholine"], dtype=object),
        np.array([None], dtype=object),
        sign_map,
    )
    assert out.dtype == np.int8
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/ingest/test_signs.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'flybrain.ingest'`

- [ ] **Step 3: Write the configuration**

```yaml
# config/nt_signs.yaml
# Presynaptic neurotransmitter -> synaptic sign (Dale's law).
# Sign 0 means the neuron contributes no output; its out-edges are dropped.
# glutamate is inhibitory in Drosophila via GluCl-alpha. Flip it here to run
# the sensitivity check the spec calls for -- no code change required.
acetylcholine: 1
gaba: -1
glutamate: -1
histamine: -1
dopamine: 0
octopamine: 0
serotonin: 0
```

- [ ] **Step 4: Write minimal implementation**

```python
# flybrain/ingest/__init__.py
```

```python
# flybrain/ingest/signs.py
"""Resolve presynaptic neurotransmitter identity to a synaptic sign."""
from pathlib import Path

import numpy as np
import yaml

UNKNOWN = {None, "unclear", ""}


def load_sign_map(path: Path) -> dict[str, int]:
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return {str(k): int(v) for k, v in raw.items()}


def resolve_signs(
    consensus_nt: np.ndarray,
    celltype_nt: np.ndarray,
    sign_map: dict[str, int],
) -> np.ndarray:
    """Per-neuron sign. Falls back to cell-type consensus, then 0.

    Sign 0 covers modulators (dopamine/octopamine/serotonin) and genuinely
    unknown neurons alike -- both contribute no output under a two-sign LIF
    model. The build stage drops their out-edges and tallies them separately.
    """
    n = len(consensus_nt)
    out = np.zeros(n, dtype=np.int8)
    for i in range(n):
        primary = consensus_nt[i]
        if primary in UNKNOWN:
            primary = celltype_nt[i]
        if primary in UNKNOWN:
            continue
        out[i] = sign_map.get(str(primary), 0)
    return out
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/ingest/test_signs.py -v`
Expected: PASS (6 tests)

- [ ] **Step 6: MANUAL VERIFICATION**

```bash
python -c "
import collections, numpy as np
from flybrain.ingest.signs import load_sign_map, resolve_signs
from flybrain.paths import ROOT, NEUROTRANSMITTERS
import pyarrow.feather as feather
m = load_sign_map(ROOT/'config'/'nt_signs.yaml')
tbl = feather.read_table(NEUROTRANSMITTERS, columns=['consensus_nt','celltype_predicted_nt'])
c = np.array(tbl.column('consensus_nt').to_pylist(), dtype=object)
ct = np.array(tbl.column('celltype_predicted_nt').to_pylist(), dtype=object)
s = resolve_signs(c, ct, m)
print('sign map:', m)
print(collections.Counter(s.tolist()))
"
```

Read the counts yourself. Positive should dominate (acetylcholine is the most
common transmitter), negatives should be a large minority, and zeros should be
a small fraction. This is the table that silently determines every downstream
result, so look at it once with your own eyes before trusting it.

- [ ] **Step 7: Commit**

```bash
git add config/nt_signs.yaml flybrain/ingest/ tests/ingest/
git commit -m "feat: neurotransmitter sign resolution with config-driven sign map"
```

---

### Task 3: Compile the graph artifact

**Files:**
- Create: `flybrain/ingest/build_graph.py`
- Test: `tests/ingest/test_build_graph.py`

**Interfaces:**
- Consumes: `flybrain.paths` constants; `load_sign_map`, `resolve_signs` from Task 2.
- Produces:
  - `load_annotations() -> dict` with keys `body_ids` (int64 array), `types` (object array), `superclasses` (object array), `hex1`, `hex2` (float arrays, NaN where absent).
  - `load_nt() -> tuple[np.ndarray, np.ndarray, np.ndarray]` of `(bodies, consensus_nt, celltype_nt)`.
  - `build_graph(threshold: int = 2, out_dir: Path = ARTIFACTS) -> Path` writing `graph_t{threshold}.npz` plus `graph_t{threshold}.manifest.json`, returning the `.npz` path.

The artifact stores `indptr` (int64, length `n+1`), `indices` (int32), `weights` (float32, already signed), `body_ids` (int64), `types`, `superclasses` (both as `<U32` arrays), `hex1`, `hex2` (float32).

- [ ] **Step 1: Write the failing test**

```python
# tests/ingest/test_build_graph.py
import json

import numpy as np
import pytest

from flybrain.ingest.build_graph import build_graph, load_annotations, load_nt
from flybrain.paths import ARTIFACTS

EXPECTED_EDGES_T2 = 57_670_765
EXPECTED_NEURONS = 166_700


def test_annotations_load_with_expected_neuron_count():
    ann = load_annotations()
    assert len(ann["body_ids"]) == EXPECTED_NEURONS
    assert len(ann["types"]) == EXPECTED_NEURONS


def test_annotations_contain_known_cell_types():
    ann = load_annotations()
    counts = {t: int((ann["types"] == t).sum()) for t in ("L1", "L2", "LPLC2", "LC4", "DNp01")}
    assert counts == {"L1": 1776, "L2": 1779, "LPLC2": 185, "LC4": 126, "DNp01": 2}


def test_nt_table_covers_all_annotated_neurons():
    ann = load_annotations()
    bodies, consensus, _ = load_nt()
    assert len(np.intersect1d(ann["body_ids"], bodies)) >= EXPECTED_NEURONS - 200


@pytest.mark.slow
def test_build_graph_edge_count_is_exact():
    path = build_graph(threshold=2, out_dir=ARTIFACTS)
    manifest = json.loads(path.with_suffix(".manifest.json").read_text())
    # Edges surviving the threshold, before sign-zero neurons are dropped.
    assert manifest["edges_after_threshold"] == EXPECTED_EDGES_T2


@pytest.mark.slow
def test_dale_law_every_neuron_has_one_output_sign():
    """Each neuron's out-edges must all share a sign. This is the single
    strongest structural check on the whole ingest pipeline."""
    data = np.load(ARTIFACTS / "graph_t2.npz", allow_pickle=False)
    indptr, weights = data["indptr"], data["weights"]
    rng = np.random.default_rng(0)
    rows = rng.choice(len(indptr) - 1, size=20_000, replace=False)
    for r in rows:
        seg = weights[indptr[r]:indptr[r + 1]]
        if seg.size:
            assert np.all(seg > 0) or np.all(seg < 0), f"neuron {r} has mixed signs"


@pytest.mark.slow
def test_known_looming_pathway_edge_exists():
    """LPLC2 -> DNp01 is the escape circuit the battery depends on."""
    data = np.load(ARTIFACTS / "graph_t2.npz", allow_pickle=True)
    indptr, indices, types = data["indptr"], data["indices"], data["types"]
    lplc2 = np.flatnonzero(types == "LPLC2")
    dnp01 = set(np.flatnonzero(types == "DNp01").tolist())
    hits = sum(
        1 for s in lplc2
        if dnp01 & set(indices[indptr[s]:indptr[s + 1]].tolist())
    )
    assert hits > 0, "no LPLC2 -> DNp01 connection survived the build"


@pytest.mark.slow
def test_manifest_hash_is_reproducible():
    first = json.loads((ARTIFACTS / "graph_t2.manifest.json").read_text())["content_hash"]
    path = build_graph(threshold=2, out_dir=ARTIFACTS)
    second = json.loads(path.with_suffix(".manifest.json").read_text())["content_hash"]
    assert first == second
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/ingest/test_build_graph.py -v -m "not slow"`
Expected: FAIL with `ModuleNotFoundError: No module named 'flybrain.ingest.build_graph'`

- [ ] **Step 3: Write minimal implementation**

```python
# flybrain/ingest/build_graph.py
"""Compile the Arrow edge list into a signed CSR artifact.

Streaming is mandatory: the source holds 151,856,684 edges and must never be
materialised whole. Filtering happens per record batch.
"""
from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.feather as feather
import pyarrow.ipc as ipc

from flybrain.ingest.signs import load_sign_map, resolve_signs
from flybrain.paths import ANNOTATIONS, ARTIFACTS, NEUROTRANSMITTERS, ROOT, WEIGHTS

SIGN_CONFIG = ROOT / "config" / "nt_signs.yaml"


def load_annotations() -> dict:
    """Annotated neurons only -- rows carrying a superclass."""
    tbl = feather.read_table(
        ANNOTATIONS,
        columns=["bodyId", "type", "superclass", "assignedOlHex1", "assignedOlHex2"],
    )
    superclass = np.array(tbl.column("superclass").to_pylist(), dtype=object)
    keep = superclass != None  # noqa: E711 -- object array, `is not None` will not vectorise
    body_ids = np.array(tbl.column("bodyId").to_pylist(), dtype=np.int64)[keep]
    types = np.array(
        [t if t is not None else "" for t in tbl.column("type").to_pylist()], dtype=object
    )[keep]
    hex1 = np.array(
        [h if h is not None else np.nan for h in tbl.column("assignedOlHex1").to_pylist()],
        dtype=np.float32,
    )[keep]
    hex2 = np.array(
        [h if h is not None else np.nan for h in tbl.column("assignedOlHex2").to_pylist()],
        dtype=np.float32,
    )[keep]
    order = np.argsort(body_ids)
    return {
        "body_ids": body_ids[order],
        "types": types[order],
        "superclasses": superclass[keep][order],
        "hex1": hex1[order],
        "hex2": hex2[order],
    }


def load_nt() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tbl = feather.read_table(
        NEUROTRANSMITTERS, columns=["body", "consensus_nt", "celltype_predicted_nt"]
    )
    bodies = np.array(tbl.column("body").to_pylist(), dtype=np.int64)
    consensus = np.array(tbl.column("consensus_nt").to_pylist(), dtype=object)
    celltype = np.array(tbl.column("celltype_predicted_nt").to_pylist(), dtype=object)
    return bodies, consensus, celltype


def _signs_for(body_ids: np.ndarray) -> np.ndarray:
    """Align the NT table onto the annotated-neuron ordering."""
    nt_bodies, consensus, celltype = load_nt()
    order = np.argsort(nt_bodies)
    nt_bodies, consensus, celltype = nt_bodies[order], consensus[order], celltype[order]
    pos = np.searchsorted(nt_bodies, body_ids)
    pos = np.clip(pos, 0, len(nt_bodies) - 1)
    found = nt_bodies[pos] == body_ids
    aligned_consensus = np.where(found, consensus[pos], None)
    aligned_celltype = np.where(found, celltype[pos], None)
    return resolve_signs(aligned_consensus, aligned_celltype, load_sign_map(SIGN_CONFIG))


def build_graph(threshold: int = 2, out_dir: Path = ARTIFACTS) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    ann = load_annotations()
    body_ids = ann["body_ids"]
    n = len(body_ids)
    signs = _signs_for(body_ids)

    pre_chunks: list[np.ndarray] = []
    post_chunks: list[np.ndarray] = []
    w_chunks: list[np.ndarray] = []
    edges_after_threshold = 0

    with pa.memory_map(str(WEIGHTS), "rb") as src:
        reader = ipc.open_file(src)
        for i in range(reader.num_record_batches):
            batch = reader.get_batch(i)
            w = batch.column("weight").to_numpy()
            mask = w >= threshold
            if not mask.any():
                continue
            pre = batch.column("body_pre").to_numpy()[mask]
            post = batch.column("body_post").to_numpy()[mask]
            w = w[mask]
            edges_after_threshold += len(w)

            # Map bodyIds to dense indices; -1 marks an endpoint that is not an
            # annotated neuron, and drops the edge.
            pi = np.searchsorted(body_ids, pre)
            qi = np.searchsorted(body_ids, post)
            pi_ok = (pi < n) & (body_ids[np.clip(pi, 0, n - 1)] == pre)
            qi_ok = (qi < n) & (body_ids[np.clip(qi, 0, n - 1)] == post)
            keep = pi_ok & qi_ok
            if not keep.any():
                continue
            pre_chunks.append(pi[keep].astype(np.int32))
            post_chunks.append(qi[keep].astype(np.int32))
            w_chunks.append(w[keep].astype(np.float32))

    pre = np.concatenate(pre_chunks)
    post = np.concatenate(post_chunks)
    weight = np.concatenate(w_chunks)
    edges_both_endpoints = len(weight)

    # Apply the presynaptic sign, then drop sign-zero sources entirely.
    edge_sign = signs[pre]
    nonzero = edge_sign != 0
    dropped_sign_zero = int((~nonzero).sum())
    pre, post = pre[nonzero], post[nonzero]
    weight = weight[nonzero] * edge_sign[nonzero].astype(np.float32)

    # CSR by stable sort on the source index.
    order = np.argsort(pre, kind="stable")
    pre, post, weight = pre[order], post[order], weight[order]
    indptr = np.searchsorted(pre, np.arange(n + 1)).astype(np.int64)

    hasher = hashlib.sha256()
    for arr in (indptr, post, weight):
        hasher.update(np.ascontiguousarray(arr).tobytes())
    content_hash = hasher.hexdigest()

    path = out_dir / f"graph_t{threshold}.npz"
    np.savez(
        path,
        indptr=indptr,
        indices=post,
        weights=weight,
        body_ids=body_ids,
        types=ann["types"].astype("<U32"),
        superclasses=np.array(
            [s if s is not None else "" for s in ann["superclasses"]], dtype="<U32"
        ),
        hex1=ann["hex1"],
        hex2=ann["hex2"],
        signs=signs,
    )

    manifest = {
        "built": date.today().isoformat(),
        "source": WEIGHTS.name,
        "threshold": threshold,
        "n_neurons": int(n),
        "edges_after_threshold": int(edges_after_threshold),
        "edges_both_endpoints_annotated": int(edges_both_endpoints),
        "edges_dropped_sign_zero": dropped_sign_zero,
        "edges_final": int(len(weight)),
        "excitatory_neurons": int((signs > 0).sum()),
        "inhibitory_neurons": int((signs < 0).sum()),
        "silent_neurons": int((signs == 0).sum()),
        "sign_map": load_sign_map(SIGN_CONFIG),
        "content_hash": content_hash,
        "limitations": [
            "Modulators (dopamine/octopamine/serotonin) are sign 0 and contribute no output.",
            "Capture frame rate bounds what T4/T5 can report; see encoder manifest entry.",
        ],
    }
    path.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2))
    return path


if __name__ == "__main__":
    import sys

    t = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    out = build_graph(threshold=t)
    print(out.with_suffix(".manifest.json").read_text())
```

- [ ] **Step 4: Run the fast tests**

Run: `pytest tests/ingest/test_build_graph.py -v -m "not slow"`
Expected: PASS (3 tests)

- [ ] **Step 5: Build the artifact and run the slow tests**

Run: `python -m flybrain.ingest.build_graph 2`
Expected: prints the manifest. Takes several minutes and roughly 2–3 GB of RAM at peak.

Run: `pytest tests/ingest/test_build_graph.py -v -m slow`
Expected: PASS (4 tests)

- [ ] **Step 6: MANUAL VERIFICATION**

Open `artifacts/graph_t2.manifest.json` and confirm by eye:
- `edges_after_threshold` is exactly `57670765`
- `n_neurons` is `166700`
- `excitatory_neurons` + `inhibitory_neurons` + `silent_neurons` equals `166700`
- `excitatory_neurons` is near `103720` and `inhibitory_neurons` near `59262`
- `content_hash` is present

Also check the artifact size: `ls -lh artifacts/graph_t2.npz` — expect roughly 500–700 MB on disk.

- [ ] **Step 7: Commit**

```bash
git add flybrain/ingest/build_graph.py tests/ingest/test_build_graph.py
git commit -m "feat: compile signed CSR graph artifact from Arrow edge stream"
```

---

### Task 4: Graph loader with hash verification and cell-type index

**Files:**
- Create: `flybrain/ingest/graph.py`
- Test: `tests/ingest/test_graph.py`

**Interfaces:**
- Consumes: the `.npz` and `.manifest.json` written by Task 3.
- Produces:
  - `@dataclass Graph` with fields `indptr: np.ndarray`, `indices: np.ndarray`, `weights: np.ndarray`, `body_ids: np.ndarray`, `types: np.ndarray`, `superclasses: np.ndarray`, `hex1: np.ndarray`, `hex2: np.ndarray`, `meta: dict`.
  - `Graph.n_neurons -> int`, `Graph.n_edges -> int`
  - `Graph.type_index(name: str) -> np.ndarray` — dense indices of neurons of that cell type
  - `Graph.superclass_index(name: str) -> np.ndarray`
  - `load_graph(path: Path, verify: bool = True) -> Graph` — raises `GraphHashMismatch` when the content hash disagrees with the manifest.

- [ ] **Step 1: Write the failing test**

```python
# tests/ingest/test_graph.py
import json

import numpy as np
import pytest

from flybrain.ingest.graph import Graph, GraphHashMismatch, load_graph
from flybrain.paths import ARTIFACTS

GRAPH = ARTIFACTS / "graph_t2.npz"

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def graph() -> Graph:
    return load_graph(GRAPH)


def test_neuron_count_exact(graph):
    assert graph.n_neurons == 166_700


def test_csr_is_structurally_valid(graph):
    assert graph.indptr[0] == 0
    assert graph.indptr[-1] == graph.n_edges
    assert np.all(np.diff(graph.indptr) >= 0), "indptr must be non-decreasing"
    assert graph.indices.max() < graph.n_neurons


def test_type_index_returns_expected_counts(graph):
    assert len(graph.type_index("DNp01")) == 2
    assert len(graph.type_index("LPLC2")) == 185
    assert len(graph.type_index("LC4")) == 126
    assert len(graph.type_index("L1")) == 1776


def test_superclass_index_returns_expected_counts(graph):
    assert len(graph.superclass_index("descending_neuron")) == 1314
    assert len(graph.superclass_index("vnc_motor")) == 708


def test_unknown_type_returns_empty(graph):
    assert len(graph.type_index("not_a_real_cell_type")) == 0


def test_hash_mismatch_refuses_to_load(tmp_path, graph):
    """A corrupted artifact must refuse to load -- every result has to be
    traceable to an exact build."""
    data = dict(np.load(GRAPH, allow_pickle=False))
    data["weights"] = data["weights"].copy()
    data["weights"][0] += 1.0
    bad = tmp_path / "graph_t2.npz"
    np.savez(bad, **data)
    manifest = json.loads(GRAPH.with_suffix(".manifest.json").read_text())
    bad.with_suffix(".manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(GraphHashMismatch):
        load_graph(bad)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/ingest/test_graph.py -v -m slow`
Expected: FAIL with `ModuleNotFoundError: No module named 'flybrain.ingest.graph'`

- [ ] **Step 3: Write minimal implementation**

```python
# flybrain/ingest/graph.py
"""Load a compiled graph artifact and refuse anything untraceable."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


class GraphHashMismatch(RuntimeError):
    """The artifact does not match the hash recorded in its manifest."""


@dataclass
class Graph:
    indptr: np.ndarray
    indices: np.ndarray
    weights: np.ndarray
    body_ids: np.ndarray
    types: np.ndarray
    superclasses: np.ndarray
    hex1: np.ndarray
    hex2: np.ndarray
    meta: dict = field(default_factory=dict)

    @property
    def n_neurons(self) -> int:
        return len(self.indptr) - 1

    @property
    def n_edges(self) -> int:
        return len(self.indices)

    def type_index(self, name: str) -> np.ndarray:
        return np.flatnonzero(self.types == name)

    def superclass_index(self, name: str) -> np.ndarray:
        return np.flatnonzero(self.superclasses == name)

    def types_matching(self, prefix: str) -> np.ndarray:
        """All neurons whose cell type starts with `prefix` (e.g. 'T4')."""
        return np.flatnonzero(np.char.startswith(self.types.astype("<U32"), prefix))


def content_hash(indptr: np.ndarray, indices: np.ndarray, weights: np.ndarray) -> str:
    hasher = hashlib.sha256()
    for arr in (indptr, indices, weights):
        hasher.update(np.ascontiguousarray(arr).tobytes())
    return hasher.hexdigest()


def load_graph(path: Path, verify: bool = True) -> Graph:
    data = np.load(path, allow_pickle=False)
    manifest_path = Path(path).with_suffix(".manifest.json")
    meta = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}

    if verify and "content_hash" in meta:
        actual = content_hash(data["indptr"], data["indices"], data["weights"])
        if actual != meta["content_hash"]:
            raise GraphHashMismatch(
                f"{path} hashes to {actual[:12]}... but its manifest records "
                f"{meta['content_hash'][:12]}... -- refusing to load"
            )

    return Graph(
        indptr=data["indptr"],
        indices=data["indices"],
        weights=data["weights"],
        body_ids=data["body_ids"],
        types=data["types"],
        superclasses=data["superclasses"],
        hex1=data["hex1"],
        hex2=data["hex2"],
        meta=meta,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/ingest/test_graph.py -v -m slow`
Expected: PASS (6 tests)

- [ ] **Step 5: MANUAL VERIFICATION**

```bash
python -c "
from flybrain.ingest.graph import load_graph
from flybrain.paths import ARTIFACTS
g = load_graph(ARTIFACTS/'graph_t2.npz')
print('neurons', g.n_neurons, 'edges', g.n_edges)
print('DNp01 at dense indices', g.type_index('DNp01'))
print('T4 subtypes', len(g.types_matching('T4')))
print('weights MB', g.weights.nbytes/1e6, 'indices MB', g.indices.nbytes/1e6)
"
```

Confirm the reported VRAM footprint (weights + indices) is under 500 MB — this is the number the spec's performance budget depends on.

- [ ] **Step 6: Commit**

```bash
git add flybrain/ingest/graph.py tests/ingest/test_graph.py
git commit -m "feat: graph loader with hash verification and cell-type index"
```

---

# Phase 2 — LIF core on CPU

### Task 5: Membrane dynamics verified against closed form

**Files:**
- Create: `flybrain/core/__init__.py`, `flybrain/core/lif.py`
- Test: `tests/core/test_membrane.py`

**Interfaces:**
- Consumes: `Graph` from Task 4 (only `indptr`, `indices`, `weights`, `n_neurons`).
- Produces:
  - `@dataclass LIFParams` with fields `v_rest: float = -52.0`, `v_th: float = -45.0`, `v_reset: float = -52.0`, `tau_m: float = 20.0`, `tau_syn: float = 5.0`, `t_ref: float = 2.2`, `dt: float = 1.0` (all ms / mV).
  - `class LIFNetwork(indptr, indices, weights, params, device="cpu")` with:
    - `.n_neurons -> int`
    - `.v -> torch.Tensor` (membrane potential, mV)
    - `.i_syn -> torch.Tensor`
    - `.reset() -> None`
    - `.step(input_current: torch.Tensor | None = None) -> torch.Tensor` returning a bool spike mask
    - `.from_graph(graph, params, device) -> LIFNetwork` classmethod

Integration is exponential-Euler, which makes the input-free solution **exact** rather than approximate — that is what the closed-form test below relies on.

- [ ] **Step 1: Write the failing test**

```python
# tests/core/test_membrane.py
import math

import numpy as np
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from flybrain.core.lif import LIFNetwork, LIFParams


def empty_net(n=3, **kw):
    """A network with no edges -- isolates membrane dynamics."""
    indptr = np.zeros(n + 1, dtype=np.int64)
    indices = np.zeros(0, dtype=np.int32)
    weights = np.zeros(0, dtype=np.float32)
    return LIFNetwork(indptr, indices, weights, LIFParams(**kw))


def test_starts_at_resting_potential():
    net = empty_net()
    assert torch.allclose(net.v, torch.full((3,), -52.0))


def test_membrane_decay_matches_closed_form():
    """V(t) = V_rest + (V0 - V_rest) * exp(-t / tau_m), to 1e-6."""
    params = LIFParams(tau_m=20.0, dt=1.0)
    net = empty_net(n=1, tau_m=20.0, dt=1.0)
    v0 = -30.0
    net.v[:] = v0
    for n_steps in range(1, 101):
        net.step()
        t = n_steps * params.dt
        expected = params.v_rest + (v0 - params.v_rest) * math.exp(-t / params.tau_m)
        assert abs(float(net.v[0]) - expected) < 1e-6, f"diverged at step {n_steps}"


def test_constant_input_drives_v_to_rest_plus_input():
    """With tau_syn -> 0 the steady state is V_rest + I. Checked at 500 ms."""
    net = empty_net(n=1, tau_syn=1e-6, tau_m=20.0, dt=1.0, v_th=1e9)
    drive = torch.tensor([5.0])
    for _ in range(500):
        net.step(drive)
    assert abs(float(net.v[0]) - (-52.0 + 5.0)) < 1e-3


def test_synaptic_current_decays_with_tau_syn():
    net = empty_net(n=1, tau_syn=5.0, dt=1.0, v_th=1e9)
    net.step(torch.tensor([10.0]))
    first = float(net.i_syn[0])
    net.step()
    second = float(net.i_syn[0])
    assert abs(second / first - math.exp(-1.0 / 5.0)) < 1e-6


def test_reset_restores_initial_state():
    net = empty_net(n=4)
    net.step(torch.full((4,), 3.0))
    net.reset()
    assert torch.allclose(net.v, torch.full((4,), -52.0))
    assert torch.allclose(net.i_syn, torch.zeros(4))


@settings(max_examples=50, deadline=None)
@given(
    tau_m=st.floats(min_value=1.0, max_value=100.0),
    v0=st.floats(min_value=-80.0, max_value=-46.0),
)
def test_decay_is_monotone_toward_rest_for_any_tau(tau_m, v0):
    net = empty_net(n=1, tau_m=tau_m, dt=1.0, v_th=1e9)
    net.v[:] = v0
    previous = v0
    for _ in range(50):
        net.step()
        current = float(net.v[0])
        if v0 > -52.0:
            assert current <= previous + 1e-9
        else:
            assert current >= previous - 1e-9
        previous = current
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/core/test_membrane.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'flybrain.core'`

- [ ] **Step 3: Write minimal implementation**

```python
# flybrain/core/__init__.py
```

```python
# flybrain/core/lif.py
"""Event-driven leaky integrate-and-fire over a signed CSR connectome.

This module is pure. It must not import from flybrain.encode, flybrain.readout,
flybrain.drivers or flybrain.bridge -- a test enforces that. It takes an input
current vector, advances one timestep, and exposes a spike mask. Nothing else.

Integration is exponential-Euler, so the input-free solution is exact:
    V(t) = V_rest + (V0 - V_rest) * exp(-t / tau_m)
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class LIFParams:
    """Millivolts and milliseconds throughout. Defaults follow the
    connectome-LIF literature (Shiu et al., Nature 2024)."""

    v_rest: float = -52.0
    v_th: float = -45.0
    v_reset: float = -52.0
    tau_m: float = 20.0
    tau_syn: float = 5.0
    t_ref: float = 2.2
    dt: float = 1.0


class LIFNetwork:
    def __init__(
        self,
        indptr: np.ndarray,
        indices: np.ndarray,
        weights: np.ndarray,
        params: LIFParams | None = None,
        device: str = "cpu",
    ) -> None:
        self.params = params or LIFParams()
        self.device = torch.device(device)
        self._n = len(indptr) - 1

        self.indptr = torch.as_tensor(np.asarray(indptr), dtype=torch.int64, device=self.device)
        self.indices = torch.as_tensor(np.asarray(indices), dtype=torch.int64, device=self.device)
        self.weights = torch.as_tensor(np.asarray(weights), dtype=torch.float32, device=self.device)

        p = self.params
        self.decay_m = math.exp(-p.dt / p.tau_m)
        self.decay_s = math.exp(-p.dt / p.tau_syn) if p.tau_syn > 0 else 0.0
        self.n_ref = int(round(p.t_ref / p.dt))

        self.gain = torch.ones(self._n, dtype=torch.float32, device=self.device)
        self.reset()

    @classmethod
    def from_graph(cls, graph, params: LIFParams | None = None, device: str = "cpu") -> "LIFNetwork":
        return cls(graph.indptr, graph.indices, graph.weights, params, device)

    @property
    def n_neurons(self) -> int:
        return self._n

    def reset(self) -> None:
        p = self.params
        self.v = torch.full((self._n,), p.v_rest, dtype=torch.float32, device=self.device)
        self.i_syn = torch.zeros(self._n, dtype=torch.float32, device=self.device)
        self.refrac = torch.zeros(self._n, dtype=torch.int16, device=self.device)
        self.last_spike_fraction = 0.0

    def _propagate(self, spike_idx: torch.Tensor) -> None:
        """Gather only the out-edges of neurons that actually fired.

        Dense propagation would touch all 57.7M edges every step and cap the
        simulation near 240 steps/s. This is the hot loop; keep it event-driven.
        """
        if spike_idx.numel() == 0:
            return
        starts = self.indptr[spike_idx]
        counts = self.indptr[spike_idx + 1] - starts
        total = int(counts.sum())
        if total == 0:
            return
        row_of = torch.repeat_interleave(
            torch.arange(spike_idx.numel(), device=self.device), counts
        )
        offsets = torch.cumsum(counts, 0) - counts
        flat = starts[row_of] + (
            torch.arange(total, device=self.device) - offsets[row_of]
        )
        targets = self.indices[flat]
        contrib = self.weights[flat] * self.gain[spike_idx][row_of]
        self.i_syn.index_add_(0, targets, contrib)

    def step(self, input_current: torch.Tensor | None = None) -> torch.Tensor:
        p = self.params

        self.i_syn.mul_(self.decay_s)
        if input_current is not None:
            self.i_syn.add_(input_current.to(self.device, torch.float32))

        # Exponential-Euler membrane update.
        self.v.copy_(
            p.v_rest
            + (self.v - p.v_rest) * self.decay_m
            + self.i_syn * (1.0 - self.decay_m)
        )

        refractory = self.refrac > 0
        if bool(refractory.any()):
            self.v = torch.where(
                refractory, torch.full_like(self.v, p.v_reset), self.v
            )
            self.refrac = torch.clamp(self.refrac - 1, min=0)

        spikes = self.v >= p.v_th
        spike_idx = spikes.nonzero(as_tuple=True)[0]
        if spike_idx.numel():
            self.v[spike_idx] = p.v_reset
            self.refrac[spike_idx] = self.n_ref
            self._propagate(spike_idx)

        self.last_spike_fraction = spike_idx.numel() / max(self._n, 1)
        return spikes
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/core/test_membrane.py -v`
Expected: PASS (6 tests, including the hypothesis property test)

- [ ] **Step 5: MANUAL VERIFICATION**

```bash
python -c "
import math, numpy as np, torch, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flybrain.core.lif import LIFNetwork, LIFParams
p = LIFParams(tau_m=20.0, dt=1.0, v_th=1e9)
net = LIFNetwork(np.zeros(2, dtype=np.int64), np.zeros(0, np.int32), np.zeros(0, np.float32), p)
net.v[:] = -30.0
sim = []
for _ in range(100):
    net.step(); sim.append(float(net.v[0]))
t = np.arange(1, 101)
analytic = p.v_rest + (-30.0 - p.v_rest) * np.exp(-t / p.tau_m)
plt.plot(t, sim, 'o', ms=3, label='simulated')
plt.plot(t, analytic, '-', label='closed form')
plt.xlabel('ms'); plt.ylabel('V (mV)'); plt.legend()
plt.savefig('figures/membrane_vs_analytic.png', dpi=120)
print('max abs error:', np.max(np.abs(np.array(sim) - analytic)))
"
```

Create `figures/` first if needed. Open `figures/membrane_vs_analytic.png` and confirm the points sit on the curve; the printed error should be below `1e-6`.

- [ ] **Step 6: Commit**

```bash
mkdir -p figures
git add flybrain/core/ tests/core/
git commit -m "feat: LIF membrane dynamics matching closed-form solution to 1e-6"
```

---

### Task 6: Spiking, refractoriness, signed propagation, and core isolation

**Files:**
- Modify: none (Task 5's implementation already covers these paths)
- Test: `tests/core/test_spiking.py`, `tests/core/test_isolation.py`

**Interfaces:**
- Consumes: `LIFNetwork`, `LIFParams` from Task 5.
- Produces: no new public interface. This task proves the remaining core behaviours and locks the module boundary.

- [ ] **Step 1: Write the failing tests**

```python
# tests/core/test_spiking.py
import numpy as np
import torch

from flybrain.core.lif import LIFNetwork, LIFParams


def two_neuron_net(weight: float, **kw):
    """Neuron 0 -> neuron 1 with the given signed weight."""
    indptr = np.array([0, 1, 1], dtype=np.int64)
    indices = np.array([1], dtype=np.int32)
    weights = np.array([weight], dtype=np.float32)
    return LIFNetwork(indptr, indices, weights, LIFParams(**kw))


def test_spike_fires_exactly_at_threshold():
    net = two_neuron_net(1.0)
    net.v[0] = -45.0  # exactly v_th
    spikes = net.step()
    assert bool(spikes[0]) is True


def test_no_spike_just_below_threshold():
    net = two_neuron_net(1.0, tau_m=1e9)  # effectively no decay
    net.v[0] = -45.001
    spikes = net.step()
    assert bool(spikes[0]) is False


def test_voltage_resets_after_spike():
    net = two_neuron_net(1.0)
    net.v[0] = -40.0
    net.step()
    assert abs(float(net.v[0]) - (-52.0)) < 1e-6


def test_refractory_lasts_exactly_ceil_tref_over_dt_steps():
    """t_ref=2.2, dt=1.0 -> n_ref = 2 steps clamped at reset."""
    net = two_neuron_net(1.0, t_ref=2.2, dt=1.0)
    assert net.n_ref == 2
    net.v[0] = -40.0
    net.step()                       # spikes, refrac = 2
    for expected in (1, 0):
        net.step(torch.tensor([100.0, 0.0]))
        assert abs(float(net.v[0]) - (-52.0)) < 1e-6, "clamped during refractory"
        assert int(net.refrac[0]) == expected
    # Free now: the same drive must move the membrane.
    net.step(torch.tensor([100.0, 0.0]))
    assert float(net.v[0]) > -52.0


def test_excitatory_edge_raises_postsynaptic_current():
    net = two_neuron_net(+3.0)
    net.v[0] = -40.0
    net.step()
    assert float(net.i_syn[1]) > 0
    assert abs(float(net.i_syn[1]) - 3.0) < 1e-6


def test_inhibitory_edge_lowers_postsynaptic_current():
    net = two_neuron_net(-3.0)
    net.v[0] = -40.0
    net.step()
    assert abs(float(net.i_syn[1]) - (-3.0)) < 1e-6


def test_postsynaptic_response_arrives_one_step_later():
    """Propagation happens after the membrane update, so a synapse costs one
    timestep. This is what the Phase 4 latency assertion measures."""
    net = two_neuron_net(+5.0)
    net.v[0] = -40.0
    net.step()
    assert abs(float(net.v[1]) - (-52.0)) < 1e-6, "target unchanged on the spike step"
    net.step()
    assert float(net.v[1]) > -52.0, "target responds on the following step"


def test_spike_fraction_is_reported():
    net = two_neuron_net(1.0)
    net.v[0] = -40.0
    net.step()
    assert abs(net.last_spike_fraction - 0.5) < 1e-9


def test_fan_out_sums_into_multiple_targets():
    indptr = np.array([0, 2, 2, 2], dtype=np.int64)
    indices = np.array([1, 2], dtype=np.int32)
    weights = np.array([2.0, -4.0], dtype=np.float32)
    net = LIFNetwork(indptr, indices, weights, LIFParams())
    net.v[0] = -40.0
    net.step()
    assert abs(float(net.i_syn[1]) - 2.0) < 1e-6
    assert abs(float(net.i_syn[2]) - (-4.0)) < 1e-6
```

```python
# tests/core/test_isolation.py
"""The core must stay pure. If this fails, the two drivers can drift apart
and validation results stop transferring to what appears on screen."""
import ast
import pathlib

FORBIDDEN = ("flybrain.encode", "flybrain.readout", "flybrain.drivers", "flybrain.bridge")
CORE = pathlib.Path("flybrain/core")


def test_core_does_not_import_io_modules():
    offenders = []
    for path in CORE.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                name = node.module
            elif isinstance(node, ast.Import):
                name = node.names[0].name
            else:
                continue
            if any(name.startswith(f) for f in FORBIDDEN):
                offenders.append(f"{path}: {name}")
    assert offenders == [], f"core imports I/O modules: {offenders}"
```

- [ ] **Step 2: Run tests to verify which fail**

Run: `pytest tests/core/test_spiking.py tests/core/test_isolation.py -v`
Expected: all PASS immediately — Task 5's implementation already satisfies them. If any fail, fix `lif.py` rather than the test. The isolation test is the one that must never be weakened.

- [ ] **Step 3: MANUAL VERIFICATION**

```bash
python -c "
import numpy as np, torch
from flybrain.core.lif import LIFNetwork, LIFParams
net = LIFNetwork(np.array([0,1,1],dtype=np.int64), np.array([1],np.int32),
                 np.array([8.0],np.float32), LIFParams())
net.v[0] = -40.0
for t in range(6):
    s = net.step()
    print(f't={t}ms  v0={float(net.v[0]):7.3f}  v1={float(net.v[1]):7.3f}  spikes={s.nonzero().flatten().tolist()}')
"
```

Read the trace yourself: neuron 0 spikes at `t=0`, neuron 1's voltage stays at rest on that step and rises from `t=1`. That one-step delay is the synaptic latency the engine test will measure.

- [ ] **Step 4: Commit**

```bash
git add tests/core/test_spiking.py tests/core/test_isolation.py
git commit -m "test: spiking, refractoriness, signed propagation, core purity"
```

---

# Phase 3 — GPU, full graph, event-driven

### Task 7: GPU/CPU agreement and the real-time performance floor

**Files:**
- Create: `tests/core/test_gpu_parity.py`, `tests/core/test_performance.py`, `flybrain/core/subgraph.py`
- Test: as above

**Interfaces:**
- Consumes: `LIFNetwork`, `LIFParams` (Task 5); `load_graph`, `Graph` (Task 4).
- Produces: `extract_subgraph(graph, keep: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]` returning `(indptr, indices, weights)` renumbered onto `keep`.

This is the task where connectome projects usually accept "it ran". It does not: it asserts the GPU agrees with the CPU, and that throughput clears real time.

Note on determinism: `index_add_` on CUDA accumulates via atomics, so floating-point summation order varies between runs. Exact bitwise equality is **not** available and asserting it would be wrong. The test asserts spike-train agreement above 99.5% and a bounded voltage difference, which is the correct standard for this operation.

- [ ] **Step 1: Write the failing tests**

```python
# tests/core/test_gpu_parity.py
import numpy as np
import pytest
import torch

from flybrain.core.lif import LIFNetwork, LIFParams
from flybrain.core.subgraph import extract_subgraph
from flybrain.ingest.graph import load_graph
from flybrain.paths import ARTIFACTS

pytestmark = pytest.mark.slow

N_KEEP = 20_000
N_STEPS = 500


@pytest.fixture(scope="module")
def subgraph():
    g = load_graph(ARTIFACTS / "graph_t2.npz")
    rng = np.random.default_rng(7)
    keep = np.sort(rng.choice(g.n_neurons, size=N_KEEP, replace=False))
    return extract_subgraph(g, keep)


def _run(subgraph, device, drive):
    indptr, indices, weights = subgraph
    net = LIFNetwork(indptr, indices, weights, LIFParams(), device=device)
    torch.manual_seed(0)
    raster = np.zeros((N_STEPS, N_KEEP), dtype=bool)
    for t in range(N_STEPS):
        spikes = net.step(drive.to(device))
        raster[t] = spikes.cpu().numpy()
    return raster, net.v.cpu().numpy()


def test_subgraph_is_structurally_valid(subgraph):
    indptr, indices, weights = subgraph
    assert len(indptr) == N_KEEP + 1
    assert indptr[0] == 0 and indptr[-1] == len(indices)
    assert len(indices) == len(weights)
    if len(indices):
        assert indices.max() < N_KEEP


def test_gpu_matches_cpu(subgraph):
    torch.manual_seed(0)
    drive = torch.rand(N_KEEP) * 2.0
    cpu_raster, cpu_v = _run(subgraph, "cpu", drive)
    gpu_raster, gpu_v = _run(subgraph, "cuda", drive)

    agreement = (cpu_raster == gpu_raster).mean()
    assert agreement > 0.995, f"spike rasters agree only {agreement:.4%}"

    total_cpu, total_gpu = cpu_raster.sum(), gpu_raster.sum()
    assert total_cpu > 0, "stimulus produced no spikes -- test is vacuous"
    assert abs(total_cpu - total_gpu) / total_cpu < 0.01

    assert np.max(np.abs(cpu_v - gpu_v)) < 1e-2
```

```python
# tests/core/test_performance.py
import time

import numpy as np
import pytest
import torch

from flybrain.core.lif import LIFNetwork, LIFParams
from flybrain.ingest.graph import load_graph
from flybrain.paths import ARTIFACTS

pytestmark = pytest.mark.slow

REALTIME_FLOOR_STEPS_PER_SEC = 1000


@pytest.fixture(scope="module")
def full_net():
    g = load_graph(ARTIFACTS / "graph_t2.npz")
    return LIFNetwork.from_graph(g, LIFParams(), device="cuda"), g


def test_graph_fits_in_vram_budget(full_net):
    _, g = full_net
    megabytes = (g.weights.nbytes + g.indices.nbytes + g.indptr.nbytes) / 1e6
    assert megabytes < 500, f"graph is {megabytes:.0f} MB, over the 4 GB card's budget"


def test_event_driven_loop_clears_realtime(full_net):
    """Dense propagation caps near 240 steps/s on this card. Anything at or
    below that means the event-driven path is not actually being taken."""
    net, g = full_net
    drive = torch.zeros(net.n_neurons, device="cuda")
    seed = torch.randint(0, net.n_neurons, (2000,), device="cuda")
    drive[seed] = 12.0

    for _ in range(50):          # warm up kernels and autotuning
        net.step(drive)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(500):
        net.step(drive)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    steps_per_sec = 500 / elapsed
    assert steps_per_sec > REALTIME_FLOOR_STEPS_PER_SEC, (
        f"{steps_per_sec:.0f} steps/s is below real time; "
        "the hot loop is not event-driven"
    )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/core/test_gpu_parity.py -v -m slow`
Expected: FAIL with `ModuleNotFoundError: No module named 'flybrain.core.subgraph'`

- [ ] **Step 3: Write minimal implementation**

```python
# flybrain/core/subgraph.py
"""Extract a renumbered CSR subgraph. Used for tests that must run on both
devices without a 4 GB allocation."""
from __future__ import annotations

import numpy as np


def extract_subgraph(graph, keep: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    keep = np.asarray(keep)
    n = len(keep)
    remap = np.full(graph.n_neurons, -1, dtype=np.int64)
    remap[keep] = np.arange(n)

    out_indptr = np.zeros(n + 1, dtype=np.int64)
    out_indices: list[np.ndarray] = []
    out_weights: list[np.ndarray] = []

    for new_i, old_i in enumerate(keep):
        lo, hi = graph.indptr[old_i], graph.indptr[old_i + 1]
        targets = remap[graph.indices[lo:hi]]
        mask = targets >= 0
        out_indices.append(targets[mask])
        out_weights.append(graph.weights[lo:hi][mask])
        out_indptr[new_i + 1] = out_indptr[new_i] + int(mask.sum())

    indices = (
        np.concatenate(out_indices).astype(np.int32)
        if out_indices
        else np.zeros(0, np.int32)
    )
    weights = (
        np.concatenate(out_weights).astype(np.float32)
        if out_weights
        else np.zeros(0, np.float32)
    )
    return out_indptr, indices, weights
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/core/test_gpu_parity.py tests/core/test_performance.py -v -m slow`
Expected: PASS (4 tests)

If `test_event_driven_loop_clears_realtime` fails, do **not** lower the threshold. It means `_propagate` is touching edges of non-spiking neurons; re-read the gather in `lif.py`.

- [ ] **Step 5: MANUAL VERIFICATION**

```bash
python -c "
import time, torch
from flybrain.core.lif import LIFNetwork, LIFParams
from flybrain.ingest.graph import load_graph
from flybrain.paths import ARTIFACTS
g = load_graph(ARTIFACTS/'graph_t2.npz')
net = LIFNetwork.from_graph(g, LIFParams(), device='cuda')
drive = torch.zeros(net.n_neurons, device='cuda')
drive[torch.randint(0, net.n_neurons, (2000,), device='cuda')] = 12.0
for _ in range(50): net.step(drive)
torch.cuda.synchronize(); t0=time.perf_counter()
for _ in range(1000): net.step(drive)
torch.cuda.synchronize()
sps = 1000/(time.perf_counter()-t0)
print(f'{sps:,.0f} steps/s  = {sps/1000:.1f}x real time')
print(f'spike fraction {net.last_spike_fraction:.4%}')
print(f'VRAM allocated {torch.cuda.memory_allocated()/1e6:.0f} MB')
"
```

Watch the three numbers yourself. Expect comfortably over 1,000 steps/s, a spike fraction in the low single-digit percent, and VRAM well under 4,000 MB. Run `nvidia-smi` in a second terminal while it executes if you want to see the card working.

- [ ] **Step 6: Commit**

```bash
git add flybrain/core/subgraph.py tests/core/test_gpu_parity.py tests/core/test_performance.py
git commit -m "feat: subgraph extraction; assert GPU/CPU agreement and real-time throughput"
```

---

### Task 8: Runaway protection and homeostatic tuning

**Files:**
- Create: `flybrain/core/runaway.py`, `flybrain/core/tuning.py`
- Test: `tests/core/test_runaway.py`, `tests/core/test_tuning.py`

**Interfaces:**
- Consumes: `LIFNetwork` (Task 5), `Graph` (Task 4).
- Produces:
  - `class RunawayDetector(ceiling: float = 0.15, window: int = 50, mode: str = "clamp")` with `.update(spike_fraction: float) -> str` returning `"ok"`, `"clamp"` or `"halt"`, and `.tripped: bool`.
  - `class RunawayHalt(RuntimeError)`
  - `homeostatic_tune(net, type_groups: dict[str, np.ndarray], drive_fn, target_hz: float = 5.0, rounds: int = 20, eta: float = 0.5, steps_per_round: int = 500) -> dict[str, float]` returning the per-type gains, and leaving them applied to `net.gain`.

`drive_fn(step_index) -> torch.Tensor` supplies the standard stimulus. The same function and the same rounds must be used for null graphs.

- [ ] **Step 1: Write the failing tests**

```python
# tests/core/test_runaway.py
import pytest

from flybrain.core.runaway import RunawayDetector, RunawayHalt


def test_quiet_activity_never_trips():
    d = RunawayDetector(ceiling=0.15, window=5)
    for _ in range(100):
        assert d.update(0.02) == "ok"
    assert d.tripped is False


def test_single_spike_above_ceiling_does_not_trip():
    """One busy millisecond is normal; sustained saturation is not."""
    d = RunawayDetector(ceiling=0.15, window=5)
    assert d.update(0.9) == "ok"
    assert d.tripped is False


def test_trips_after_exactly_window_consecutive_breaches():
    d = RunawayDetector(ceiling=0.15, window=5)
    for i in range(4):
        assert d.update(0.20) == "ok", f"tripped early at breach {i + 1}"
    assert d.update(0.20) == "clamp"
    assert d.tripped is True


def test_counter_resets_on_a_quiet_step():
    d = RunawayDetector(ceiling=0.15, window=3)
    d.update(0.2)
    d.update(0.2)
    d.update(0.01)
    assert d.update(0.2) == "ok"


def test_halt_mode_returns_halt():
    d = RunawayDetector(ceiling=0.15, window=2, mode="halt")
    d.update(0.5)
    assert d.update(0.5) == "halt"


def test_rejects_unknown_mode():
    with pytest.raises(ValueError):
        RunawayDetector(mode="explode")
```

```python
# tests/core/test_tuning.py
import numpy as np
import torch

from flybrain.core.lif import LIFNetwork, LIFParams
from flybrain.core.tuning import homeostatic_tune


def chain_net(n=60, weight=1.2):
    """A ring so activity can sustain itself and gain actually matters."""
    indptr = np.arange(n + 1, dtype=np.int64)
    indices = ((np.arange(n) + 1) % n).astype(np.int32)
    weights = np.full(n, weight, dtype=np.float32)
    return LIFNetwork(indptr, indices, weights, LIFParams())


def test_tuning_moves_rates_toward_target():
    net = chain_net()
    groups = {"ring": np.arange(net.n_neurons)}

    def drive(_):
        return torch.full((net.n_neurons,), 1.5)

    before = _measure(net, drive)
    homeostatic_tune(net, groups, drive, target_hz=5.0, rounds=20, steps_per_round=300)
    after = _measure(net, drive)
    assert abs(after - 5.0) < abs(before - 5.0), (
        f"tuning moved rate from {before:.2f} Hz to {after:.2f} Hz, away from 5 Hz"
    )


def test_tuning_returns_a_gain_per_group():
    net = chain_net()
    groups = {"a": np.arange(0, 30), "b": np.arange(30, 60)}
    gains = homeostatic_tune(
        net, groups, lambda _: torch.full((60,), 1.5), rounds=3, steps_per_round=100
    )
    assert set(gains) == {"a", "b"}
    assert all(g > 0 for g in gains.values())


def test_gains_are_applied_to_the_network():
    net = chain_net()
    groups = {"all": np.arange(net.n_neurons)}
    gains = homeostatic_tune(
        net, groups, lambda _: torch.full((60,), 1.5), rounds=3, steps_per_round=100
    )
    assert abs(float(net.gain[0]) - gains["all"]) < 1e-6


def test_tuning_is_deterministic_for_identical_inputs():
    """Real and null graphs must be tuned by an identical, reproducible
    procedure or the comparison means nothing."""
    results = []
    for _ in range(2):
        net = chain_net()
        results.append(
            homeostatic_tune(
                net,
                {"all": np.arange(60)},
                lambda _: torch.full((60,), 1.5),
                rounds=5,
                steps_per_round=200,
            )["all"]
        )
    assert abs(results[0] - results[1]) < 1e-9


def _measure(net, drive, steps=300):
    net.reset()
    count = 0
    for t in range(steps):
        count += int(net.step(drive(t)).sum())
    return count / net.n_neurons / (steps * net.params.dt / 1000.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/core/test_runaway.py tests/core/test_tuning.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'flybrain.core.runaway'`

- [ ] **Step 3: Write minimal implementation**

```python
# flybrain/core/runaway.py
"""Detect sustained saturation.

Live and batch fail differently on purpose: a frozen stream is dead, but an
experiment that silently clamps its gain and keeps recording produces
contaminated data.
"""
from __future__ import annotations


class RunawayHalt(RuntimeError):
    """Raised by the batch driver when activity saturates."""


class RunawayDetector:
    def __init__(self, ceiling: float = 0.15, window: int = 50, mode: str = "clamp") -> None:
        if mode not in ("clamp", "halt"):
            raise ValueError(f"mode must be 'clamp' or 'halt', got {mode!r}")
        self.ceiling = ceiling
        self.window = window
        self.mode = mode
        self._consecutive = 0
        self.tripped = False

    def update(self, spike_fraction: float) -> str:
        if spike_fraction > self.ceiling:
            self._consecutive += 1
        else:
            self._consecutive = 0
        if self._consecutive >= self.window:
            self.tripped = True
            self._consecutive = 0
            return self.mode
        return "ok"
```

```python
# flybrain/core/tuning.py
"""Homeostatic gain calibration.

The connectome fixes who connects to whom; it does not fix synaptic strength
in millivolts. Gains are set by driving firing rates into a plausible band.

The procedure is deterministic and must be applied identically to null models.
Tuning the real network into working while leaving controls to fail would
invalidate every comparison in the battery.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import torch


def homeostatic_tune(
    net,
    type_groups: dict[str, np.ndarray],
    drive_fn: Callable[[int], torch.Tensor],
    target_hz: float = 5.0,
    rounds: int = 20,
    eta: float = 0.5,
    steps_per_round: int = 500,
    floor: float = 1e-3,
    ceiling: float = 1e3,
) -> dict[str, float]:
    gains = {name: 1.0 for name in type_groups}
    seconds = steps_per_round * net.params.dt / 1000.0

    for _ in range(rounds):
        net.reset()
        counts = torch.zeros(net.n_neurons, dtype=torch.float32, device=net.device)
        for t in range(steps_per_round):
            counts += net.step(drive_fn(t)).to(torch.float32)

        for name, idx in type_groups.items():
            members = torch.as_tensor(np.asarray(idx), dtype=torch.long, device=net.device)
            observed = float(counts[members].mean()) / seconds
            if observed <= 0.0:
                gains[name] = min(gains[name] * (1.0 + eta), ceiling)
            else:
                gains[name] = float(
                    np.clip(gains[name] * (target_hz / observed) ** eta, floor, ceiling)
                )
            net.gain[members] = gains[name]

    net.reset()
    return gains
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/core/test_runaway.py tests/core/test_tuning.py -v`
Expected: PASS (10 tests)

- [ ] **Step 5: MANUAL VERIFICATION**

```bash
python -c "
import numpy as np, torch
from flybrain.core.lif import LIFNetwork, LIFParams
from flybrain.core.tuning import homeostatic_tune
n=60
net = LIFNetwork(np.arange(n+1,dtype=np.int64), ((np.arange(n)+1)%n).astype(np.int32),
                 np.full(n,1.2,np.float32), LIFParams())
drive = lambda _: torch.full((n,), 1.5)
def rate(net):
    net.reset(); c=0
    for t in range(300): c += int(net.step(drive(t)).sum())
    return c/n/0.3
print('before tuning: %.2f Hz' % rate(net))
g = homeostatic_tune(net, {'ring': np.arange(n)}, drive, target_hz=5.0)
print('gain chosen: %.4f' % g['ring'])
print('after tuning:  %.2f Hz' % rate(net))
"
```

Confirm the rate moves toward 5 Hz. It will not land exactly — the ring is a toy and gain is a blunt instrument. What matters is direction and that the gain is finite and positive.

- [ ] **Step 6: Commit**

```bash
git add flybrain/core/runaway.py flybrain/core/tuning.py tests/core/test_runaway.py tests/core/test_tuning.py
git commit -m "feat: runaway detection and deterministic homeostatic gain tuning"
```

---

# Phase 4 — Engine test (encoder-free)

### Task 9: Known-answer latency and sign propagation on the real connectome

**Files:**
- Create: `flybrain/analysis/__init__.py`, `flybrain/analysis/latency.py`
- Test: `tests/analysis/test_engine.py`

**Interfaces:**
- Consumes: `load_graph`, `Graph` (Task 4); `LIFNetwork`, `LIFParams` (Task 5).
- Produces:
  - `inject_and_record(net, source_idx: np.ndarray, watch_idx: np.ndarray, amplitude: float = 30.0, steps: int = 60, inject_steps: int = 5) -> dict` returning `{"first_spike_ms": np.ndarray, "raster": np.ndarray, "n_responding": int}` where `first_spike_ms[i]` is the first spike time of `watch_idx[i]` or `np.nan`.
  - `monosynaptic_targets(graph, source_idx: np.ndarray, min_weight: float = 5.0) -> np.ndarray`

This is the diagnostic that separates engine faults from encoder faults. If it passes and the visual battery later fails, the encoder is at fault.

- [ ] **Step 1: Write the failing test**

```python
# tests/analysis/test_engine.py
import numpy as np
import pytest

from flybrain.analysis.latency import inject_and_record, monosynaptic_targets
from flybrain.core.lif import LIFNetwork, LIFParams
from flybrain.ingest.graph import load_graph
from flybrain.paths import ARTIFACTS

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def graph():
    return load_graph(ARTIFACTS / "graph_t2.npz")


@pytest.fixture(scope="module")
def net(graph):
    return LIFNetwork.from_graph(graph, LIFParams(), device="cuda")


def test_lplc2_has_monosynaptic_targets(graph):
    src = graph.type_index("LPLC2")
    assert len(src) == 185
    targets = monosynaptic_targets(graph, src, min_weight=5.0)
    assert len(targets) > 0, "LPLC2 has no strong outputs -- ingest is wrong"


def test_driven_population_actually_spikes(net, graph):
    src = graph.type_index("LPLC2")
    out = inject_and_record(net, src, src, amplitude=30.0, steps=40)
    assert out["n_responding"] > 0, "injected population did not spike"


def test_monosynaptic_latency_is_one_to_three_ms(net, graph):
    """One synapse costs one timestep of transmission plus membrane charging.
    Anything outside 1-3 ms means the propagation path is wrong."""
    src = graph.type_index("LPLC2")
    targets = monosynaptic_targets(graph, src, min_weight=10.0)
    assert len(targets) >= 5, "not enough strong postsynaptic partners to test"

    out = inject_and_record(net, src, targets, amplitude=30.0, steps=60)
    latencies = out["first_spike_ms"]
    responded = latencies[~np.isnan(latencies)]
    assert len(responded) > 0, "no monosynaptic partner responded"

    median = float(np.median(responded))
    assert 1.0 <= median <= 3.0, f"median monosynaptic latency {median} ms is out of range"


def test_partners_that_respond_are_connectome_partners(net, graph):
    """Responses must land on neurons the wiring actually predicts, not
    scattered across the network."""
    src = graph.type_index("LPLC2")
    targets = monosynaptic_targets(graph, src, min_weight=10.0)
    unconnected = np.setdiff1d(
        graph.type_index("L1"), monosynaptic_targets(graph, src, min_weight=0.0)
    )[: len(targets)]

    connected_out = inject_and_record(net, src, targets, amplitude=30.0, steps=60)
    control_out = inject_and_record(net, src, unconnected, amplitude=30.0, steps=60)

    connected_rate = np.mean(~np.isnan(connected_out["first_spike_ms"]))
    control_rate = np.mean(~np.isnan(control_out["first_spike_ms"]))
    assert connected_rate > control_rate, (
        f"connected partners responded at {connected_rate:.2%} but unconnected "
        f"neurons at {control_rate:.2%} -- propagation is not following the wiring"
    )


def test_inhibitory_source_suppresses_rather_than_drives(net, graph):
    """A GABAergic population must lower its targets' voltages."""
    gaba_sources = np.flatnonzero(graph.weights[graph.indptr[:-1]] < 0)
    assert len(gaba_sources) > 0
    src = gaba_sources[:200]
    targets = monosynaptic_targets(graph, src, min_weight=0.0)[:500]
    assert len(targets) > 0

    net.reset()
    baseline = float(net.v[targets].mean())
    out = inject_and_record(net, src, targets, amplitude=30.0, steps=30)
    assert out["n_responding"] >= 0
    assert float(net.v[targets].mean()) <= baseline + 1e-3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/analysis/test_engine.py -v -m slow`
Expected: FAIL with `ModuleNotFoundError: No module named 'flybrain.analysis'`

- [ ] **Step 3: Write minimal implementation**

```python
# flybrain/analysis/__init__.py
```

```python
# flybrain/analysis/latency.py
"""Encoder-free engine diagnostics.

Inject current directly into an identified population and measure how its
connectome-predicted partners respond. No visual pathway is involved, so a
failure here is unambiguously a simulation fault.
"""
from __future__ import annotations

import numpy as np
import torch


def monosynaptic_targets(graph, source_idx: np.ndarray, min_weight: float = 5.0) -> np.ndarray:
    """Distinct postsynaptic partners reached by an edge of at least the given
    absolute weight."""
    collected: list[np.ndarray] = []
    for s in np.asarray(source_idx):
        lo, hi = graph.indptr[s], graph.indptr[s + 1]
        seg_w = np.abs(graph.weights[lo:hi])
        collected.append(graph.indices[lo:hi][seg_w >= min_weight])
    if not collected:
        return np.zeros(0, dtype=np.int64)
    return np.unique(np.concatenate(collected)).astype(np.int64)


def inject_and_record(
    net,
    source_idx: np.ndarray,
    watch_idx: np.ndarray,
    amplitude: float = 30.0,
    steps: int = 60,
    inject_steps: int = 5,
) -> dict:
    net.reset()
    source = torch.as_tensor(np.asarray(source_idx), dtype=torch.long, device=net.device)
    watch = torch.as_tensor(np.asarray(watch_idx), dtype=torch.long, device=net.device)

    drive = torch.zeros(net.n_neurons, dtype=torch.float32, device=net.device)
    drive[source] = amplitude
    silent = torch.zeros_like(drive)

    first = np.full(len(watch_idx), np.nan, dtype=np.float64)
    raster = np.zeros((steps, len(watch_idx)), dtype=bool)

    for t in range(steps):
        spikes = net.step(drive if t < inject_steps else silent)
        watched = spikes[watch].cpu().numpy()
        raster[t] = watched
        newly = watched & np.isnan(first)
        first[newly] = t * net.params.dt

    return {
        "first_spike_ms": first,
        "raster": raster,
        "n_responding": int(np.sum(~np.isnan(first))),
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/analysis/test_engine.py -v -m slow`
Expected: PASS (5 tests)

If `test_monosynaptic_latency_is_one_to_three_ms` fails high, raise `amplitude`; if it fails low, the propagation is firing targets on the same step and the ordering in `LIFNetwork.step` is wrong.

- [ ] **Step 5: MANUAL VERIFICATION — latency histogram**

```bash
python -c "
import numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from flybrain.analysis.latency import inject_and_record, monosynaptic_targets
from flybrain.core.lif import LIFNetwork, LIFParams
from flybrain.ingest.graph import load_graph
from flybrain.paths import ARTIFACTS
g = load_graph(ARTIFACTS/'graph_t2.npz')
net = LIFNetwork.from_graph(g, LIFParams(), device='cuda')
src = g.type_index('LPLC2')
tgt = monosynaptic_targets(g, src, min_weight=10.0)
out = inject_and_record(net, src, tgt, amplitude=30.0, steps=60)
lat = out['first_spike_ms']; lat = lat[~np.isnan(lat)]
print(f'{len(tgt)} strong partners, {len(lat)} responded, median {np.median(lat):.2f} ms')
plt.hist(lat, bins=np.arange(0, 20, 1))
plt.xlabel('first-spike latency (ms)'); plt.ylabel('partners')
plt.title('LPLC2 -> monosynaptic partners')
plt.savefig('figures/engine_latency.png', dpi=120)
"
```

Open `figures/engine_latency.png`. You should see a clear peak in the 1–3 ms bins. A flat or right-shifted distribution means the engine is not propagating along the wiring, and **you should not proceed to Phase 5 until it looks right** — every visual result downstream depends on this being correct.

- [ ] **Step 6: Commit**

```bash
git add flybrain/analysis/ tests/analysis/
git commit -m "feat: encoder-free engine test asserting monosynaptic latency and sign"
```

---

# Phase 5 — Sensory encoding

### Task 10: Hex lattice and the pixel sampling matrix

**Files:**
- Modify: `flybrain/ingest/build_graph.py` — add `soma_side` to the saved artifact
- Modify: `flybrain/ingest/graph.py` — add `soma_side` to `Graph`
- Create: `flybrain/encode/__init__.py`, `flybrain/encode/lattice.py`
- Test: `tests/encode/test_lattice.py`

**Interfaces:**
- Consumes: `Graph` (Task 4) — specifically `types`, `hex1`, `hex2`, `soma_side`.
- Produces:
  - `@dataclass HexLattice` with `neuron_idx: np.ndarray`, `hex1: np.ndarray`, `hex2: np.ndarray`, `xy: np.ndarray` (shape `(n_columns, 2)`, normalised to `[0, 1]`), `side: str`.
  - `build_lattice(graph, cell_type: str, side: str) -> HexLattice`
  - `sampling_matrix(lattice: HexLattice, height: int, width: int, sigma_px: float = 3.0, fov_fraction: float = 0.9) -> scipy.sparse.csr_matrix` of shape `(n_columns, height * width)`, each row summing to 1.

Axial hex coordinates convert to cartesian by `x = h1 + h2 / 2`, `y = h2 * sqrt(3) / 2`.

**Why this task amends Task 3:** the eyes are distinguished by `somaSide` (L1 is R 892 / L 884), which the first artifact did not store. Not every hex column has a neuron on both sides — 875 L1 columns have two, 17 have one, and 9 L1 neurons carry no coordinates at all. The lattice builder must tolerate that rather than assume a perfect tiling.

- [ ] **Step 1: Write the failing test**

```python
# tests/encode/test_lattice.py
import numpy as np
import pytest

from flybrain.encode.lattice import build_lattice, sampling_matrix
from flybrain.ingest.graph import load_graph
from flybrain.paths import ARTIFACTS

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def graph():
    return load_graph(ARTIFACTS / "graph_t2.npz")


def test_artifact_carries_soma_side(graph):
    assert hasattr(graph, "soma_side")
    l1 = graph.type_index("L1")
    sides = graph.soma_side[l1]
    assert int((sides == "R").sum()) == 892
    assert int((sides == "L").sum()) == 884


def test_right_l1_lattice_has_892_columns(graph):
    lat = build_lattice(graph, "L1", "R")
    assert len(lat.neuron_idx) == 892


def test_lattice_excludes_neurons_without_coordinates(graph):
    lat = build_lattice(graph, "L1", "R")
    assert not np.isnan(lat.hex1).any()
    assert not np.isnan(lat.hex2).any()


def test_lattice_xy_is_normalised(graph):
    lat = build_lattice(graph, "L1", "R")
    assert lat.xy.shape == (len(lat.neuron_idx), 2)
    assert lat.xy.min() >= 0.0 and lat.xy.max() <= 1.0
    # A hex lattice must actually span both axes, not collapse to a line.
    assert lat.xy[:, 0].ptp() > 0.9 and lat.xy[:, 1].ptp() > 0.9


def test_neighbouring_columns_are_near_in_xy(graph):
    """Retinotopy must be preserved: hex neighbours map to nearby points."""
    lat = build_lattice(graph, "L1", "R")
    h1, h2, xy = lat.hex1, lat.hex2, lat.xy
    pairs = 0
    for i in range(0, len(h1), 37):
        neighbour = np.flatnonzero((h1 == h1[i] + 1) & (h2 == h2[i]))
        if neighbour.size:
            j = neighbour[0]
            far = np.flatnonzero((h1 == h1[i]) & (h2 == h2[i] + 10))
            if far.size:
                assert np.linalg.norm(xy[i] - xy[j]) < np.linalg.norm(xy[i] - xy[far[0]])
                pairs += 1
    assert pairs > 5, "not enough neighbour pairs tested"


def test_sampling_matrix_rows_are_normalised(graph):
    lat = build_lattice(graph, "L1", "R")
    m = sampling_matrix(lat, 120, 160)
    assert m.shape == (892, 120 * 160)
    sums = np.asarray(m.sum(axis=1)).ravel()
    assert np.allclose(sums, 1.0, atol=1e-5)


def test_uniform_image_yields_uniform_column_response(graph):
    lat = build_lattice(graph, "L1", "R")
    m = sampling_matrix(lat, 120, 160)
    frame = np.full(120 * 160, 0.5, dtype=np.float32)
    out = m @ frame
    assert np.allclose(out, 0.5, atol=1e-4)


def test_left_half_bright_image_activates_one_side_of_the_lattice(graph):
    """A half-bright frame must produce a spatial gradient across columns."""
    lat = build_lattice(graph, "L1", "R")
    m = sampling_matrix(lat, 120, 160)
    img = np.zeros((120, 160), dtype=np.float32)
    img[:, :80] = 1.0
    out = m @ img.ravel()
    left_cols = out[lat.xy[:, 0] < 0.35]
    right_cols = out[lat.xy[:, 0] > 0.65]
    assert left_cols.mean() > right_cols.mean() + 0.3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/encode/test_lattice.py -v -m slow`
Expected: FAIL with `ModuleNotFoundError: No module named 'flybrain.encode'`

- [ ] **Step 3: Amend the artifact to carry `somaSide`**

In `flybrain/ingest/build_graph.py`, add `"somaSide"` to the `columns=[...]` list in `load_annotations`, then add to the returned dict:

```python
    soma_side = np.array(
        [s if s is not None else "" for s in tbl.column("somaSide").to_pylist()], dtype=object
    )[keep]
```

and include `"soma_side": soma_side[order]` in the returned dict. Then in `build_graph`, add to the `np.savez(...)` call:

```python
        soma_side=np.array(
            [s if s is not None else "" for s in ann["soma_side"]], dtype="<U4"
        ),
```

In `flybrain/ingest/graph.py`, add `soma_side: np.ndarray` to the `Graph` dataclass (after `superclasses`) and `soma_side=data["soma_side"],` to the constructor call in `load_graph`.

- [ ] **Step 4: Rebuild the artifact**

Run: `python -m flybrain.ingest.build_graph 2`
Expected: a new manifest with a **different** `content_hash` is fine — the hash covers `indptr`, `indices` and `weights`, which are unchanged, so it should in fact match the previous value. If it differs, the edge pipeline changed unintentionally; stop and find out why.

- [ ] **Step 5: Write the lattice implementation**

```python
# flybrain/encode/__init__.py
```

```python
# flybrain/encode/lattice.py
"""Map the optic lobe's hexagonal column lattice onto image pixels.

892 columns tile each lobe. Coverage is not perfect -- some columns have a
neuron on only one side -- so the builder takes what exists rather than
assuming a complete tiling.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp


@dataclass
class HexLattice:
    neuron_idx: np.ndarray
    hex1: np.ndarray
    hex2: np.ndarray
    xy: np.ndarray
    side: str


def build_lattice(graph, cell_type: str, side: str) -> HexLattice:
    candidates = graph.type_index(cell_type)
    on_side = candidates[graph.soma_side[candidates] == side]
    has_coords = on_side[
        ~np.isnan(graph.hex1[on_side]) & ~np.isnan(graph.hex2[on_side])
    ]

    h1 = graph.hex1[has_coords].astype(np.float64)
    h2 = graph.hex2[has_coords].astype(np.float64)

    x = h1 + h2 / 2.0
    y = h2 * (math.sqrt(3.0) / 2.0)

    def norm(a: np.ndarray) -> np.ndarray:
        span = a.max() - a.min()
        return (a - a.min()) / span if span > 0 else np.zeros_like(a)

    return HexLattice(
        neuron_idx=has_coords,
        hex1=h1,
        hex2=h2,
        xy=np.stack([norm(x), norm(y)], axis=1),
        side=side,
    )


def sampling_matrix(
    lattice: HexLattice,
    height: int,
    width: int,
    sigma_px: float = 3.0,
    fov_fraction: float = 0.9,
) -> sp.csr_matrix:
    """Each column draws a Gaussian-weighted patch. Rows sum to 1, so a
    uniform frame yields a uniform column response.

    `fov_fraction` insets the lattice from the frame edge -- the fly's real
    field of view is far wider than a webcam's, so the lattice occupies the
    central region and the periphery is simply dark.
    """
    margin = (1.0 - fov_fraction) / 2.0
    cx = (margin + lattice.xy[:, 0] * fov_fraction) * (width - 1)
    cy = (margin + lattice.xy[:, 1] * fov_fraction) * (height - 1)

    radius = int(math.ceil(3 * sigma_px))
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    vals: list[np.ndarray] = []

    for c in range(len(cx)):
        x0, x1 = max(0, int(cx[c]) - radius), min(width, int(cx[c]) + radius + 1)
        y0, y1 = max(0, int(cy[c]) - radius), min(height, int(cy[c]) + radius + 1)
        xs = np.arange(x0, x1)
        ys = np.arange(y0, y1)
        gx = np.exp(-((xs - cx[c]) ** 2) / (2 * sigma_px**2))
        gy = np.exp(-((ys - cy[c]) ** 2) / (2 * sigma_px**2))
        patch = np.outer(gy, gx)
        total = patch.sum()
        if total <= 0:
            continue
        patch /= total
        flat = (ys[:, None] * width + xs[None, :]).ravel()
        rows.append(np.full(flat.size, c, dtype=np.int32))
        cols.append(flat.astype(np.int32))
        vals.append(patch.ravel().astype(np.float32))

    return sp.csr_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(len(cx), height * width),
    )
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `pytest tests/encode/test_lattice.py -v -m slow`
Expected: PASS (8 tests)

- [ ] **Step 7: MANUAL VERIFICATION — see the fly's-eye view**

```bash
python -c "
import numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from flybrain.encode.lattice import build_lattice, sampling_matrix
from flybrain.ingest.graph import load_graph
from flybrain.paths import ARTIFACTS
g = load_graph(ARTIFACTS/'graph_t2.npz')
lat = build_lattice(g, 'L1', 'R')
m = sampling_matrix(lat, 120, 160)
img = np.zeros((120,160), np.float32)
img[30:90, 40:70] = 1.0            # a bright bar
out = m @ img.ravel()
fig, ax = plt.subplots(1, 2, figsize=(10,4))
ax[0].imshow(img, cmap='gray'); ax[0].set_title('frame')
ax[1].scatter(lat.xy[:,0], -lat.xy[:,1], c=out, s=18, cmap='viridis')
ax[1].set_title(f'{len(lat.neuron_idx)} L1 columns, right eye'); ax[1].set_aspect('equal')
plt.tight_layout(); plt.savefig('figures/hex_view.png', dpi=120)
print('columns:', len(lat.neuron_idx), 'response range:', out.min(), out.max())
"
```

Open `figures/hex_view.png`. **The bar must be recognisable in the hex rendering** — same position, same shape, in the characteristic hexagonal packing. If the hex panel looks like noise or a straight line, the axial-to-cartesian conversion is wrong and every visual result afterwards is meaningless.

- [ ] **Step 8: Commit**

```bash
git add flybrain/encode/ flybrain/ingest/ tests/encode/
git commit -m "feat: hex lattice and Gaussian pixel sampling matrix for both eyes"
```

---

### Task 11: Frame sources and the adapting lamina encoder

**Files:**
- Create: `flybrain/encode/source.py`, `flybrain/encode/encoder.py`
- Test: `tests/encode/test_source.py`, `tests/encode/test_encoder.py`

**Interfaces:**
- Consumes: `HexLattice`, `build_lattice`, `sampling_matrix` (Task 10); `Graph` (Task 4).
- Produces:
  - `@dataclass Frame` with `image: np.ndarray` (float32, `[0, 1]`, shape `(H, W)`), `timestamp: float`, `stale: bool`.
  - `class FrameSource(ABC)` with `.read() -> Frame`, `.close() -> None`, `.shape -> tuple[int, int]`.
  - `WebcamSource(index=0, height=120, width=160, fps=30)`, `ScreenSource(monitor=1, height=120, width=160)`, `SyntheticSource(generator: Callable[[int], np.ndarray], height, width)`.
  - `class LaminaEncoder(graph, height, width, tau_adapt_ms=500.0, drive_mv=8.0)` with `.encode(frame: Frame) -> np.ndarray` returning a length-`n_neurons` float32 current vector, and `.reset()`.

`LaminaEncoder` drives **L1 with positive contrast and L2 with negative**, reproducing the ON/OFF split. Input enters at the lamina rather than the photoreceptors because photoreceptors carry no column assignment — and the photoreceptor→LMC synapse is inhibitory, a sign inversion already present in the data.

- [ ] **Step 1: Write the failing tests**

```python
# tests/encode/test_source.py
import numpy as np

from flybrain.encode.source import Frame, SyntheticSource


def test_synthetic_source_returns_requested_shape():
    src = SyntheticSource(lambda i: np.full((30, 40), 0.25, np.float32), 30, 40)
    frame = src.read()
    assert isinstance(frame, Frame)
    assert frame.image.shape == (30, 40)
    assert frame.image.dtype == np.float32


def test_synthetic_source_advances_its_index():
    src = SyntheticSource(lambda i: np.full((4, 4), i / 10.0, np.float32), 4, 4)
    assert abs(float(src.read().image[0, 0]) - 0.0) < 1e-6
    assert abs(float(src.read().image[0, 0]) - 0.1) < 1e-6


def test_frames_are_clipped_to_unit_range():
    src = SyntheticSource(lambda i: np.full((4, 4), 5.0, np.float32), 4, 4)
    assert float(src.read().image.max()) <= 1.0


def test_timestamps_increase():
    src = SyntheticSource(lambda i: np.zeros((4, 4), np.float32), 4, 4)
    first = src.read().timestamp
    assert src.read().timestamp >= first
```

```python
# tests/encode/test_encoder.py
import numpy as np
import pytest

from flybrain.encode.encoder import LaminaEncoder
from flybrain.encode.source import Frame
from flybrain.ingest.graph import load_graph
from flybrain.paths import ARTIFACTS

pytestmark = pytest.mark.slow

H, W = 120, 160


@pytest.fixture(scope="module")
def graph():
    return load_graph(ARTIFACTS / "graph_t2.npz")


@pytest.fixture(scope="module")
def encoder(graph):
    return LaminaEncoder(graph, H, W)


def frame(value):
    return Frame(image=np.full((H, W), value, np.float32), timestamp=0.0, stale=False)


def test_output_length_matches_neuron_count(encoder, graph):
    out = encoder.encode(frame(0.5))
    assert out.shape == (graph.n_neurons,)
    assert out.dtype == np.float32


def test_only_lamina_neurons_receive_current(encoder, graph):
    encoder.reset()
    out = encoder.encode(frame(1.0))
    driven = np.flatnonzero(out != 0)
    lamina = np.concatenate([graph.type_index("L1"), graph.type_index("L2")])
    assert len(np.setdiff1d(driven, lamina)) == 0, "current leaked outside L1/L2"


def test_l1_and_l2_receive_opposite_polarity(encoder, graph):
    encoder.reset()
    out = encoder.encode(frame(1.0))     # bright against a dark-adapted mean
    l1 = out[graph.type_index("L1")]
    l2 = out[graph.type_index("L2")]
    l1_active = l1[l1 != 0]
    l2_active = l2[l2 != 0]
    assert l1_active.mean() > 0, "L1 should depolarise to positive contrast"
    assert l2_active.mean() < 0, "L2 should carry the opposite sign"


def test_adaptation_decays_response_to_a_constant_scene(encoder):
    """Real photoreceptors adapt. Without this a bright scene saturates the
    model permanently."""
    encoder.reset()
    first = np.abs(encoder.encode(frame(1.0))).sum()
    for _ in range(200):
        encoder.encode(frame(1.0))
    settled = np.abs(encoder.encode(frame(1.0))).sum()
    assert settled < first * 0.5, (
        f"response only fell from {first:.1f} to {settled:.1f} -- not adapting"
    )


def test_contrast_step_produces_a_transient(encoder):
    encoder.reset()
    for _ in range(200):
        encoder.encode(frame(0.2))
    adapted = np.abs(encoder.encode(frame(0.2))).sum()
    stepped = np.abs(encoder.encode(frame(0.8))).sum()
    assert stepped > adapted * 2, "a contrast step must produce a transient"


def test_stale_frame_still_encodes(encoder):
    encoder.reset()
    stale = Frame(image=np.full((H, W), 0.5, np.float32), timestamp=0.0, stale=True)
    out = encoder.encode(stale)
    assert np.isfinite(out).all()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/encode/test_source.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'flybrain.encode.source'`

- [ ] **Step 3: Write the frame sources**

```python
# flybrain/encode/source.py
"""Frame acquisition. Webcam, screen and synthetic all emit the same thing,
so the input source is a runtime switch rather than an architectural choice."""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class Frame:
    image: np.ndarray
    timestamp: float
    stale: bool = False


class FrameSource(ABC):
    def __init__(self, height: int, width: int) -> None:
        self.height = height
        self.width = width
        self._last: Frame | None = None

    @property
    def shape(self) -> tuple[int, int]:
        return self.height, self.width

    @abstractmethod
    def read(self) -> Frame: ...

    def close(self) -> None:
        return None

    def _finish(self, image: np.ndarray) -> Frame:
        image = np.clip(np.asarray(image, dtype=np.float32), 0.0, 1.0)
        self._last = Frame(image=image, timestamp=time.perf_counter(), stale=False)
        return self._last

    def _hold_last(self) -> Frame:
        """Source failed. Hold the previous frame and flag it; the simulation
        keeps its own clock regardless."""
        if self._last is None:
            return Frame(np.zeros((self.height, self.width), np.float32), time.perf_counter(), True)
        return Frame(self._last.image, time.perf_counter(), True)


class SyntheticSource(FrameSource):
    def __init__(self, generator: Callable[[int], np.ndarray], height: int, width: int) -> None:
        super().__init__(height, width)
        self.generator = generator
        self.index = 0

    def read(self) -> Frame:
        image = self.generator(self.index)
        self.index += 1
        return self._finish(image)


class WebcamSource(FrameSource):
    def __init__(self, index: int = 0, height: int = 120, width: int = 160, fps: int = 30) -> None:
        super().__init__(height, width)
        import cv2

        self._cv2 = cv2
        self.capture = cv2.VideoCapture(index)
        self.capture.set(cv2.CAP_PROP_FPS, fps)

    def read(self) -> Frame:
        ok, raw = self.capture.read()
        if not ok:
            return self._hold_last()
        gray = self._cv2.cvtColor(raw, self._cv2.COLOR_BGR2GRAY)
        small = self._cv2.resize(gray, (self.width, self.height))
        return self._finish(small.astype(np.float32) / 255.0)

    def close(self) -> None:
        self.capture.release()


class ScreenSource(FrameSource):
    def __init__(self, monitor: int = 1, height: int = 120, width: int = 160) -> None:
        super().__init__(height, width)
        import cv2
        import mss

        self._cv2 = cv2
        self._sct = mss.mss()
        self._monitor = self._sct.monitors[monitor]

    def read(self) -> Frame:
        try:
            shot = np.asarray(self._sct.grab(self._monitor))
        except Exception:
            return self._hold_last()
        gray = self._cv2.cvtColor(shot[:, :, :3], self._cv2.COLOR_BGR2GRAY)
        small = self._cv2.resize(gray, (self.width, self.height))
        return self._finish(small.astype(np.float32) / 255.0)

    def close(self) -> None:
        self._sct.close()
```

- [ ] **Step 4: Write the encoder**

```python
# flybrain/encode/encoder.py
"""Luminance -> adapted contrast -> lamina current.

Input enters at L1/L2 rather than the photoreceptors: photoreceptors carry no
hex-column assignment in this dataset, and the photoreceptor->LMC synapse is
histaminergic and inhibitory, so the sign inversion is already accounted for.

L1 carries the ON pathway and L2 the OFF pathway, so they receive opposite
contrast polarity.
"""
from __future__ import annotations

import math

import numpy as np

from flybrain.encode.lattice import build_lattice, sampling_matrix
from flybrain.encode.source import Frame


class LaminaEncoder:
    def __init__(
        self,
        graph,
        height: int,
        width: int,
        tau_adapt_ms: float = 500.0,
        drive_mv: float = 8.0,
        dt_ms: float = 33.0,
        sigma_px: float = 3.0,
    ) -> None:
        self.n_neurons = graph.n_neurons
        self.drive_mv = drive_mv
        self.adapt_decay = math.exp(-dt_ms / tau_adapt_ms)

        self.channels = []
        for cell_type, polarity in (("L1", +1.0), ("L2", -1.0)):
            for side in ("R", "L"):
                lattice = build_lattice(graph, cell_type, side)
                if len(lattice.neuron_idx) == 0:
                    continue
                self.channels.append(
                    {
                        "idx": lattice.neuron_idx,
                        "matrix": sampling_matrix(lattice, height, width, sigma_px=sigma_px),
                        "polarity": polarity,
                    }
                )
        self.reset()

    def reset(self) -> None:
        self._mean = [None] * len(self.channels)

    def encode(self, frame: Frame) -> np.ndarray:
        flat = frame.image.ravel().astype(np.float32)
        out = np.zeros(self.n_neurons, dtype=np.float32)

        for c, channel in enumerate(self.channels):
            luminance = channel["matrix"] @ flat

            if self._mean[c] is None:
                self._mean[c] = luminance.copy()
            running = self._mean[c]

            # Divisive adaptation against the running local mean.
            contrast = (luminance - running) / (running + 0.05)
            self._mean[c] = running * self.adapt_decay + luminance * (1.0 - self.adapt_decay)

            out[channel["idx"]] = (
                channel["polarity"] * self.drive_mv * np.clip(contrast, -4.0, 4.0)
            ).astype(np.float32)

        return out
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/encode/ -v`
Expected: PASS (4 source tests + 6 encoder tests)

Note `test_l1_and_l2_receive_opposite_polarity` relies on the first frame being encoded against a freshly initialised mean. If it fails, check that `reset()` clears `_mean` to `None` rather than to zeros.

- [ ] **Step 6: MANUAL VERIFICATION — your own face through a fly's eye**

```bash
python -c "
import numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from flybrain.encode.source import WebcamSource
from flybrain.encode.encoder import LaminaEncoder
from flybrain.encode.lattice import build_lattice
from flybrain.ingest.graph import load_graph
from flybrain.paths import ARTIFACTS
g = load_graph(ARTIFACTS/'graph_t2.npz')
src = WebcamSource(0, 120, 160)
enc = LaminaEncoder(g, 120, 160)
for _ in range(30): f = src.read(); enc.encode(f)   # let adaptation settle
f = src.read(); out = enc.encode(f)
lat = build_lattice(g, 'L1', 'R')
fig, ax = plt.subplots(1, 2, figsize=(10,4))
ax[0].imshow(f.image, cmap='gray'); ax[0].set_title('webcam')
ax[1].scatter(lat.xy[:,0], -lat.xy[:,1], c=out[lat.neuron_idx], s=18, cmap='RdBu_r')
ax[1].set_title('L1 drive (right eye)'); ax[1].set_aspect('equal')
plt.tight_layout(); plt.savefig('figures/webcam_hex.png', dpi=120)
src.close()
print('nonzero driven neurons:', int((out!=0).sum()))
"
```

Open `figures/webcam_hex.png`. Move in front of the camera and re-run. **You should be able to recognise the rough shape of what the camera saw in the hex panel.** This is the single most informative manual check in the project — if the fly can't see, nothing downstream is real.

- [ ] **Step 7: Commit**

```bash
git add flybrain/encode/source.py flybrain/encode/encoder.py tests/encode/
git commit -m "feat: webcam/screen/synthetic sources and adapting ON/OFF lamina encoder"
```

---

# Phase 6 — Null models and the validation battery

### Task 12: Four shuffled-wiring null models

**Files:**
- Create: `flybrain/nulls/__init__.py`, `flybrain/nulls/rewire.py`
- Test: `tests/nulls/test_rewire.py`

**Interfaces:**
- Consumes: `Graph` (Task 4).
- Produces, each returning a new `(indptr, indices, weights)` triple and never mutating its input:
  - `weight_permutation(indptr, indices, weights, seed: int)`
  - `degree_preserving_rewire(indptr, indices, weights, seed: int)`
  - `sign_permutation(indptr, indices, weights, seed: int)`
  - `type_preserving_rewire(indptr, indices, weights, types: np.ndarray, seed: int)`
  - `NULL_MODELS: dict[str, Callable]` mapping name → function.

**Every null must preserve Dale's law**, because the real graph does. A null that violates it would differ from the real network in two ways at once and the comparison would not isolate wiring. `weight_permutation` therefore shuffles magnitudes and re-applies each edge's original sign; `sign_permutation` permutes the *per-neuron* sign and re-applies it across all of that neuron's out-edges.

- [ ] **Step 1: Write the failing test**

```python
# tests/nulls/test_rewire.py
import numpy as np
import pytest

from flybrain.nulls.rewire import (
    NULL_MODELS,
    degree_preserving_rewire,
    sign_permutation,
    type_preserving_rewire,
    weight_permutation,
)


@pytest.fixture
def toy():
    """6 neurons. Rows 0-2 excitatory, rows 3-5 inhibitory (Dale's law holds)."""
    indptr = np.array([0, 2, 4, 5, 7, 8, 9], dtype=np.int64)
    indices = np.array([1, 2, 3, 4, 5, 0, 1, 2, 3], dtype=np.int32)
    weights = np.array([2.0, 3.0, 1.0, 5.0, 4.0, -2.0, -6.0, -1.0, -3.0], dtype=np.float32)
    types = np.array(["A", "A", "A", "B", "B", "B"], dtype="<U32")
    return indptr, indices, weights, types


def out_degrees(indptr):
    return np.diff(indptr)


def in_degrees(indices, n):
    return np.bincount(indices, minlength=n)


def dale_holds(indptr, weights):
    for r in range(len(indptr) - 1):
        seg = weights[indptr[r]:indptr[r + 1]]
        if seg.size and not (np.all(seg > 0) or np.all(seg < 0)):
            return False
    return True


def test_all_four_models_are_registered():
    assert set(NULL_MODELS) == {
        "weight_permutation",
        "degree_preserving",
        "sign_permutation",
        "type_preserving",
    }


def test_weight_permutation_preserves_topology_and_magnitude_multiset(toy):
    indptr, indices, weights, _ = toy
    p, i, w = weight_permutation(indptr, indices, weights, seed=0)
    assert np.array_equal(p, indptr)
    assert np.array_equal(i, indices)
    assert np.allclose(np.sort(np.abs(w)), np.sort(np.abs(weights)))
    assert dale_holds(p, w)


def test_weight_permutation_actually_changes_something(toy):
    indptr, indices, weights, _ = toy
    _, _, w = weight_permutation(indptr, indices, weights, seed=0)
    assert not np.allclose(w, weights)


def test_degree_preserving_keeps_both_degree_sequences(toy):
    indptr, indices, weights, _ = toy
    n = len(indptr) - 1
    p, i, w = degree_preserving_rewire(indptr, indices, weights, seed=0)
    assert np.array_equal(out_degrees(p), out_degrees(indptr))
    assert np.array_equal(
        np.sort(in_degrees(i, n)), np.sort(in_degrees(indices, n))
    )
    assert dale_holds(p, w)


def test_degree_preserving_rewires_targets(toy):
    indptr, indices, weights, _ = toy
    _, i, _ = degree_preserving_rewire(indptr, indices, weights, seed=3)
    assert not np.array_equal(i, indices)


def test_sign_permutation_preserves_topology_and_sign_counts(toy):
    indptr, indices, weights, _ = toy
    p, i, w = sign_permutation(indptr, indices, weights, seed=0)
    assert np.array_equal(i, indices)
    assert dale_holds(p, w)
    # The number of excitatory and inhibitory neurons is unchanged.
    def neuron_signs(ptr, ws):
        return sorted(
            np.sign(ws[ptr[r]]) for r in range(len(ptr) - 1) if ptr[r + 1] > ptr[r]
        )
    assert neuron_signs(p, w) == neuron_signs(indptr, weights)


def test_type_preserving_keeps_type_to_type_counts(toy):
    indptr, indices, weights, types = toy
    p, i, w = type_preserving_rewire(indptr, indices, weights, types, seed=0)

    def block_counts(ptr, idx):
        counts: dict[tuple[str, str], int] = {}
        for r in range(len(ptr) - 1):
            for t in idx[ptr[r]:ptr[r + 1]]:
                key = (types[r], types[t])
                counts[key] = counts.get(key, 0) + 1
        return counts

    assert block_counts(p, i) == block_counts(indptr, indices)
    assert dale_holds(p, w)


def test_all_models_are_reproducible_for_a_seed(toy):
    indptr, indices, weights, types = toy
    for name, fn in NULL_MODELS.items():
        args = (indptr, indices, weights, types, 11) if name == "type_preserving" else (
            indptr, indices, weights, 11
        )
        a = fn(*args)
        b = fn(*args)
        for x, y in zip(a, b):
            assert np.array_equal(x, y), f"{name} is not reproducible"


def test_models_do_not_mutate_their_input(toy):
    indptr, indices, weights, types = toy
    before = weights.copy()
    weight_permutation(indptr, indices, weights, seed=0)
    degree_preserving_rewire(indptr, indices, weights, seed=0)
    sign_permutation(indptr, indices, weights, seed=0)
    type_preserving_rewire(indptr, indices, weights, types, seed=0)
    assert np.array_equal(weights, before)


def test_edge_count_is_conserved_by_every_model(toy):
    indptr, indices, weights, types = toy
    for name, fn in NULL_MODELS.items():
        args = (indptr, indices, weights, types, 5) if name == "type_preserving" else (
            indptr, indices, weights, 5
        )
        _, i, w = fn(*args)
        assert len(i) == len(indices), name
        assert len(w) == len(weights), name
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/nulls/test_rewire.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'flybrain.nulls'`

- [ ] **Step 3: Write minimal implementation**

```python
# flybrain/nulls/__init__.py
```

```python
# flybrain/nulls/rewire.py
"""Shuffled-wiring controls.

Every model preserves Dale's law, because the real graph does. A null that
broke it would differ from the real network in two ways at once, and the
comparison would no longer isolate the contribution of wiring.
"""
from __future__ import annotations

import numpy as np


def _row_signs(indptr: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """One sign per neuron, taken from its first out-edge (Dale's law)."""
    n = len(indptr) - 1
    signs = np.zeros(n, dtype=np.float32)
    has_edges = np.diff(indptr) > 0
    rows = np.flatnonzero(has_edges)
    signs[rows] = np.sign(weights[indptr[rows]])
    return signs


def _expand_row_signs(indptr: np.ndarray, signs: np.ndarray, n_edges: int) -> np.ndarray:
    counts = np.diff(indptr)
    return np.repeat(signs, counts)[:n_edges]


def weight_permutation(indptr, indices, weights, seed: int):
    """Shuffle magnitudes; keep topology and each edge's sign."""
    rng = np.random.default_rng(seed)
    magnitudes = np.abs(weights).copy()
    rng.shuffle(magnitudes)
    signed = magnitudes * np.sign(weights)
    return indptr.copy(), indices.copy(), signed.astype(np.float32)


def degree_preserving_rewire(indptr, indices, weights, seed: int):
    """Permute the entire target list.

    Out-degree is preserved because row lengths are untouched; in-degree is
    preserved exactly because the multiset of targets is unchanged.
    """
    rng = np.random.default_rng(seed)
    shuffled = indices.copy()
    rng.shuffle(shuffled)
    return indptr.copy(), shuffled, weights.copy()


def sign_permutation(indptr, indices, weights, seed: int):
    """Permute which neurons are excitatory and which inhibitory."""
    rng = np.random.default_rng(seed)
    signs = _row_signs(indptr, weights)
    active = np.flatnonzero(signs != 0)
    permuted = signs.copy()
    permuted[active] = rng.permutation(signs[active])
    edge_old = _expand_row_signs(indptr, signs, len(weights))
    edge_new = _expand_row_signs(indptr, permuted, len(weights))
    out = np.abs(weights) * edge_new
    # Edges from sign-0 rows cannot occur in a compiled graph, but stay safe.
    out = np.where(edge_new == 0, weights * 0.0, out)
    del edge_old
    return indptr.copy(), indices.copy(), out.astype(np.float32)


def type_preserving_rewire(indptr, indices, weights, types: np.ndarray, seed: int):
    """Randomise individual partners while holding cell-type-to-cell-type
    connection counts fixed. The strictest null: it asks whether the specific
    wiring matters beyond the type-level summary."""
    rng = np.random.default_rng(seed)
    counts = np.diff(indptr)
    source_of_edge = np.repeat(np.arange(len(counts)), counts)
    pre_types = types[source_of_edge]
    post_types = types[indices]

    new_indices = indices.copy()
    key = np.char.add(np.char.add(pre_types.astype("<U32"), "->"), post_types.astype("<U32"))
    for block in np.unique(key):
        members = np.flatnonzero(key == block)
        if members.size > 1:
            new_indices[members] = rng.permutation(indices[members])
    return indptr.copy(), new_indices, weights.copy()


NULL_MODELS = {
    "weight_permutation": weight_permutation,
    "degree_preserving": degree_preserving_rewire,
    "sign_permutation": sign_permutation,
    "type_preserving": type_preserving_rewire,
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/nulls/test_rewire.py -v`
Expected: PASS (10 tests)

- [ ] **Step 5: MANUAL VERIFICATION**

```bash
python -c "
import numpy as np
from flybrain.ingest.graph import load_graph
from flybrain.nulls.rewire import NULL_MODELS
from flybrain.paths import ARTIFACTS
g = load_graph(ARTIFACTS/'graph_t2.npz')
print(f'real: {g.n_edges:,} edges')
for name, fn in NULL_MODELS.items():
    args = (g.indptr, g.indices, g.weights, g.types, 0) if name=='type_preserving' else (g.indptr, g.indices, g.weights, 0)
    p,i,w = fn(*args)
    same = float(np.mean(i == g.indices))
    print(f'{name:22s} edges={len(i):,}  targets unchanged={same:.1%}  '
          f'weight sum={w.sum():+.0f} (real {g.weights.sum():+.0f})')
"
```

Confirm each null keeps the same edge count, and that `degree_preserving` and `type_preserving` leave only a small fraction of targets unchanged. `type_preserving` will keep more than `degree_preserving` does — that's expected and is exactly why it's the harder null.

- [ ] **Step 6: Commit**

```bash
git add flybrain/nulls/ tests/nulls/
git commit -m "feat: four Dale-preserving null models for shuffled-wiring controls"
```

---

### Task 13: Visual stimuli and the batch driver

**Files:**
- Create: `flybrain/stimuli/__init__.py`, `flybrain/stimuli/patterns.py`, `flybrain/drivers/__init__.py`, `flybrain/drivers/batch.py`
- Test: `tests/stimuli/test_patterns.py`, `tests/drivers/test_batch.py`

**Interfaces:**
- Consumes: `SyntheticSource`, `Frame` (Task 11); `LaminaEncoder` (Task 11); `LIFNetwork` (Task 5); `RunawayDetector`, `RunawayHalt` (Task 8).
- Produces:
  - `drifting_grating(height, width, direction_deg: float, spatial_period_px: float = 24.0, speed_px_per_frame: float = 2.0) -> Callable[[int], np.ndarray]`
  - `looming_disc(height, width, r_over_v: float = 40.0, n_frames: int = 90) -> Callable[[int], np.ndarray]`
  - `contrast_step(height, width, low: float = 0.2, high: float = 0.8, switch_frame: int = 30) -> Callable[[int], np.ndarray]`
  - `run_batch(net, encoder, source, watch: dict[str, np.ndarray], n_frames: int, steps_per_frame: int = 33, detector=None) -> dict[str, np.ndarray]` returning per-group spike counts per frame, shape `(n_frames,)` per key, plus `"spike_fraction"`.

The batch driver **halts** on runaway (`RunawayHalt`), per the spec. The live driver clamps instead.

- [ ] **Step 1: Write the failing tests**

```python
# tests/stimuli/test_patterns.py
import numpy as np

from flybrain.stimuli.patterns import contrast_step, drifting_grating, looming_disc

H, W = 60, 80


def test_grating_has_expected_shape_and_range():
    g = drifting_grating(H, W, 0.0)
    frame = g(0)
    assert frame.shape == (H, W)
    assert 0.0 <= frame.min() and frame.max() <= 1.0


def test_horizontal_grating_moves_between_frames():
    g = drifting_grating(H, W, 0.0, speed_px_per_frame=4.0)
    assert not np.allclose(g(0), g(1))


def test_opposite_directions_produce_different_sequences():
    a = drifting_grating(H, W, 0.0)
    b = drifting_grating(H, W, 180.0)
    assert not np.allclose(a(3), b(3))


def test_grating_is_periodic_along_its_axis():
    g = drifting_grating(H, W, 0.0, spatial_period_px=20.0, speed_px_per_frame=0.0)
    frame = g(0)
    assert np.allclose(frame[:, 0], frame[:, 20], atol=1e-5)


def test_looming_disc_grows_monotonically():
    loom = looming_disc(H, W, n_frames=60)
    dark = [float((loom(i) < 0.5).sum()) for i in range(0, 60, 6)]
    assert all(b >= a for a, b in zip(dark, dark[1:])), dark
    assert dark[-1] > dark[0]


def test_looming_disc_is_centred():
    loom = looming_disc(H, W, n_frames=60)
    frame = loom(50)
    ys, xs = np.nonzero(frame < 0.5)
    assert abs(ys.mean() - H / 2) < 3 and abs(xs.mean() - W / 2) < 3


def test_contrast_step_switches_at_the_requested_frame():
    step = contrast_step(H, W, low=0.2, high=0.8, switch_frame=10)
    assert abs(float(step(9).mean()) - 0.2) < 1e-6
    assert abs(float(step(10).mean()) - 0.8) < 1e-6
```

```python
# tests/drivers/test_batch.py
import numpy as np
import pytest
import torch

from flybrain.core.lif import LIFNetwork, LIFParams
from flybrain.core.runaway import RunawayDetector, RunawayHalt
from flybrain.drivers.batch import run_batch
from flybrain.encode.source import Frame, SyntheticSource


class StubEncoder:
    """Drives neuron 0 in proportion to mean frame luminance."""

    def __init__(self, n):
        self.n = n

    def reset(self):
        pass

    def encode(self, frame: Frame) -> np.ndarray:
        out = np.zeros(self.n, dtype=np.float32)
        out[0] = float(frame.image.mean()) * 40.0
        return out


def tiny_net(n=4):
    indptr = np.array([0, 1, 1, 1, 1], dtype=np.int64)
    indices = np.array([1], dtype=np.int32)
    weights = np.array([9.0], dtype=np.float32)
    return LIFNetwork(indptr, indices, weights, LIFParams())


def test_returns_one_value_per_frame_per_group():
    net = tiny_net()
    src = SyntheticSource(lambda i: np.full((4, 4), 1.0, np.float32), 4, 4)
    out = run_batch(
        net, StubEncoder(4), src, {"driven": np.array([0]), "target": np.array([1])},
        n_frames=5, steps_per_frame=10,
    )
    assert out["driven"].shape == (5,)
    assert out["target"].shape == (5,)
    assert out["spike_fraction"].shape == (5,)


def test_bright_input_produces_spikes_in_the_driven_group():
    net = tiny_net()
    src = SyntheticSource(lambda i: np.full((4, 4), 1.0, np.float32), 4, 4)
    out = run_batch(
        net, StubEncoder(4), src, {"driven": np.array([0])},
        n_frames=5, steps_per_frame=10,
    )
    assert out["driven"].sum() > 0


def test_dark_input_produces_no_spikes():
    net = tiny_net()
    src = SyntheticSource(lambda i: np.zeros((4, 4), np.float32), 4, 4)
    out = run_batch(
        net, StubEncoder(4), src, {"driven": np.array([0])},
        n_frames=5, steps_per_frame=10,
    )
    assert out["driven"].sum() == 0


def test_batch_driver_halts_on_runaway():
    """Batch must halt, never silently clamp -- clamped data is contaminated."""
    net = tiny_net()
    src = SyntheticSource(lambda i: np.full((4, 4), 1.0, np.float32), 4, 4)
    detector = RunawayDetector(ceiling=0.0, window=1, mode="halt")
    with pytest.raises(RunawayHalt):
        run_batch(
            net, StubEncoder(4), src, {"driven": np.array([0])},
            n_frames=5, steps_per_frame=10, detector=detector,
        )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/stimuli/ tests/drivers/ -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'flybrain.stimuli'`

- [ ] **Step 3: Write the stimuli**

```python
# flybrain/stimuli/__init__.py
```

```python
# flybrain/stimuli/patterns.py
"""Synthetic stimuli with published response properties to compare against."""
from __future__ import annotations

import math
from typing import Callable

import numpy as np


def drifting_grating(
    height: int,
    width: int,
    direction_deg: float,
    spatial_period_px: float = 24.0,
    speed_px_per_frame: float = 2.0,
) -> Callable[[int], np.ndarray]:
    theta = math.radians(direction_deg)
    ys, xs = np.mgrid[0:height, 0:width]
    projection = xs * math.cos(theta) + ys * math.sin(theta)

    def generate(index: int) -> np.ndarray:
        phase = 2 * math.pi * (projection - index * speed_px_per_frame) / spatial_period_px
        return ((np.sin(phase) + 1.0) / 2.0).astype(np.float32)

    return generate


def looming_disc(
    height: int,
    width: int,
    r_over_v: float = 40.0,
    n_frames: int = 90,
) -> Callable[[int], np.ndarray]:
    """A dark disc expanding from the centre -- the classic escape stimulus.

    Angular size follows theta(t) = 2 * arctan(r / (v * t_to_collision)), so
    growth accelerates as collision approaches, as it does for a real
    approaching object.
    """
    ys, xs = np.mgrid[0:height, 0:width]
    distance = np.sqrt((xs - width / 2.0) ** 2 + (ys - height / 2.0) ** 2)
    max_radius = min(height, width) / 2.0

    def generate(index: int) -> np.ndarray:
        remaining = max(n_frames - index, 1)
        theta = 2.0 * math.atan(r_over_v / remaining)
        radius = min(theta / (math.pi / 2.0), 1.0) * max_radius
        frame = np.ones((height, width), dtype=np.float32)
        frame[distance <= radius] = 0.0
        return frame

    return generate


def contrast_step(
    height: int,
    width: int,
    low: float = 0.2,
    high: float = 0.8,
    switch_frame: int = 30,
) -> Callable[[int], np.ndarray]:
    def generate(index: int) -> np.ndarray:
        value = high if index >= switch_frame else low
        return np.full((height, width), value, dtype=np.float32)

    return generate
```

- [ ] **Step 4: Write the batch driver**

```python
# flybrain/drivers/__init__.py
```

```python
# flybrain/drivers/batch.py
"""Stimulus -> recorded activity.

Halts on runaway rather than clamping: an experiment that silently reduces its
gain and keeps recording produces data that looks fine and is not.
"""
from __future__ import annotations

import numpy as np
import torch

from flybrain.core.runaway import RunawayHalt


def run_batch(
    net,
    encoder,
    source,
    watch: dict[str, np.ndarray],
    n_frames: int,
    steps_per_frame: int = 33,
    detector=None,
) -> dict[str, np.ndarray]:
    net.reset()
    if hasattr(encoder, "reset"):
        encoder.reset()

    results = {name: np.zeros(n_frames, dtype=np.int64) for name in watch}
    results["spike_fraction"] = np.zeros(n_frames, dtype=np.float64)
    groups = {
        name: torch.as_tensor(np.asarray(idx), dtype=torch.long, device=net.device)
        for name, idx in watch.items()
    }

    for f in range(n_frames):
        frame = source.read()
        current = torch.as_tensor(encoder.encode(frame), device=net.device)
        fractions = []

        for _ in range(steps_per_frame):
            spikes = net.step(current)
            fractions.append(net.last_spike_fraction)
            for name, idx in groups.items():
                results[name][f] += int(spikes[idx].sum())
            if detector is not None:
                action = detector.update(net.last_spike_fraction)
                if action == "halt":
                    raise RunawayHalt(
                        f"activity saturated at frame {f}: "
                        f"{net.last_spike_fraction:.1%} of neurons spiking"
                    )
                if action == "clamp":
                    net.gain.mul_(0.5)

        results["spike_fraction"][f] = float(np.mean(fractions))

    return results
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/stimuli/ tests/drivers/ -v`
Expected: PASS (7 stimulus tests + 4 driver tests)

- [ ] **Step 6: MANUAL VERIFICATION**

```bash
mkdir -p figures
python -c "
import numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from flybrain.stimuli.patterns import drifting_grating, looming_disc, contrast_step
g = drifting_grating(120,160, 0.0); l = looming_disc(120,160, n_frames=90)
fig, ax = plt.subplots(2, 4, figsize=(12,6))
for k, i in enumerate([0, 5, 10, 15]):
    ax[0,k].imshow(g(i), cmap='gray', vmin=0, vmax=1); ax[0,k].set_title(f'grating f{i}')
for k, i in enumerate([0, 40, 70, 88]):
    ax[1,k].imshow(l(i), cmap='gray', vmin=0, vmax=1); ax[1,k].set_title(f'loom f{i}')
for a in ax.ravel(): a.axis('off')
plt.tight_layout(); plt.savefig('figures/stimuli.png', dpi=110)
"
```

Open `figures/stimuli.png`. The grating must visibly shift between frames, and the disc must expand — slowly at first, then rapidly. If the disc's growth looks linear, the `r_over_v` geometry is wrong and the looming tuning result will be meaningless.

- [ ] **Step 7: Commit**

```bash
git add flybrain/stimuli/ flybrain/drivers/ tests/stimuli/ tests/drivers/
git commit -m "feat: drifting gratings, looming discs, contrast steps, and batch driver"
```

---

### Task 14: Metrics, the battery, and the pass criterion

**Files:**
- Create: `flybrain/analysis/metrics.py`, `flybrain/analysis/battery.py`
- Test: `tests/analysis/test_metrics.py`, `tests/analysis/test_battery.py`

**Interfaces:**
- Consumes: `run_batch` (Task 13); stimuli (Task 13); `NULL_MODELS` (Task 12); `homeostatic_tune` (Task 8); `LIFNetwork` (Task 5); `Graph` (Task 4).
- Produces:
  - `direction_selectivity_index(preferred: float, opposite: float) -> float` — `(pref - opp) / (pref + opp)`, `0.0` when both are zero.
  - `bootstrap_ci(values: np.ndarray, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float]`
  - `intervals_overlap(a: tuple[float, float], b: tuple[float, float]) -> bool`
  - `looming_discrimination(loom_counts: np.ndarray, control_counts: np.ndarray) -> float` — the normalised difference in total escape-pathway spiking.
  - `run_battery(graph, device="cuda", seeds=(0, 1, 2, 3, 4), out_dir=Path("out")) -> dict` — runs the real graph and every null at every seed, tuning each identically, and writes `out/battery.json`.
  - `evaluate_pass_criterion(results: dict) -> dict` returning `{"dsi_pass": bool, "looming_pass": bool, "overall": bool, "detail": str}`.

**The pass criterion is committed in advance and lives in code, not in a later judgement call:** the real connectome must beat `degree_preserving` on both direction selectivity and looming discrimination with non-overlapping 95% confidence intervals. If it does not, `overall` is `False`, the finding goes in the README, and the project reports it.

- [ ] **Step 1: Write the failing tests**

```python
# tests/analysis/test_metrics.py
import numpy as np
import pytest

from flybrain.analysis.metrics import (
    bootstrap_ci,
    direction_selectivity_index,
    intervals_overlap,
    looming_discrimination,
)


def test_dsi_is_one_for_a_perfectly_selective_response():
    assert direction_selectivity_index(10.0, 0.0) == pytest.approx(1.0)


def test_dsi_is_zero_for_an_unselective_response():
    assert direction_selectivity_index(10.0, 10.0) == pytest.approx(0.0)


def test_dsi_is_negative_when_the_opposite_direction_wins():
    assert direction_selectivity_index(2.0, 8.0) == pytest.approx(-0.6)


def test_dsi_of_silence_is_zero_not_nan():
    assert direction_selectivity_index(0.0, 0.0) == 0.0


def test_bootstrap_ci_brackets_the_mean():
    values = np.random.default_rng(0).normal(5.0, 1.0, size=200)
    low, high = bootstrap_ci(values, seed=0)
    assert low < values.mean() < high


def test_bootstrap_ci_is_reproducible():
    values = np.arange(50, dtype=float)
    assert bootstrap_ci(values, seed=3) == bootstrap_ci(values, seed=3)


def test_tighter_data_gives_a_narrower_interval():
    rng = np.random.default_rng(1)
    wide = bootstrap_ci(rng.normal(0, 5.0, 200), seed=0)
    tight = bootstrap_ci(rng.normal(0, 0.1, 200), seed=0)
    assert (tight[1] - tight[0]) < (wide[1] - wide[0])


def test_overlap_detection():
    assert intervals_overlap((0.0, 1.0), (0.5, 2.0)) is True
    assert intervals_overlap((0.0, 1.0), (1.5, 2.0)) is False
    assert intervals_overlap((1.5, 2.0), (0.0, 1.0)) is False


def test_looming_discrimination_is_positive_when_loom_drives_more():
    loom = np.array([1, 2, 8, 20], dtype=float)
    control = np.array([1, 1, 1, 1], dtype=float)
    assert looming_discrimination(loom, control) > 0


def test_looming_discrimination_is_zero_for_identical_responses():
    same = np.array([3, 3, 3], dtype=float)
    assert looming_discrimination(same, same) == pytest.approx(0.0)
```

```python
# tests/analysis/test_battery.py
from flybrain.analysis.battery import evaluate_pass_criterion


def _results(real_dsi_ci, null_dsi_ci, real_loom_ci, null_loom_ci):
    return {
        "dsi": {"real": {"ci": real_dsi_ci}, "degree_preserving": {"ci": null_dsi_ci}},
        "looming": {"real": {"ci": real_loom_ci}, "degree_preserving": {"ci": null_loom_ci}},
    }


def test_passes_when_real_clearly_beats_the_null_on_both():
    out = evaluate_pass_criterion(
        _results((0.6, 0.8), (0.0, 0.2), (0.5, 0.7), (0.0, 0.1))
    )
    assert out["dsi_pass"] and out["looming_pass"] and out["overall"]


def test_fails_when_intervals_overlap():
    out = evaluate_pass_criterion(
        _results((0.3, 0.8), (0.2, 0.5), (0.5, 0.7), (0.0, 0.1))
    )
    assert out["dsi_pass"] is False
    assert out["overall"] is False


def test_fails_when_the_null_scores_higher():
    out = evaluate_pass_criterion(
        _results((0.0, 0.1), (0.6, 0.8), (0.5, 0.7), (0.0, 0.1))
    )
    assert out["dsi_pass"] is False
    assert out["overall"] is False


def test_both_metrics_are_required():
    out = evaluate_pass_criterion(
        _results((0.6, 0.8), (0.0, 0.2), (0.1, 0.2), (0.15, 0.4))
    )
    assert out["dsi_pass"] is True
    assert out["looming_pass"] is False
    assert out["overall"] is False


def test_detail_explains_the_outcome():
    out = evaluate_pass_criterion(
        _results((0.6, 0.8), (0.0, 0.2), (0.5, 0.7), (0.0, 0.1))
    )
    assert "dsi" in out["detail"].lower()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/analysis/test_metrics.py tests/analysis/test_battery.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'flybrain.analysis.metrics'`

- [ ] **Step 3: Write the metrics**

```python
# flybrain/analysis/metrics.py
"""Effect sizes and confidence intervals for the validation battery."""
from __future__ import annotations

import numpy as np


def direction_selectivity_index(preferred: float, opposite: float) -> float:
    total = preferred + opposite
    if total == 0:
        return 0.0
    return float((preferred - opposite) / total)


def bootstrap_ci(
    values: np.ndarray, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    return (
        float(np.percentile(draws, 100 * alpha / 2)),
        float(np.percentile(draws, 100 * (1 - alpha / 2))),
    )


def intervals_overlap(a: tuple[float, float], b: tuple[float, float]) -> bool:
    return not (a[1] < b[0] or b[1] < a[0])


def looming_discrimination(loom_counts: np.ndarray, control_counts: np.ndarray) -> float:
    """How much more the escape pathway fires to a looming disc than to a
    non-looming control, normalised to [-1, 1]."""
    loom_total = float(np.sum(loom_counts))
    control_total = float(np.sum(control_counts))
    total = loom_total + control_total
    if total == 0:
        return 0.0
    return (loom_total - control_total) / total
```

- [ ] **Step 4: Write the battery**

```python
# flybrain/analysis/battery.py
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


def _tuning_drive(net, seed: int = 12345):
    """A fixed, reproducible background stimulus for calibration.

    Tuning against zero input would be meaningless -- with no activity the
    update rule can only ratchet gains upward. A deterministic noise drive
    gives every condition, real and null, the identical calibration stimulus.
    """
    generator = torch.Generator(device="cpu").manual_seed(seed)
    drive = (
        torch.rand(net.n_neurons, generator=generator) * 2.0
    ).to(net.device)
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
        conditions["real"].append(
            _one_condition(graph, graph.indptr, graph.indices, graph.weights, device, seed)
        )

    for name, fn in NULL_MODELS.items():
        conditions[name] = []
        for seed in seeds:
            args = (
                (graph.indptr, graph.indices, graph.weights, graph.types, seed)
                if name == "type_preserving"
                else (graph.indptr, graph.indices, graph.weights, seed)
            )
            p, i, w = fn(*args)
            conditions[name].append(_one_condition(graph, p, i, w, device, seed))

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
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/analysis/test_metrics.py tests/analysis/test_battery.py -v`
Expected: PASS (15 tests)

- [ ] **Step 6: Run the real battery**

```bash
python -c "
from pathlib import Path
from flybrain.analysis.battery import run_battery
from flybrain.ingest.graph import load_graph
from flybrain.paths import ARTIFACTS
g = load_graph(ARTIFACTS/'graph_t2.npz')
r = run_battery(g, device='cuda', seeds=(0,1,2,3,4), out_dir=Path('out'))
print(r['verdict']['detail'])
for m in ('dsi','looming'):
    for name, v in r[m].items():
        print(f'{m:8s} {name:20s} mean={v[\"mean\"]:+.4f}  CI=[{v[\"ci\"][0]:+.4f}, {v[\"ci\"][1]:+.4f}]')
"
```

This is the long one — 25 conditions (5 seeds × real + 4 nulls), each running four stimulus sweeps. Expect a substantial wall-clock time. Run it when you can leave the machine alone.

- [ ] **Step 7: Draw the comparison figure**

```bash
python -c "
import json, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
r = json.load(open('out/battery.json'))
fig, axes = plt.subplots(1, 2, figsize=(11,4))
for ax, metric in zip(axes, ('dsi','looming')):
    names = list(r[metric].keys())
    means = [r[metric][n]['mean'] for n in names]
    los = [r[metric][n]['mean']-r[metric][n]['ci'][0] for n in names]
    his = [r[metric][n]['ci'][1]-r[metric][n]['mean'] for n in names]
    colors = ['#c0392b' if n=='real' else '#7f8c8d' for n in names]
    ax.bar(range(len(names)), means, yerr=[los,his], capsize=4, color=colors)
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=30, ha='right')
    ax.set_title(metric); ax.axhline(0, color='k', lw=0.6)
plt.suptitle('Real connectome vs shuffled-wiring nulls (95% CI, 5 seeds)')
plt.tight_layout(); plt.savefig('figures/battery.png', dpi=130)
print(r['verdict']['detail'])
"
```

- [ ] **Step 8: MANUAL VERIFICATION — and record the result honestly**

Open `figures/battery.png`. Read the verdict line.

- **If `overall` is True:** the red bar clears the grey `degree_preserving` bar with no overlap on both panels. You have a real result.
- **If `overall` is False:** that is the honest outcome the spec committed to reporting. **Write it in the README.** Do not retune, reseed, or adjust the criterion to get a pass — the criterion was fixed in advance precisely so this moment isn't negotiable. The avatar works either way; what changes is the claim you make about it.

Either way, add a `## Validation` section to `README.md` quoting the verdict line and the four CI pairs, and commit `out/battery.json` so the numbers are in version control.

- [ ] **Step 9: Commit**

```bash
git add flybrain/analysis/metrics.py flybrain/analysis/battery.py tests/analysis/ out/battery.json README.md
git commit -m "feat: validation battery with pre-committed pass criterion against nulls"
```

---

# Phase 7 — Readout and the signal bus

### Task 15: Named signals with provenance

**Files:**
- Create: `flybrain/readout/__init__.py`, `flybrain/readout/signals.py`
- Test: `tests/readout/test_signals.py`

**Interfaces:**
- Consumes: `Graph` (Task 4).
- Produces:
  - `@dataclass SignalSpec` with `name: str`, `cell_types: list[str]`, `description: str`.
  - `SIGNAL_SPECS: list[SignalSpec]`
  - `class SignalReadout(graph, dt_ms: float = 1.0, arousal_tau_ms: float = 2000.0, startle_threshold: int = 1)` with:
    - `.update(spikes: torch.Tensor) -> dict` returning `{"startle": float, "flow": [fx, fy], "salience": [sx, sy], "arousal": float, "schema": 1}`
    - `.provenance -> dict[str, dict]` — for each signal, the cell types and the number of neurons behind it
    - `.reset()`

**`valence` is deliberately absent and a test enforces that.** Fly valence runs substantially through dopamine and octopamine, which the ingest stage assigns sign `0`. Emitting a "connectome-derived valence" would invent exactly what this data cannot support.

T4/T5 subtypes tile four cardinal directions: `a` → front-to-back (0°), `b` → back-to-front (180°), `c` → upward (90°), `d` → downward (270°). The flow signal is their population vector sum.

- [ ] **Step 1: Write the failing test**

```python
# tests/readout/test_signals.py
import numpy as np
import pytest
import torch

from flybrain.ingest.graph import load_graph
from flybrain.paths import ARTIFACTS
from flybrain.readout.signals import SIGNAL_SPECS, SignalReadout

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def graph():
    return load_graph(ARTIFACTS / "graph_t2.npz")


@pytest.fixture
def readout(graph):
    return SignalReadout(graph)


def silence(graph):
    return torch.zeros(graph.n_neurons, dtype=torch.bool)


def fire(graph, idx):
    s = silence(graph)
    s[torch.as_tensor(np.asarray(idx), dtype=torch.long)] = True
    return s


def test_emits_exactly_the_documented_signals(readout, graph):
    out = readout.update(silence(graph))
    assert set(out) == {"startle", "flow", "salience", "arousal", "schema"}


def test_valence_is_never_emitted(readout, graph):
    """Modulators are zeroed in v1, so a connectome-derived valence would be
    invented. Its absence is a design decision, not an oversight."""
    assert "valence" not in readout.update(silence(graph))
    assert all(spec.name != "valence" for spec in SIGNAL_SPECS)


def test_every_signal_declares_its_neurons(readout):
    for name, info in readout.provenance.items():
        assert info["cell_types"], f"{name} declares no cell types"
        assert info["n_neurons"] > 0, f"{name} is backed by no neurons"
        assert info["description"]


def test_silence_produces_zero_signals(readout, graph):
    out = readout.update(silence(graph))
    assert out["startle"] == 0.0
    assert out["flow"] == [0.0, 0.0]
    assert out["arousal"] == pytest.approx(0.0, abs=1e-6)


def test_giant_fiber_spike_produces_startle(readout, graph):
    out = readout.update(fire(graph, graph.type_index("DNp01")))
    assert out["startle"] > 0.0


def test_startle_scales_with_how_many_escape_neurons_fire(readout, graph):
    small = SignalReadout(graph).update(fire(graph, graph.type_index("DNp01")))
    big_idx = np.concatenate([graph.type_index("DNp01"), graph.type_index("DNp02")])
    big = SignalReadout(graph).update(fire(graph, big_idx))
    assert big["startle"] >= small["startle"]


def test_opposite_t4_subtypes_cancel(readout, graph):
    """T4a and T4b are tuned to opposite directions; firing both equally must
    yield near-zero net flow."""
    both = np.concatenate([graph.type_index("T4a"), graph.type_index("T4b")])
    out = readout.update(fire(graph, both))
    assert abs(out["flow"][0]) < 0.15


def test_single_t4_subtype_produces_directed_flow(readout, graph):
    out = readout.update(fire(graph, graph.type_index("T4a")))
    assert abs(out["flow"][0]) > 0.5
    assert out["flow"][0] > 0, "T4a is tuned front-to-back, i.e. +x"


def test_t4c_drives_the_vertical_axis(readout, graph):
    out = readout.update(fire(graph, graph.type_index("T4c")))
    assert abs(out["flow"][1]) > 0.5


def test_arousal_persists_after_input_stops(readout, graph):
    """Persistence is the Anderson-lab emotion primitive this signal borrows."""
    central = graph.superclass_index("cb_intrinsic")
    for _ in range(50):
        readout.update(fire(graph, central[:5000]))
    peak = readout.update(silence(graph))["arousal"]
    for _ in range(10):
        after = readout.update(silence(graph))["arousal"]
    assert peak > 0
    assert after > 0, "arousal collapsed immediately -- it is not persistent"
    assert after < peak, "arousal never decays"


def test_arousal_scales_with_activity(readout, graph):
    central = graph.superclass_index("cb_intrinsic")
    weak = SignalReadout(graph)
    strong = SignalReadout(graph)
    for _ in range(30):
        weak.update(fire(graph, central[:1000]))
        strong.update(fire(graph, central[:10000]))
    assert strong.update(silence(graph))["arousal"] > weak.update(silence(graph))["arousal"]


def test_reset_clears_arousal(readout, graph):
    central = graph.superclass_index("cb_intrinsic")
    for _ in range(30):
        readout.update(fire(graph, central[:5000]))
    readout.reset()
    assert readout.update(silence(graph))["arousal"] == pytest.approx(0.0, abs=1e-6)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/readout/test_signals.py -v -m slow`
Expected: FAIL with `ModuleNotFoundError: No module named 'flybrain.readout'`

- [ ] **Step 3: Write minimal implementation**

```python
# flybrain/readout/__init__.py
```

```python
# flybrain/readout/signals.py
"""Spike trains -> named signals.

Every signal declares the neurons it comes from. That is what keeps the claim
"the fly brain is driving this" checkable rather than decorative.

There is no valence signal. Fly valence runs substantially through dopamine
and octopamine, which the ingest stage assigns sign 0, so emitting one would
mean inventing what the data cannot support.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch

SCHEMA_VERSION = 1

# T4/T5 subtypes tile four cardinal directions.
DIRECTION_VECTORS = {
    "a": (1.0, 0.0),    # front-to-back
    "b": (-1.0, 0.0),   # back-to-front
    "c": (0.0, 1.0),    # upward
    "d": (0.0, -1.0),   # downward
}


@dataclass
class SignalSpec:
    name: str
    cell_types: list[str]
    description: str


SIGNAL_SPECS: list[SignalSpec] = [
    SignalSpec(
        "startle",
        ["DNp01", "DNp02", "DNp03", "LC4", "LPLC2"],
        "Escape pathway: looming detectors gating the Giant Fiber.",
    ),
    SignalSpec(
        "flow",
        [f"{t}{s}" for t in ("T4", "T5") for s in "abcd"],
        "Population vector over direction-selective cells tiling four directions.",
    ),
    SignalSpec(
        "salience",
        ["LC4", "LC6", "LPLC1", "LPLC2"],
        "Object detectors indicating where something notable is.",
    ),
    SignalSpec(
        "arousal",
        ["cb_intrinsic (superclass)"],
        "Central-brain population rate through a leaky integrator; "
        "persistent and scalable, after Anderson-lab emotion primitives.",
    ),
]


class SignalReadout:
    def __init__(
        self,
        graph,
        dt_ms: float = 1.0,
        arousal_tau_ms: float = 2000.0,
        startle_threshold: int = 1,
    ) -> None:
        self.startle_threshold = startle_threshold
        self.arousal_decay = math.exp(-dt_ms / arousal_tau_ms)

        self._escape = self._gather(graph, ["DNp01", "DNp02", "DNp03"])
        self._loom = self._gather(graph, ["LC4", "LPLC2"])
        self._salience = self._gather(graph, ["LC4", "LC6", "LPLC1", "LPLC2"])
        self._central = graph.superclass_index("cb_intrinsic")

        self._flow_groups = []
        for family in ("T4", "T5"):
            for suffix, vector in DIRECTION_VECTORS.items():
                idx = graph.type_index(f"{family}{suffix}")
                if len(idx):
                    self._flow_groups.append((idx, vector))

        self._provenance = {
            spec.name: {
                "cell_types": spec.cell_types,
                "description": spec.description,
                "n_neurons": self._count(graph, spec),
            }
            for spec in SIGNAL_SPECS
        }
        self.reset()

    @staticmethod
    def _gather(graph, names: list[str]) -> np.ndarray:
        parts = [graph.type_index(n) for n in names]
        parts = [p for p in parts if len(p)]
        return np.concatenate(parts) if parts else np.zeros(0, dtype=np.int64)

    def _count(self, graph, spec: SignalSpec) -> int:
        if spec.name == "arousal":
            return int(len(self._central))
        return int(len(self._gather(graph, spec.cell_types)))

    @property
    def provenance(self) -> dict:
        return self._provenance

    def reset(self) -> None:
        self._arousal = 0.0

    def update(self, spikes: torch.Tensor) -> dict:
        spikes_cpu = spikes.to("cpu")

        def count(idx: np.ndarray) -> int:
            if len(idx) == 0:
                return 0
            return int(spikes_cpu[torch.as_tensor(idx, dtype=torch.long)].sum())

        escape_n = count(self._escape)
        loom_n = count(self._loom)
        startle = 0.0
        if escape_n >= self.startle_threshold:
            startle = float(escape_n) * (1.0 + math.log1p(loom_n))

        fx = fy = 0.0
        for idx, (vx, vy) in self._flow_groups:
            n = count(idx)
            if n:
                rate = n / len(idx)
                fx += vx * rate
                fy += vy * rate
        magnitude = math.hypot(fx, fy)
        if magnitude > 1.0:
            fx, fy = fx / magnitude, fy / magnitude

        salience_n = count(self._salience)
        salience = [
            float(np.clip(fx * salience_n / max(len(self._salience), 1) * 10.0, -1, 1)),
            float(np.clip(fy * salience_n / max(len(self._salience), 1) * 10.0, -1, 1)),
        ]

        central_rate = count(self._central) / max(len(self._central), 1)
        self._arousal = self._arousal * self.arousal_decay + central_rate * (
            1.0 - self.arousal_decay
        )

        return {
            "startle": startle,
            "flow": [float(fx), float(fy)],
            "salience": salience,
            "arousal": float(self._arousal),
            "schema": SCHEMA_VERSION,
        }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/readout/test_signals.py -v -m slow`
Expected: PASS (12 tests)

- [ ] **Step 5: MANUAL VERIFICATION**

```bash
python -c "
import json
from flybrain.ingest.graph import load_graph
from flybrain.paths import ARTIFACTS
from flybrain.readout.signals import SignalReadout
g = load_graph(ARTIFACTS/'graph_t2.npz')
r = SignalReadout(g)
print(json.dumps(r.provenance, indent=2))
"
```

Read the provenance block. Every signal must name real cell types and a non-zero neuron count. **This block is the honesty contract** — it's what you show anyone who asks whether the fly brain is really driving the avatar. If a signal here is backed by zero neurons, it is decoration and must be removed rather than shipped.

- [ ] **Step 6: Commit**

```bash
git add flybrain/readout/ tests/readout/
git commit -m "feat: named signals with declared provenance; valence deliberately omitted"
```

---

### Task 16: Live driver and WebSocket signal bus

**Files:**
- Create: `flybrain/drivers/live.py`, `flybrain/bus.py`
- Test: `tests/drivers/test_live.py`, `tests/test_bus.py`

**Interfaces:**
- Consumes: `LIFNetwork` (Task 5), `RunawayDetector` (Task 8), `LaminaEncoder` + `FrameSource` (Task 11), `SignalReadout` (Task 15).
- Produces:
  - `class SignalBus(host="127.0.0.1", port=8765, max_queue=4)` with `async .serve()`, `.publish(payload: dict) -> None` (non-blocking, drops for slow subscribers), `.subscriber_count -> int`, `async .close()`.
  - `class LiveDriver(net, encoder, source, readout, bus=None, steps_per_publish=16, speed=1.0, detector=None)` with `.run_once() -> dict` (one publish interval) and `.stats -> dict` carrying `drift_ms`, `steps`, `degraded`.

The driver **clamps** on runaway and sets `degraded` — it never halts, because a frozen stream is dead.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_bus.py
import asyncio
import json

import pytest
import websockets

from flybrain.bus import SignalBus


@pytest.mark.asyncio
async def test_subscriber_receives_published_payload():
    bus = SignalBus(port=8791)
    server = await bus.serve()
    try:
        async with websockets.connect("ws://127.0.0.1:8791") as client:
            await asyncio.sleep(0.05)
            bus.publish({"arousal": 0.5, "schema": 1})
            message = await asyncio.wait_for(client.recv(), timeout=2.0)
            assert json.loads(message)["arousal"] == 0.5
    finally:
        server.close()
        await bus.close()


@pytest.mark.asyncio
async def test_publish_without_subscribers_does_not_raise():
    bus = SignalBus(port=8792)
    server = await bus.serve()
    try:
        bus.publish({"arousal": 0.1})
        assert bus.subscriber_count == 0
    finally:
        server.close()
        await bus.close()


@pytest.mark.asyncio
async def test_subscriber_count_tracks_connections():
    bus = SignalBus(port=8793)
    server = await bus.serve()
    try:
        async with websockets.connect("ws://127.0.0.1:8793"):
            await asyncio.sleep(0.05)
            assert bus.subscriber_count == 1
        await asyncio.sleep(0.1)
        assert bus.subscriber_count == 0
    finally:
        server.close()
        await bus.close()


@pytest.mark.asyncio
async def test_slow_subscriber_is_dropped_not_awaited():
    """Backpressure must never reach the simulation core."""
    bus = SignalBus(port=8794, max_queue=2)
    server = await bus.serve()
    try:
        async with websockets.connect("ws://127.0.0.1:8794"):
            await asyncio.sleep(0.05)
            for i in range(500):
                bus.publish({"n": i})       # never awaits, never blocks
            assert bus.dropped > 0
    finally:
        server.close()
        await bus.close()
```

```python
# tests/drivers/test_live.py
import numpy as np
import torch

from flybrain.core.lif import LIFNetwork, LIFParams
from flybrain.core.runaway import RunawayDetector
from flybrain.drivers.live import LiveDriver
from flybrain.encode.source import Frame, SyntheticSource


class StubEncoder:
    def __init__(self, n):
        self.n = n

    def reset(self):
        pass

    def encode(self, frame: Frame) -> np.ndarray:
        out = np.zeros(self.n, dtype=np.float32)
        out[0] = float(frame.image.mean()) * 40.0
        return out


class StubReadout:
    def __init__(self):
        self.calls = 0

    def reset(self):
        self.calls = 0

    def update(self, spikes):
        self.calls += 1
        return {"arousal": float(spikes.sum()), "schema": 1}


def tiny_net(n=4):
    return LIFNetwork(
        np.array([0, 1, 1, 1, 1], dtype=np.int64),
        np.array([1], dtype=np.int32),
        np.array([9.0], dtype=np.float32),
        LIFParams(),
    )


def make_driver(**kw):
    net = tiny_net()
    src = SyntheticSource(lambda i: np.full((4, 4), 1.0, np.float32), 4, 4)
    return LiveDriver(net, StubEncoder(4), src, StubReadout(), steps_per_publish=8, **kw)


def test_run_once_returns_a_signal_payload():
    payload = make_driver().run_once()
    assert "arousal" in payload and payload["schema"] == 1


def test_stats_report_steps_and_drift():
    d = make_driver()
    d.run_once()
    assert d.stats["steps"] == 8
    assert "drift_ms" in d.stats


def test_repeated_calls_accumulate_steps():
    d = make_driver()
    d.run_once()
    d.run_once()
    assert d.stats["steps"] == 16


def test_runaway_clamps_gain_and_flags_degraded_without_raising():
    """Live must survive what batch refuses to tolerate."""
    d = make_driver(detector=RunawayDetector(ceiling=0.0, window=1, mode="clamp"))
    before = float(d.net.gain[0])
    d.run_once()
    assert d.stats["degraded"] is True
    assert float(d.net.gain[0]) < before


def test_source_failure_does_not_stop_the_loop():
    class FailingSource(SyntheticSource):
        def read(self):
            return self._hold_last()

    net = tiny_net()
    d = LiveDriver(
        net, StubEncoder(4), FailingSource(lambda i: np.zeros((4, 4), np.float32), 4, 4),
        StubReadout(), steps_per_publish=4,
    )
    payload = d.run_once()
    assert payload is not None
    assert d.stats["steps"] == 4
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_bus.py tests/drivers/test_live.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'flybrain.bus'`

- [ ] **Step 3: Write the bus**

```python
# flybrain/bus.py
"""WebSocket signal bus.

VTube Studio's plugin API is a WebSocket and a browser viewer is WebSocket
native, so both are just subscribers here. Later consumers -- an LLM layer, an
OBS overlay, a recorder -- attach the same way without touching the engine.

publish() never awaits and never blocks. A slow subscriber loses packets; the
simulation's clock is not negotiable.
"""
from __future__ import annotations

import asyncio
import json

import websockets


class SignalBus:
    def __init__(self, host: str = "127.0.0.1", port: int = 8765, max_queue: int = 4) -> None:
        self.host = host
        self.port = port
        self.max_queue = max_queue
        self._queues: dict[object, asyncio.Queue] = {}
        self.dropped = 0

    @property
    def subscriber_count(self) -> int:
        return len(self._queues)

    async def _handler(self, connection) -> None:
        queue: asyncio.Queue = asyncio.Queue(maxsize=self.max_queue)
        self._queues[connection] = queue
        try:
            while True:
                payload = await queue.get()
                await connection.send(payload)
        except Exception:
            pass
        finally:
            self._queues.pop(connection, None)

    async def serve(self):
        self._server = await websockets.serve(self._handler, self.host, self.port)
        return self._server

    def publish(self, payload: dict) -> None:
        if not self._queues:
            return
        message = json.dumps(payload)
        for queue in list(self._queues.values()):
            try:
                queue.put_nowait(message)
            except asyncio.QueueFull:
                self.dropped += 1

    async def close(self) -> None:
        self._queues.clear()
        server = getattr(self, "_server", None)
        if server is not None:
            server.close()
            await server.wait_closed()
```

- [ ] **Step 4: Write the live driver**

```python
# flybrain/drivers/live.py
"""Real-time loop with its own clock.

The core runs far faster than real time, so it throttles against the wall
clock and records drift. Input is read non-blocking: with no new frame the
last one persists, so a camera stall means the fly sees a static scene rather
than the simulation stuttering.

Clamps on runaway and flags `degraded`. It never halts -- a frozen stream is
dead, while a briefly desensitised one is merely worse.
"""
from __future__ import annotations

import time

import numpy as np
import torch


class LiveDriver:
    def __init__(
        self,
        net,
        encoder,
        source,
        readout,
        bus=None,
        steps_per_publish: int = 16,
        speed: float = 1.0,
        detector=None,
    ) -> None:
        self.net = net
        self.encoder = encoder
        self.source = source
        self.readout = readout
        self.bus = bus
        self.steps_per_publish = steps_per_publish
        self.speed = speed
        self.detector = detector

        self._current = torch.zeros(net.n_neurons, dtype=torch.float32, device=net.device)
        self._next_deadline = time.perf_counter()
        self.stats = {"steps": 0, "drift_ms": 0.0, "degraded": False, "frames": 0}

    def _refresh_input(self) -> None:
        frame = self.source.read()
        self.stats["frames"] += 1
        self._current = torch.as_tensor(
            self.encoder.encode(frame), dtype=torch.float32, device=self.net.device
        )

    def run_once(self) -> dict:
        self._refresh_input()

        spikes = None
        for _ in range(self.steps_per_publish):
            spikes = self.net.step(self._current)
            self.stats["steps"] += 1
            if self.detector is not None:
                action = self.detector.update(self.net.last_spike_fraction)
                if action in ("clamp", "halt"):
                    # Live never halts, whatever the detector's configured mode.
                    self.net.gain.mul_(0.5)
                    self.stats["degraded"] = True

        payload = self.readout.update(spikes)
        payload["degraded"] = self.stats["degraded"]

        if self.bus is not None:
            self.bus.publish(payload)

        interval = self.steps_per_publish * self.net.params.dt / 1000.0 / self.speed
        self._next_deadline += interval
        now = time.perf_counter()
        self.stats["drift_ms"] = (now - self._next_deadline) * 1000.0
        if now < self._next_deadline:
            time.sleep(self._next_deadline - now)
        else:
            self._next_deadline = now

        return payload
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_bus.py tests/drivers/test_live.py -v`
Expected: PASS (4 bus tests + 5 driver tests)

If the bus tests hang, confirm `pytest-asyncio` is in asyncio mode — add `asyncio_mode = auto` under `[pytest]` in `pytest.ini`.

- [ ] **Step 6: MANUAL VERIFICATION — the debug viewer**

Create `tools/viewer.html`:

```html
<!doctype html>
<meta charset="utf-8">
<title>Fly signals</title>
<style>
  body { background:#111; color:#eee; font:14px monospace; padding:24px; }
  .row { margin:8px 0; }
  .bar { display:inline-block; height:14px; background:#4fc3f7; vertical-align:middle; }
  #startle { background:#e53935; }
</style>
<h2>Live connectome signals</h2>
<div class="row">arousal <span class="bar" id="arousal"></span> <span id="av"></span></div>
<div class="row">startle <span class="bar" id="startle"></span> <span id="sv"></span></div>
<div class="row">flow    <span id="fv"></span></div>
<div class="row">steps   <span id="meta"></span></div>
<script>
const ws = new WebSocket("ws://127.0.0.1:8765");
ws.onmessage = (e) => {
  const d = JSON.parse(e.data);
  arousal.style.width = Math.min(300, d.arousal * 3000) + "px";
  av.textContent = d.arousal.toFixed(5);
  startle.style.width = Math.min(300, d.startle * 20) + "px";
  sv.textContent = d.startle.toFixed(2);
  fv.textContent = `x=${d.flow[0].toFixed(3)}  y=${d.flow[1].toFixed(3)}`;
  meta.textContent = d.degraded ? "DEGRADED" : "ok";
};
</script>
```

Then run the live loop against your webcam:

```bash
python -c "
import asyncio, threading
from flybrain.bus import SignalBus
from flybrain.core.lif import LIFNetwork, LIFParams
from flybrain.core.runaway import RunawayDetector
from flybrain.drivers.live import LiveDriver
from flybrain.encode.encoder import LaminaEncoder
from flybrain.encode.source import WebcamSource
from flybrain.ingest.graph import load_graph
from flybrain.paths import ARTIFACTS
from flybrain.readout.signals import SignalReadout

g = load_graph(ARTIFACTS/'graph_t2.npz')
net = LIFNetwork.from_graph(g, LIFParams(), device='cuda')
drv = LiveDriver(net, LaminaEncoder(g,120,160), WebcamSource(0,120,160),
                 SignalReadout(g), detector=RunawayDetector())

async def main():
    bus = SignalBus(port=8765); await bus.serve(); drv.bus = bus
    print('bus on ws://127.0.0.1:8765 -- open tools/viewer.html')
    loop = asyncio.get_running_loop()
    while True:
        await loop.run_in_executor(None, drv.run_once)
        await asyncio.sleep(0)
asyncio.run(main())
"
```

Open `tools/viewer.html` in a browser. **Wave your hand quickly across the camera and watch the bars move.** Arousal should build and decay slowly; flow should swing with the direction of your movement; startle should spike on sudden large motion. Check `drift_ms` stays small — large positive drift means the loop is falling behind real time.

- [ ] **Step 7: Commit**

```bash
mkdir -p tools
git add flybrain/bus.py flybrain/drivers/live.py tools/viewer.html tests/test_bus.py tests/drivers/test_live.py
git commit -m "feat: WebSocket signal bus and real-time driver with clock discipline"
```

---

# Phase 8 — VTube Studio bridge

### Task 17: Drive the avatar

**Files:**
- Create: `config/avatar_mapping.yaml`, `flybrain/bridge/__init__.py`, `flybrain/bridge/vts.py`
- Test: `tests/bridge/test_vts.py`

**Interfaces:**
- Consumes: signal payloads from the bus (Task 16).
- Produces:
  - `@dataclass Mapping` with `signal: str`, `component: int | None`, `parameter: str`, `scale: float`, `smooth_ms: float`, `decay: float`, `clamp: tuple[float, float]`.
  - `load_mappings(path) -> list[Mapping]`
  - `class ParameterShaper(mappings, dt_ms=16.0)` with `.shape(payload: dict) -> dict[str, float]` — **all smoothing and easing lives here, never in the core.**
  - `class VTSBridge(uri, mappings, plugin_name="FlyBrain", token_path=Path(".vts_token"))` with `async .connect()`, `async .authenticate()`, `async .send(values: dict[str, float])`, `async .run(bus_uri)` with reconnect backoff.

- [ ] **Step 1: Write the failing test**

```python
# tests/bridge/test_vts.py
import asyncio
import json

import pytest
import websockets

from flybrain.bridge.vts import Mapping, ParameterShaper, VTSBridge, load_mappings
from flybrain.paths import ROOT


@pytest.fixture
def mappings():
    return load_mappings(ROOT / "config" / "avatar_mapping.yaml")


def test_config_declares_mappings_for_the_real_signals(mappings):
    signals = {m.signal for m in mappings}
    assert {"startle", "arousal"} <= signals
    assert "valence" not in signals, "there is no valence signal to map"


def test_every_mapping_names_a_vts_parameter(mappings):
    for m in mappings:
        assert m.parameter and isinstance(m.scale, float)


def test_shaper_scales_a_scalar_signal():
    shaper = ParameterShaper(
        [Mapping("arousal", None, "FlyArousal", 100.0, 0.0, 1.0, (-1.0, 1.0))]
    )
    assert shaper.shape({"arousal": 0.005})["FlyArousal"] == pytest.approx(0.5)


def test_shaper_reads_a_vector_component():
    shaper = ParameterShaper(
        [Mapping("flow", 0, "FlyFlowX", 1.0, 0.0, 1.0, (-1.0, 1.0))]
    )
    assert shaper.shape({"flow": [0.7, -0.2]})["FlyFlowX"] == pytest.approx(0.7)


def test_shaper_clamps_to_the_configured_range():
    shaper = ParameterShaper(
        [Mapping("arousal", None, "FlyArousal", 1000.0, 0.0, 1.0, (-1.0, 1.0))]
    )
    assert shaper.shape({"arousal": 5.0})["FlyArousal"] == 1.0


def test_smoothing_approaches_the_target_over_several_frames():
    shaper = ParameterShaper(
        [Mapping("arousal", None, "FlyArousal", 1.0, 100.0, 1.0, (-1.0, 1.0))],
        dt_ms=16.0,
    )
    first = shaper.shape({"arousal": 1.0})["FlyArousal"]
    assert 0.0 < first < 1.0, "smoothing should not jump straight to target"
    for _ in range(40):
        value = shaper.shape({"arousal": 1.0})["FlyArousal"]
    assert value == pytest.approx(1.0, abs=0.05)


def test_decay_pulls_an_unfed_parameter_back_toward_zero():
    shaper = ParameterShaper(
        [Mapping("startle", None, "FlyStartle", 1.0, 0.0, 0.5, (-1.0, 1.0))]
    )
    shaper.shape({"startle": 1.0})
    after = shaper.shape({"startle": 0.0})["FlyStartle"]
    assert 0.0 <= after < 1.0


def test_missing_signal_does_not_raise():
    shaper = ParameterShaper(
        [Mapping("startle", None, "FlyStartle", 1.0, 0.0, 1.0, (-1.0, 1.0))]
    )
    assert shaper.shape({"arousal": 0.1}) == {}


@pytest.mark.asyncio
async def test_bridge_authenticates_and_sends_against_a_mock_vts():
    """A mock VTube Studio: assert the plugin handshake and the injection
    message are well formed without needing the real application."""
    received = []

    async def mock_vts(connection):
        async for raw in connection:
            message = json.loads(raw)
            received.append(message)
            kind = message["messageType"]
            if kind == "AuthenticationTokenRequest":
                await connection.send(json.dumps({
                    "messageType": "AuthenticationTokenResponse",
                    "data": {"authenticationToken": "tok123"},
                }))
            elif kind == "AuthenticationRequest":
                await connection.send(json.dumps({
                    "messageType": "AuthenticationResponse",
                    "data": {"authenticated": True},
                }))
            else:
                await connection.send(json.dumps({
                    "messageType": "InjectParameterDataResponse", "data": {},
                }))

    server = await websockets.serve(mock_vts, "127.0.0.1", 8801)
    try:
        bridge = VTSBridge("ws://127.0.0.1:8801", [], token_path=None)
        await bridge.connect()
        assert await bridge.authenticate() is True
        await bridge.send({"FlyArousal": 0.5})
        await bridge.close()
    finally:
        server.close()
        await server.wait_closed()

    kinds = [m["messageType"] for m in received]
    assert "AuthenticationTokenRequest" in kinds
    assert "AuthenticationRequest" in kinds
    assert "InjectParameterDataRequest" in kinds

    injection = next(m for m in received if m["messageType"] == "InjectParameterDataRequest")
    assert injection["data"]["parameterValues"][0]["id"] == "FlyArousal"
    assert injection["data"]["parameterValues"][0]["value"] == 0.5


@pytest.mark.asyncio
async def test_bridge_reconnects_after_vts_drops_the_connection():
    """VTube Studio disconnecting must not kill the bridge -- and must never
    touch the simulation. Start the bridge against a dead port, bring the
    server up late, and confirm it connects on a later retry."""
    connected = asyncio.Event()

    async def late_vts(connection):
        connected.set()
        async for raw in connection:
            message = json.loads(raw)
            kind = message["messageType"]
            if kind == "AuthenticationTokenRequest":
                await connection.send(json.dumps({
                    "messageType": "AuthenticationTokenResponse",
                    "data": {"authenticationToken": "tok"},
                }))
            else:
                await connection.send(json.dumps({
                    "messageType": "AuthenticationResponse",
                    "data": {"authenticated": True},
                }))

    bridge = VTSBridge("ws://127.0.0.1:8802", [], token_path=None)
    task = asyncio.create_task(bridge.run("ws://127.0.0.1:8803", backoff=0.2))

    await asyncio.sleep(0.4)               # bridge is failing and backing off
    assert not connected.is_set()

    server = await websockets.serve(late_vts, "127.0.0.1", 8802)
    try:
        await asyncio.wait_for(connected.wait(), timeout=5.0)
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        server.close()
        await server.wait_closed()


@pytest.mark.asyncio
async def test_bridge_run_survives_a_failing_bus():
    """A bus that never appears must produce retries, not an exception."""
    bridge = VTSBridge("ws://127.0.0.1:8899", [], token_path=None)
    task = asyncio.create_task(bridge.run("ws://127.0.0.1:8898", backoff=0.1))
    await asyncio.sleep(0.5)
    assert not task.done(), "bridge died instead of retrying"
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/bridge/test_vts.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'flybrain.bridge'`

- [ ] **Step 3: Write the mapping configuration**

```yaml
# config/avatar_mapping.yaml
# Signal -> VTube Studio parameter. Tuning how twitchy the character feels is
# editing this file; no engine change, no rebuild.
#
# Custom parameters must be created in VTube Studio (Settings -> plugin) or via
# ParameterCreation before injection will move anything.
mappings:
  - signal: arousal
    parameter: FlyArousal
    scale: 120.0
    smooth_ms: 400.0
    decay: 1.0
    clamp: [0.0, 1.0]

  - signal: startle
    parameter: FlyStartle
    scale: 0.05
    smooth_ms: 40.0
    decay: 0.88          # snaps up, falls away over roughly a second
    clamp: [0.0, 1.0]

  - signal: flow
    component: 0
    parameter: FlyLookX
    scale: 1.0
    smooth_ms: 120.0
    decay: 0.97
    clamp: [-1.0, 1.0]

  - signal: flow
    component: 1
    parameter: FlyLookY
    scale: 1.0
    smooth_ms: 120.0
    decay: 0.97
    clamp: [-1.0, 1.0]
```

- [ ] **Step 4: Write the bridge**

```python
# flybrain/bridge/__init__.py
```

```python
# flybrain/bridge/vts.py
"""Signal bus -> VTube Studio.

All smoothing and easing happens here, never in the core. The core emits the
raw truth that validation recorded; this module shapes it for presentation. If
the two ever disagree, it is provable which is which.
"""
from __future__ import annotations

import asyncio
import json
import math
from dataclasses import dataclass
from pathlib import Path

import websockets
import yaml


@dataclass
class Mapping:
    signal: str
    component: int | None
    parameter: str
    scale: float
    smooth_ms: float
    decay: float
    clamp: tuple[float, float]


def load_mappings(path: Path) -> list[Mapping]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    out = []
    for entry in raw["mappings"]:
        low, high = entry.get("clamp", [-1.0, 1.0])
        out.append(
            Mapping(
                signal=entry["signal"],
                component=entry.get("component"),
                parameter=entry["parameter"],
                scale=float(entry.get("scale", 1.0)),
                smooth_ms=float(entry.get("smooth_ms", 0.0)),
                decay=float(entry.get("decay", 1.0)),
                clamp=(float(low), float(high)),
            )
        )
    return out


class ParameterShaper:
    def __init__(self, mappings: list[Mapping], dt_ms: float = 16.0) -> None:
        self.mappings = mappings
        self.dt_ms = dt_ms
        self._state: dict[str, float] = {}

    def shape(self, payload: dict) -> dict[str, float]:
        out: dict[str, float] = {}
        for m in self.mappings:
            if m.signal not in payload:
                continue
            raw = payload[m.signal]
            if m.component is not None:
                raw = raw[m.component]
            target = float(raw) * m.scale

            previous = self._state.get(m.parameter, 0.0) * m.decay
            if m.smooth_ms > 0:
                alpha = 1.0 - math.exp(-self.dt_ms / m.smooth_ms)
                value = previous + (target - previous) * alpha
            else:
                value = target

            value = max(m.clamp[0], min(m.clamp[1], value))
            self._state[m.parameter] = value
            out[m.parameter] = value
        return out


class VTSBridge:
    def __init__(
        self,
        uri: str = "ws://127.0.0.1:8001",
        mappings: list[Mapping] | None = None,
        plugin_name: str = "FlyBrain",
        developer: str = "flybrain",
        token_path: Path | None = Path(".vts_token"),
    ) -> None:
        self.uri = uri
        self.mappings = mappings or []
        self.plugin_name = plugin_name
        self.developer = developer
        self.token_path = token_path
        self.connection = None

    def _envelope(self, message_type: str, data: dict) -> str:
        return json.dumps({
            "apiName": "VTubeStudioPublicAPI",
            "apiVersion": "1.0",
            "requestID": message_type,
            "messageType": message_type,
            "data": data,
        })

    async def connect(self) -> None:
        self.connection = await websockets.connect(self.uri)

    async def _request(self, message_type: str, data: dict) -> dict:
        await self.connection.send(self._envelope(message_type, data))
        return json.loads(await self.connection.recv())

    async def authenticate(self) -> bool:
        token = None
        if self.token_path and Path(self.token_path).exists():
            token = Path(self.token_path).read_text().strip()

        if not token:
            reply = await self._request(
                "AuthenticationTokenRequest",
                {"pluginName": self.plugin_name, "pluginDeveloper": self.developer},
            )
            token = reply.get("data", {}).get("authenticationToken")
            if token and self.token_path:
                Path(self.token_path).write_text(token)

        reply = await self._request(
            "AuthenticationRequest",
            {
                "pluginName": self.plugin_name,
                "pluginDeveloper": self.developer,
                "authenticationToken": token,
            },
        )
        return bool(reply.get("data", {}).get("authenticated"))

    async def send(self, values: dict[str, float]) -> None:
        if not values:
            return
        await self._request(
            "InjectParameterDataRequest",
            {
                "faceFound": False,
                "mode": "set",
                "parameterValues": [
                    {"id": name, "value": float(v)} for name, v in values.items()
                ],
            },
        )

    async def close(self) -> None:
        if self.connection is not None:
            await self.connection.close()
            self.connection = None

    async def run(self, bus_uri: str = "ws://127.0.0.1:8765", backoff: float = 1.0) -> None:
        """Subscribe to the bus and drive the avatar, reconnecting on failure.
        The simulation is unaffected by anything that happens here."""
        shaper = ParameterShaper(self.mappings)
        while True:
            try:
                await self.connect()
                await self.authenticate()
                async with websockets.connect(bus_uri) as bus:
                    async for raw in bus:
                        await self.send(shaper.shape(json.loads(raw)))
                backoff = 1.0
            except Exception as exc:  # noqa: BLE001 -- the bridge must never die
                print(f"[vts] {type(exc).__name__}: {exc}; retrying in {backoff:.0f}s")
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30.0)
            finally:
                await self.close()
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/bridge/test_vts.py -v`
Expected: PASS (11 tests)

- [ ] **Step 6: MANUAL VERIFICATION — the avatar reacts to you**

1. Install **VTube Studio** (free on Steam) and launch it with any bundled model (Hiyori, Mao, Mark or Akari).
2. In VTube Studio: **Settings → enable the plugin API on port 8001**.
3. Create the four custom parameters — `FlyArousal`, `FlyStartle`, `FlyLookX`, `FlyLookY` — in the model's parameter settings, and bind each to something visible (eye open, head angle, brow).
4. Start the live loop from Task 16 in one terminal.
5. In a second terminal:

```bash
python -c "
import asyncio
from pathlib import Path
from flybrain.bridge.vts import VTSBridge, load_mappings
b = VTSBridge('ws://127.0.0.1:8001', load_mappings(Path('config/avatar_mapping.yaml')))
asyncio.run(b.run('ws://127.0.0.1:8765'))
"
```

6. Accept the plugin permission prompt inside VTube Studio the first time.
7. **Wave your hand quickly across the webcam.**

The avatar must react — a startle twitch on sudden motion, gaze drifting with the direction of movement, and a slow arousal change that persists after you stop moving. **That persistence is the tell that the signal is coming from the connectome's own dynamics and not from a frame-by-frame image filter.**

To confirm the mapping layer is doing only presentation, edit `config/avatar_mapping.yaml` — raise `startle`'s `scale`, lower its `decay` — and restart just the bridge. The character's feel changes; the engine never restarts.

- [ ] **Step 7: Commit**

```bash
git add config/avatar_mapping.yaml flybrain/bridge/ tests/bridge/
git commit -m "feat: VTube Studio bridge with YAML-configured signal mapping"
```
