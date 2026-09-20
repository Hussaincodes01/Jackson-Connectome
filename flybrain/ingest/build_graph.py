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
        columns=["bodyId", "type", "superclass", "assignedOlHex1", "assignedOlHex2", "somaSide"],
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
    soma_side = np.array(
        [s if s is not None else "" for s in tbl.column("somaSide").to_pylist()], dtype=object
    )[keep]
    order = np.argsort(body_ids)
    return {
        "body_ids": body_ids[order],
        "types": types[order],
        "superclasses": superclass[keep][order],
        "hex1": hex1[order],
        "hex2": hex2[order],
        "soma_side": soma_side[order],
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
        soma_side=np.array(
            [s if s is not None else "" for s in ann["soma_side"]], dtype="<U4"
        ),
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
