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
    soma_side: np.ndarray
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

    if verify:
        if not manifest_path.exists():
            raise GraphHashMismatch(
                f"manifest file missing for {path} -- verification required but "
                f"no manifest found at {manifest_path}"
            )
        if "content_hash" not in meta:
            raise GraphHashMismatch(
                f"content_hash missing from manifest {manifest_path} -- "
                f"verification required but hash key not found"
            )
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
        soma_side=data["soma_side"],
        meta=meta,
    )
