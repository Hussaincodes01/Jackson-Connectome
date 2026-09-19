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
