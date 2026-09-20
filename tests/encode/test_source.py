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
