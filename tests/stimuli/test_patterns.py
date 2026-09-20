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
    """Compares a SEQUENCE, not one frame. A sinusoidal grating at a single
    instant is a static pattern carrying no direction -- opposite directions
    are genuinely identical at certain phases (frames 3 and 9 at these
    defaults), so a single-frame assertion tests the phase convention rather
    than the stimulus."""
    a = drifting_grating(H, W, 0.0)
    b = drifting_grating(H, W, 180.0)
    seq_a = np.stack([a(i) for i in range(6)])
    seq_b = np.stack([b(i) for i in range(6)])
    assert not np.allclose(seq_a, seq_b)


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
