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
