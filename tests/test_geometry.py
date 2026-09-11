import numpy as np

from geometry import compute_distribution_distance


def test_sliced_wasserstein_identity_and_symmetry():
    rng = np.random.default_rng(4)
    a = rng.normal(size=(32, 8))
    b = rng.normal(loc=0.5, size=(32, 8))
    assert compute_distribution_distance(a, a, num_projections=32) == 0.0
    ab = compute_distribution_distance(a, b, num_projections=32, seed=7)
    ba = compute_distribution_distance(b, a, num_projections=32, seed=7)
    assert ab > 0
    assert np.isclose(ab, ba)


def test_control_metrics_are_available():
    a = np.zeros((8, 3))
    b = np.ones((8, 3))
    for method in ["euclidean_mean", "cosine", "gaussian_w2", "mmd"]:
        assert np.isfinite(compute_distribution_distance(a, b, method=method))
