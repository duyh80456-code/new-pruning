from __future__ import annotations

import numpy as np
import torch
from scipy import linalg
from scipy.stats import wasserstein_distance


def _as_numpy(features) -> np.ndarray:
    if isinstance(features, torch.Tensor):
        features = features.detach().cpu().numpy()
    result = np.asarray(features, dtype=np.float64)
    if result.ndim != 2:
        raise ValueError(f"Expected [samples, dimensions], got {result.shape}")
    return result


def _sliced_wasserstein(a: np.ndarray, b: np.ndarray, num_projections: int, seed: int) -> float:
    rng = np.random.default_rng(seed)
    directions = rng.normal(size=(num_projections, a.shape[1]))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True).clip(min=1e-12)
    projected_a = a @ directions.T
    projected_b = b @ directions.T
    # SciPy provides the reliable exact 1-D Wasserstein primitive; this code only
    # constructs the standard sliced estimator around it.
    values = [
        wasserstein_distance(projected_a[:, i], projected_b[:, i])
        for i in range(num_projections)
    ]
    return float(np.mean(values))


def _gaussian_w2(a: np.ndarray, b: np.ndarray) -> float:
    mean_term = np.sum((a.mean(0) - b.mean(0)) ** 2)
    cov_a, cov_b = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
    middle = linalg.sqrtm(linalg.sqrtm(cov_a) @ cov_b @ linalg.sqrtm(cov_a))
    cov_term = np.trace(cov_a + cov_b - 2 * middle.real)
    return float(np.sqrt(max(mean_term + cov_term, 0.0)))


def _mmd_rbf(a: np.ndarray, b: np.ndarray) -> float:
    joined = np.concatenate([a, b], axis=0)
    distances = np.sum((joined[:, None] - joined[None, :]) ** 2, axis=-1)
    positive = distances[distances > 0]
    bandwidth = float(np.median(positive)) if positive.size else 1.0
    kernel = np.exp(-distances / max(2 * bandwidth, 1e-12))
    n = len(a)
    value = kernel[:n, :n].mean() + kernel[n:, n:].mean() - 2 * kernel[:n, n:].mean()
    return float(np.sqrt(max(value, 0.0)))


def compute_distribution_distance(
    features_a,
    features_b,
    method: str = "sliced_wasserstein",
    num_projections: int = 256,
    seed: int = 0,
) -> float:
    """Compute a distribution distance through a stable pluggable interface."""
    a, b = _as_numpy(features_a), _as_numpy(features_b)
    if a.shape[1] != b.shape[1]:
        raise ValueError(f"Feature dimensions differ: {a.shape[1]} != {b.shape[1]}")
    if method == "sliced_wasserstein":
        return _sliced_wasserstein(a, b, num_projections, seed)
    if method == "euclidean_mean":
        return float(np.linalg.norm(a.mean(0) - b.mean(0)))
    if method == "cosine":
        ma, mb = a.mean(0), b.mean(0)
        return float(1 - np.dot(ma, mb) / max(np.linalg.norm(ma) * np.linalg.norm(mb), 1e-12))
    if method == "gaussian_w2":
        return _gaussian_w2(a, b)
    if method == "mmd":
        return _mmd_rbf(a, b)
    if method == "sinkhorn":
        try:
            import ot
        except ImportError as exc:
            raise RuntimeError("Sinkhorn requires the optional POT package: pip install POT") from exc
        cost = ot.dist(a, b)
        return float(ot.sinkhorn2(np.ones(len(a)) / len(a), np.ones(len(b)) / len(b), cost, reg=0.05))
    raise ValueError(f"Unknown distance method: {method}")
