import numpy as np

from rq2_pairwise_surrogate_regret import PAIR_INDICES
from rq2_uniform_win_diagnostics import (
    _momentum_metrics_from_gram,
    _safe_corr,
)
from scripts.run_uniform_win_diagnostics import MOMENTUM_STATES


def test_momentum_two_step_summary_is_finite():
    rng = np.random.default_rng(41)
    gradients = rng.normal(size=(16, 29))
    momentum = rng.normal(size=29)
    parameters = rng.normal(size=29)
    probabilities = np.full(len(PAIR_INDICES), 1.0 / len(PAIR_INDICES))
    width, pair, policy = _momentum_metrics_from_gram(
        gradients @ gradients.T,
        gradients @ momentum,
        gradients @ parameters,
        float(momentum @ momentum),
        float(momentum @ parameters),
        float(parameters @ parameters),
        0.9,
        5e-4,
        {"uniform": probabilities},
    )
    assert len(width) == 16
    assert len(pair) == len(PAIR_INDICES)
    assert len(policy) == 1
    assert np.isfinite(policy.select_dtypes(include="number").to_numpy()).all()
    assert policy.expected_two_update_momentum_stability.between(-1, 1).all()


def test_safe_correlation_rejects_constant_signal():
    assert np.isnan(_safe_corr(np.ones(5), np.arange(5), "pearson"))
    assert np.isclose(_safe_corr(np.arange(5), np.arange(5), "spearman"), 1.0)


def test_zero_momentum_is_reported_as_undefined_not_zero_alignment():
    rng = np.random.default_rng(9)
    gradients = rng.normal(size=(16, 17))
    parameters = rng.normal(size=17)
    probabilities = np.full(len(PAIR_INDICES), 1.0 / len(PAIR_INDICES))
    _, _, policy = _momentum_metrics_from_gram(
        gradients @ gradients.T,
        np.zeros(16),
        gradients @ parameters,
        0.0,
        0.0,
        float(parameters @ parameters),
        0.9,
        5e-4,
        {"uniform": probabilities},
    )
    assert np.isnan(policy.expected_cos_momentum_pair_update.iloc[0])
    assert np.isnan(policy.expected_cos_momentum_after_one_update.iloc[0])
    assert np.isfinite(policy.expected_two_update_momentum_stability.iloc[0])


def test_momentum_states_cover_common_and_four_branches():
    assert MOMENTUM_STATES[0] == ("common_warmup", 10)
    assert len(MOMENTUM_STATES) == 9
    assert set(method for method, _ in MOMENTUM_STATES[1:]) == {
        "uniform", "resource", "pure_sw", "resource_geo"
    }
    assert set(epoch for _, epoch in MOMENTUM_STATES[1:]) == {50, 100}
