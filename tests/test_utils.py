"""Tests for the pure math helpers and device selection in positronic_brain.utils.

These helpers are exported from the package root and used across the model,
trainers and visualizer, but previously had no dedicated coverage.
"""

import numpy as np
import pytest
import torch

from positronic_brain.utils import (
    compute_distance,
    get_device,
    grid_to_index,
    index_to_grid,
    moving_average,
    normalize,
    sigmoid,
    zone_onehot,
)


def test_sigmoid_midpoint_and_monotonicity():
    assert sigmoid(0.0) == pytest.approx(0.5)
    assert 0.0 < sigmoid(-10.0) < sigmoid(10.0) < 1.0


def test_sigmoid_is_stable_at_extreme_inputs():
    # The clip at ±60 must prevent overflow for arbitrarily large inputs.
    x = np.array([-1e9, -100.0, 0.0, 100.0, 1e9])
    y = sigmoid(x)
    assert np.all(np.isfinite(y))
    assert y[0] == pytest.approx(0.0)
    assert y[-1] == pytest.approx(1.0)


def test_normalize_maps_to_unit_range():
    out = normalize(np.array([2.0, 4.0, 6.0]))
    assert out.min() == pytest.approx(0.0)
    assert out.max() == pytest.approx(1.0, abs=1e-6)
    assert np.all((out >= 0.0) & (out <= 1.0))


def test_normalize_constant_array_returns_zeros():
    # A flat array has no dynamic range; dividing by ~0 must not blow up.
    assert np.array_equal(normalize(np.full(5, 3.14)), np.zeros(5))


def test_grid_index_round_trip_covers_whole_lattice():
    G = 5
    for idx in range(G**3):
        x, y, z = index_to_grid(idx, G)
        assert 0 <= x < G and 0 <= y < G and 0 <= z < G
        assert grid_to_index(x, y, z, G) == idx


def test_compute_distance_known_value_and_mixed_inputs():
    assert compute_distance((0, 0, 0), (3, 4, 0)) == pytest.approx(5.0)
    assert compute_distance(np.array([1.0, 1.0, 1.0]), (1.0, 1.0, 1.0)) == pytest.approx(0.0)


def test_zone_onehot_shape_and_rows():
    zones = np.array([0, 2, 1, 2])
    oh = zone_onehot(zones, num_zones=3)
    assert oh.shape == (4, 3)
    assert np.array_equal(oh.sum(axis=1), np.ones(4, dtype=np.float32))
    assert np.array_equal(oh.argmax(axis=1), zones)


def test_moving_average_preserves_length_and_constants():
    x = np.arange(10, dtype=np.float64)
    assert moving_average(x, window=3).shape == x.shape
    const = np.full(8, 2.5)
    assert np.allclose(moving_average(const, window=5), const)


def test_moving_average_window_one_is_identity_copy():
    x = np.array([1.0, 2.0, 3.0])
    out = moving_average(x, window=1)
    assert np.array_equal(out, x)
    assert out is not x


def test_get_device_cpu_is_always_honoured():
    assert get_device("cpu") == torch.device("cpu")


def test_get_device_auto_returns_available_backend():
    assert get_device("auto").type in ("cpu", "mps", "cuda")


def test_get_device_falls_back_to_cpu_when_backend_missing():
    if not torch.cuda.is_available():
        assert get_device("cuda") == torch.device("cpu")
    if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        assert get_device("mps") == torch.device("cpu")
