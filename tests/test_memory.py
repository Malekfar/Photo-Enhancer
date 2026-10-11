import math

from app.service import available_memory_bytes, memory_fit_scale


def test_fits_without_scaling():
    assert memory_fit_scale(1_000_000, 100, 0.8, 1_000_000_000) == 1.0
    assert memory_fit_scale(10**9, 100, 0.8, None) == 1.0


def test_scales_to_budget():
    s = memory_fit_scale(12_000_000, 110, 0.8, 1_000_000_000)
    assert s < 1.0
    assert math.isclose(12_000_000 * s * s * 110, 0.8e9, rel_tol=1e-6)


def test_available_memory_is_sane():
    m = available_memory_bytes()
    assert m is None or m >= 0
