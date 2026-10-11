"""Keep numerical and performance evidence scoped to the kernel being measured."""

from __future__ import annotations

import pytest

from benchmarks.multimem.gemm_all_reduce import summarize, validate_summary


def _measurement(**updates):
    return {
        "shape": "cutlass_default",
        "origin": "cutlass",
        "config": "{}",
        "origin_us": 100.0,
        "tirx_us": 100.0,
        "ratio": 1.0,
        "bitwise_equal": True,
        **updates,
    }


def test_historical_origin_ordering_cannot_validate_the_system_ordered_kernel():
    rows = [_measurement(), _measurement(tirx_ordering="sys_fenced", tirx_us=110.0, ratio=1.1)]
    summary = summarize(rows)

    assert summary[0]["ratio"] == 1.1
    with pytest.raises(RuntimeError, match="99% parity"):
        validate_summary(summary, ["cutlass_default"], ["cutlass"])
    assert summarize(rows, ordering="origin")[0]["ratio"] == 1.0
    assert summarize([_measurement()]) == []


def test_missing_shape_or_origin_is_not_a_successful_comparison():
    summary = summarize([_measurement(tirx_ordering="sys_fenced")])

    with pytest.raises(RuntimeError, match="missing cutlass_default/flashinfer"):
        validate_summary(summary, ["cutlass_default"], ["cutlass", "flashinfer"])
    with pytest.raises(RuntimeError, match="missing flashinfer_test/cutlass"):
        validate_summary(summary, ["cutlass_default", "flashinfer_test"], ["cutlass"])


def test_wrong_output_fails_even_when_timing_beats_the_origin():
    summary = summarize(
        [_measurement(tirx_ordering="sys_fenced", bitwise_equal=False, tirx_us=90.0, ratio=0.9)]
    )

    with pytest.raises(RuntimeError, match="numerical comparison failed"):
        validate_summary(summary, ["cutlass_default"], ["cutlass"])


@pytest.mark.parametrize("ratio", [0.0, float("inf"), float("nan"), 1 / 0.989])
def test_invalid_or_insufficient_performance_fails(ratio):
    summary = summarize(
        [_measurement(tirx_ordering="sys_fenced", tirx_us=100 * ratio, ratio=ratio)]
    )

    with pytest.raises(RuntimeError, match="99% parity"):
        validate_summary(summary, ["cutlass_default"], ["cutlass"])


def test_exactly_99_percent_performance_meets_the_gate():
    summary = summarize(
        [_measurement(tirx_ordering="sys_fenced", tirx_us=100 / 0.99, ratio=1 / 0.99)]
    )

    validate_summary(summary, ["cutlass_default"], ["cutlass"])


@pytest.mark.parametrize("shapes,origins", [([], ["cutlass"]), (["cutlass_default"], [])])
def test_empty_comparison_is_not_success(shapes, origins):
    with pytest.raises(RuntimeError, match="requires shapes and origins"):
        validate_summary([], shapes, origins)
