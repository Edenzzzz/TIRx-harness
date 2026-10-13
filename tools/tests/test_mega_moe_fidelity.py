"""The faithful Mega MoE gate excludes repaired, stale, or incomplete runs."""

import pytest

from benchmarks.peer.mega_moe import validate_rows


def row(**changes):
    return {"shape": "shape", "source": "current", "comparison_kind": "port_fidelity",
            "ordering": "origin", "max_abs_diff": 0, "stats_max_abs_diff": 0,
            "ratio": 1.0, **changes}


@pytest.mark.parametrize("changes", (
    {"source": "stale"}, {"ordering": "sys_fenced"}, {"comparison_kind": "historical"},
    {"max_abs_diff": 1}, {"stats_max_abs_diff": 1}, {"ratio": 1.02},
    {"ratio": float("nan")}, {"ratio": 0},
))
def test_ineligible_measurements_do_not_establish_parity(changes):
    with pytest.raises(RuntimeError, match="Mega MoE fidelity validation failed"):
        validate_rows([row(**changes)], ["shape"], "current")


def test_every_requested_shape_requires_current_evidence():
    validate_rows([row()], ["shape"], "current")
    with pytest.raises(RuntimeError, match="missing"):
        validate_rows([row()], ["shape", "missing"], "current")
