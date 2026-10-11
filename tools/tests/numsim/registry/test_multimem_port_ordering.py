"""Check the memory-ordering contract of the public full-kernel builder."""

from __future__ import annotations

import ast

import pytest

from ported.cutlass.sm100_gemm_all_reduce import gemm_all_reduce


def _kernel(protocol, use_tma_store, use_2cta, rank, **options):
    return gemm_all_reduce(
        512,
        128,
        128,
        "bf16",
        "f32",
        rank,
        4,
        use_2cta=use_2cta,
        mma_tiler=(256, 128) if use_2cta else (128, 64),
        cluster=(2, 1) if use_2cta else (1, 1),
        use_tma_store=use_tma_store,
        protocol=protocol,
        max_active_clusters=1,
        flag_len=16,
        **options,
    ).func


def _modifiers(function, instruction):
    tree = ast.parse(function.script())
    return [
        tuple(
            argument.value
            for argument in node.args
            if isinstance(argument, ast.Constant) and isinstance(argument.value, str)
        )
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and ast.unparse(node.func) == f"T.ptx.{instruction}"
    ]


@pytest.mark.parametrize("protocol", ("cutlass", "flashinfer"))
@pytest.mark.parametrize("use_tma_store", (True, False))
@pytest.mark.parametrize("use_2cta", (True, False))
@pytest.mark.parametrize("rank", (None, 0))
def test_full_kernel_defaults_to_system_publication(protocol, use_tma_store, use_2cta, rank):
    function = _kernel(protocol, use_tma_store, use_2cta, rank)
    arrivals = _modifiers(function, "multimem_red")
    polls = _modifiers(function, "atom")

    assert arrivals and all(modifiers[:2] == ("release", "sys") for modifiers in arrivals)
    assert polls and all(modifiers[:2] == ("acquire", "sys") for modifiers in polls)
    assert ("proxy", "alias", "", "") in _modifiers(function, "fence")
    parameters = {parameter.name for parameter in function.params}
    assert "out" in parameters
    assert ("rank" in parameters) == (rank is None)


@pytest.mark.parametrize("protocol", ("cutlass", "flashinfer"))
def test_origin_ordering_requires_explicit_opt_in(protocol):
    function = _kernel(protocol, True, False, None, ordering="origin")

    assert any(modifiers[1] == "gpu" for modifiers in _modifiers(function, "multimem_red"))
    assert any(modifiers[1] == "gpu" for modifiers in _modifiers(function, "atom"))


def test_unknown_ordering_is_rejected():
    with pytest.raises(ValueError, match="unsupported all-reduce ordering"):
        _kernel("cutlass", True, False, None, ordering="unknown")
