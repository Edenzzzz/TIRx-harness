"""Keep baseline instructions and work assignment distinct from semantic repairs."""

import ast
import inspect

import pytest

from ported.deepgemm._sm100_fp8_fp4_mega_moe.kernel import get_kernel
from tests.numsim.support.peer_mega_moe import Shape, kernel


@pytest.fixture(scope="module", params=[(world, mode) for world in (1, 4)
                                       for mode in ("origin", "sys_fenced")])
def emitted(request):
    world, mode = request.param
    shape = Shape(world=world, num_tokens=(4,) * world, hidden=7168, intermediate_hidden=2048)
    return mode, ast.parse(kernel(shape, ordering=mode).script())


def calls(tree, name):
    return [node for node in ast.walk(tree) if isinstance(node, ast.Call)
            and ast.unparse(node.func) == name]


def modifiers(call):
    return tuple(argument.value for argument in call.args
                 if isinstance(argument, ast.Constant) and isinstance(argument.value, str))


def test_default_port_retains_timeouts_and_reference_memory_forms(emitted):
    mode, tree = emitted
    assert inspect.signature(get_kernel).parameters["ordering"].default == "origin"
    assert bool(calls(tree, "T.cuda.clock64")) == (mode == "origin")
    assert bool(calls(tree, "T.cuda.printf")) == (mode == "origin")
    assert any("nc" in modifiers(call) for call in calls(tree, "T.ptx.ld"))
    if mode == "origin":
        assert any("volatile" in modifiers(call) for call in calls(tree, "T.ptx.ld"))
        assert not calls(tree, "T.cuda.wait_until")
    global_fences = [call for call in calls(tree, "T.ptx.fence")
                     if modifiers(call) == ("proxy", "async", "global", "")]
    assert bool(global_fences) == (mode == "sys_fenced")


def test_combine_keeps_chunks_serial_within_each_tokens_warp(emitted):
    _, tree = emitted
    additions = [call for call in calls(tree, "T.ptx.add") if "bf16" in modifiers(call)]
    assert len(additions) == 2
    enclosing = [node for node in ast.walk(tree) if isinstance(node, ast.For)
                 and all(call in list(ast.walk(node)) for call in additions)]
    assert any(ast.unparse(loop.iter) == "range(2)" for loop in enclosing)


def test_combine_load_sync_is_guarded_by_the_nonempty_mask(emitted):
    _, tree = emitted
    additions = [call for call in calls(tree, "T.ptx.add") if "bf16" in modifiers(call)]
    chunk_loop = next(node for node in ast.walk(tree) if isinstance(node, ast.For)
                      and ast.unparse(node.iter) == "range(2)"
                      and all(call in list(ast.walk(node)) for call in additions))
    copies = calls(chunk_loop, "T.ptx.cp")
    loads = [call for call in copies if "mbarrier::complete_tx::bytes" in modifiers(call)]
    assert len(loads) == 2
    for load in loads:
        guards = [node for node in ast.walk(chunk_loop) if isinstance(node, ast.If)
                  and load in list(ast.walk(node))
                  and not calls(node.test, "T.cuda.elect_sync")]
        assert len(guards) == 1
        assert calls(guards[0], "T.cuda.warp_sync")


def test_unknown_ordering_is_rejected():
    with pytest.raises(ValueError, match="unsupported Mega MoE ordering"):
        kernel(ordering="unknown")
