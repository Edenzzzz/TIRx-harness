# CUDA loop and execution-coordinate lowering

## Finding

Two generic CUDA lowering decisions increase generated code: serial loops
without annotations allow automatic CUDA unrolling, and execution-coordinate
bindings substitute builtin expressions at every use. CUDA/NVRTC 13.2 duplicates
the large two-iteration combine loop, partially unrolls dynamic loops, and
repeatedly reads execution coordinates in the faithful Mega MoE port.

The compiler code is in `src/backend/cuda/codegen/codegen_cuda.cc`:
`CodeGenCUDA::Dispatch_(const tirx::ForNode*)` and
`CodeGenCUDA::BindThreadIndex`. The [complete patch and regression tests](tvm.patch)
target TVM commit
`ea0cfa320fcdea0567e2f4c19d363625c66630d1`.

The proposed generic lowering preserves explicit unroll hints, leaves small
static serial loops to the CUDA compiler, and emits `#pragma unroll 1` for an
otherwise unannotated serial loop with a dynamic extent or nested `while`.
This matches the existing treatment of `WhileNode` and prevents duplicating
dynamic loop bodies. It does not inspect kernel names, dimensions, or hardware
models.

The loop emitter adds this fallback after processing explicit unroll hints:

```cpp
} else if (op->kind == tirx::ForKind::kSerial &&
           (!op->extent.as<IntImmNode>() ||
            tirx::ContainsNode<tirx::WhileNode>(op->body))) {
  PrintIndent();
  stream << "#pragma unroll 1\n";
}
```

The original coordinate binding expands every use to the builtin:

```cpp
var_idmap_[iv->var.get()] =
    CastFromTo(iv->thread_tag, PrimType::UInt(32), iv->var.ty());
```

The patched binding emits one local value from the same native special-register
read and retains its launch-derived range. For example, the 512-thread launch
emits:

```cuda
uint32_t __tirx_cached_tid_x;
asm volatile("mov.u32 %0, %%tid.x;" : "=r"(__tirx_cached_tid_x));
__builtin_assume(__tirx_cached_tid_x < 512);
```

Bounds come from the corresponding static `thread_extent`, including partial
CTAs. Dynamic extents receive no assumption. Cluster coordinates retain their
existing coordinate-versus-linear-rank mapping.

## Evidence

The four-rank `dsv3_t1` case uses one token per rank, hidden size 7168,
intermediate size 2048, 256 routed experts, top-8, and one shared expert.
All captures use the same port sources, CUDA/NVRTC 13.2.78, and matched
reference launch configuration. Counts include predicated instructions.

| Capture | SASS instructions | BF16 accumulations | Registers | Static shared bytes |
|---|---:|---:|---:|---:|
| DeepGEMM reference | 5024 | 112 | 128 | 1024 |
| Current TIRx lowering | 5944 | 224 | 128 | 1024 |
| Generic serial-loop lowering prototype | 4264 | 112 | 128 | 1024 |
| Loop and coordinate lowering | 4104 | 112 | 128 | 1024 |

The current port contains one combine body in generated CUDA; its two copies
appear during CUDA compilation. Suppressing unrolling only on serial loops
containing `while` reduces the count to 5400. Suppressing it only on dynamic
serial loops reduces the count to 4808. Applying both gives 4264.

An IR-level prototype that adds the existing `disable_unroll` annotation under
the proposed conditions produces cubins byte-identical to the emitted-CUDA
prototype, with and without statistics. The rebuilt C++ compiler produces
byte-identical cubins to the loop-and-coordinate prototype as well. These are
compiler changes; no kernel-specific instruction selection or schedule is added.

Two independent, 11-round, four-GPU-lease benchmark invocations compare each
variant with DeepGEMM, alternating kernel order inside each invocation:

| Lowering | TIRx microseconds | Reference microseconds | TIRx/reference | Output/statistics |
|---|---:|---:|---:|---|
| Current | 100.457 | 97.072 | 1.034871 | Bitwise equal |
| Prototype | 99.711 | 97.141 | 1.026456 | Bitwise equal |

The loop-only prototype improves latency but still fails the 99% gate. After
restoring the reference's mask-guarded combine-load synchronization, the
loop-and-coordinate prototype passes two independent 11-round comparisons:
98.138/97.452 microseconds (1.007039x) and 98.917/98.713 microseconds (1.002067x),
both with bitwise-equal output and statistics. The rebuilt compiler's complete
first 14-shape matrix passes, with a worst ratio of 1.0092574x (`dsv3_t1`). The
independent one-token repeat is 98.196/97.157 microseconds (1.0106940x), just
outside the 99% gate, so full parity remains incomplete. Further lowering
investigation and the other distributed kernels' matrices are in progress.
Reduced static instruction count alone does not establish any unmeasured gate.

## Regression tests

The patch adds code-generation checks for a dynamic serial loop, a static
serial loop containing a dynamic inner loop, explicit unroll policies, and a
small static loop whose existing compiler choice must remain intact. Coordinate
tests cover 17-, 32-, and 64-thread launches and cluster-coordinate mapping.

On the unpatched installed compiler, the two default-policy regressions fail
and the four explicit-policy/static-loop controls pass. On the rebuilt patched
compiler, the complete non-GPU `test_codegen_cuda.py` suite reports 55 passed
and one skipped. The focused command is:

```bash
python -m pytest -q -n 2 tests/python/tirx/codegen/test_codegen_cuda.py \
  -k 'serial_dynamic_loop_preserves_explicit_unroll_policy or serial_loop_with_dynamic_inner_loop or small_static_serial_loop or serial_pragma_unroll or serial_disable_unroll or native_execution_coordinates'
```

## Reproduction

Apply the patch in a separate TVM checkout at the revision above, initialize its
submodules, and build using TVM's normal build instructions. Use that build's
matching Python sources and libraries for validation, for example:

```bash
git -C /path/to/tvm apply /path/to/TIRx-harness/tools/docs/numsim/lowering/tvm.patch
cmake --build /path/to/tvm/build --parallel 8
export TVM_LIBRARY_PATH=/path/to/tvm/build/lib
export PYTHONPATH=/path/to/tvm/python:/path/to/TIRx-harness/tools
```

This repository does not replace the installed compiler through a runtime
callback. The patch must be built and selected explicitly until available in a
compatible compiler release.

Use `tools/benchmarks/peer/mega_moe.py` to prepare and compare `dsv3_t1` against
the pinned DeepGEMM reference. Keep separate preparation caches for the original
and patched compiler: the benchmark's source stamp does not include compiler
binary identity. Acquire the node's shared GPU lease before running comparisons.

```bash
PYTHONPATH=tools python -m benchmarks.peer.mega_moe prepare \
  --shapes dsv3_t1 --jobs 1 --cache /path/to/compiler-specific-cache
PYTHONPATH=tools python -m benchmarks.peer.mega_moe compare \
  --shapes dsv3_t1 --rounds 11 --cache /path/to/compiler-specific-cache \
  --out /path/outside-the-repository/results.jsonl
```

Capture each compiled module's CUDA source and native cubin, then use
`cuobjdump --dump-sass --dump-resource-usage` on those cubins. Count every
instruction-offset line, including lines with predicate prefixes. Do not count
only unpredicated opcodes or infer instructions from disassembler line totals.
