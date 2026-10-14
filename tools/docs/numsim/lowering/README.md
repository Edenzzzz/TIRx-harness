# CUDA lowering parity

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

The further copy simplification lives in
`src/tirx/transform/stmt_simplify.cc`,
`StmtSimplifier::Mutate_(const EvaluateNode*)`. It converts an unpredicated
`tirx.ptx.mov` between identical scalar register types into an ordinary local
`BufferStore`, allowing normal value simplification and CUDA optimization.
Predicated copies, predicate-register moves, mixed-type bit reinterpretations,
and memory destinations retain their instruction forms. The PTX table and its
one-instruction renderer are unchanged.

### Parameter-table lowering

The reference's `layout/sym_buffer.cuh::SymBuffer::map` reads a rank offset
from its by-value kernel-parameter array. The port previously staged the
offsets into an extra shared-memory cache. That staging is not present in
DeepGEMM and is now removed: the port selects its existing scalar offset
parameters directly, without changing the work assignment or synchronization.

`ScalarParameterGroups` in `src/target/build_common.h` recognizes four-value
selection tables over contiguous homogeneous scalar parameters. CUDA codegen
packs those parameters into a `const __grid_constant__` array struct and emits
an indexed parameter read. It preserves the original selection's fallback to
element zero for negative or out-of-range indices. Other table shapes retain
their existing lowering. Selectors must be safe to repeat; a global load
materialized into a local value is read only once.

`src/runtime/pack_args.h::PackFuncVoidAddr` repacks the four converted scalar
values for the internal device ABI. Metadata tags preserve that packing across
library export/reload. The public scalar-call signature, 72 rank-offset
arguments, and NumSim multi-rank bindings do not change. Compiler and runtime
must both come from the patched build; this introduces no newer instruction
or toolkit requirement than the existing CUDA 13.2 comparison.

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
| Loop, coordinate, and scalar-copy lowering | 4096 | 112 | 128 | 1024 |
| With parameter-table lowering and reference parameter reads | 4080 | 112 | 128 | 1024 |

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

Adding transparent scalar copies passes two prototype comparisons at 1.0059873x
and 1.0042761x, both numerically exact. The corresponding rebuilt IR simplifier
produces 4096 instructions as well; eight independent constant-register moves
have a different assembly order, so those timings are not substituted for the
rebuilt compiler's own matrix. That matrix is running from an immutable compiler
snapshot and a separate prepared-kernel cache.

The new matrix's first tiny `port_t2_h1024_e8_k1` comparison is 1.041x, despite
bitwise-equal outputs and statistics; its independent repeat is 0.972x. A
separate same-session diagnostic compares both immutable compiler versions and
DeepGEMM on the same inputs, alternating all three implementations over 11
rounds. It retains the benchmark's cold-cache flush and inter-implementation
collective. Taking the median of each round's slowest rank gives:

| Shape | Loop/coordinate microseconds | With scalar copies microseconds | DeepGEMM microseconds | Ratios to DeepGEMM |
|---|---:|---:|---:|---|
| `port_t2_h1024_e8_k1` | 38.50697 | 38.49857 | 39.18763 | 0.982631 / 0.982416 |
| `dsv3_t1` | 99.60723 | 99.93063 | 99.29060 | 1.003189 / 1.006446 |

Both versions meet the gate in this diagnostic. This does not erase the earlier
failed comparisons or replace the independent full-matrix gate; the scalar-copy
patch remains provisional while that validation finishes.

An audit of the other benchmark kernels covers 456 GEMM/all-reduce rank/config
functions, 274 all-gather functions (including auxiliary kernels), and 72
two-shot rank functions. None contains `tirx.ptx.mov`; the copy simplification
does not match them. Their loop/coordinate compiler-regression matrices continue
against the frozen preceding build.

The same 802-function audit finds no four-parameter table candidates in those
three other benchmarks. The new parameter-based Mega MoE port passes the
four-rank numerical/race/sync examples and its first `dsv3_t1` GPU comparison:
98.726/98.530 microseconds (1.001989x), with bitwise-equal output/statistics.
All 14 shapes are prepared under a separate immutable compiler snapshot; the
complete matrix and independent repeat are running. This first result is not
a full-parity claim.

## Regression tests

The patch adds code-generation checks for a dynamic serial loop, a static
serial loop containing a dynamic inner loop, explicit unroll policies, and a
small static loop whose existing compiler choice must remain intact. Coordinate
tests cover 17-, 32-, and 64-thread launches and cluster-coordinate mapping.

On the unpatched installed compiler, the two default-policy regressions fail
and the four explicit-policy/static-loop controls pass. On the rebuilt patched
compiler, the complete non-GPU `test_codegen_cuda.py` suite reports 75 passed
and one skipped. Both PTX dispatch/helper-golden regressions pass. Two GPU
round trips preserve all bits of 32-/64-bit NaNs, signed zeros, infinities, and
subnormals; five same-kernel partial-CTA GPU cases and all 59 simplifier tests
also pass. Invalid width/type modifiers remain visible for normal rejection.
This includes direct-codegen controls for contiguous, noncontiguous, and
reordered device parameters.
Seven GPU tests pass: the two special-bit copy cases, four
parameter-table scalar types, and mixed-type multiple parameter groups. The
table cases exercise both direct launches and exported/reloaded libraries,
signed out-of-range indices, unrelated scalar arguments, and direct uses of a
grouped parameter outside its selection table.
With this compiler selected, the harness's complete multimem/peer numerical,
ordering-rule, and NumSim registry/ABI run passes 943 tests; all 94 live
multimem GPU microtests also pass with no skips under a four-GPU socket lease.
The focused command is:

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
