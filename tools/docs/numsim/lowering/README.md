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

The corresponding while lowering is in the same class. It lets CUDA unroll a
provably small literal upper bound (at most eight iterations) and keeps other
dynamic trip counts opaque:

```cpp
void CodeGenCUDA::Dispatch_(const WhileNode* op) {
  PrintIndent();
  bool small_static_bound = false;
  bool scalar_parameter_bound = false;
  if (const auto* less = op->condition.as<LTNode>()) {
    PrimExpr simplified = sym::Analyzer()->Simplify(less->b);
    const auto* immediate = simplified.as<IntImmNode>();
    small_static_bound = immediate && immediate->value >= 0 && immediate->value <= 8;
    PrimExpr bound_expr = less->b;
    if (const auto* cast = bound_expr.as<CastNode>()) bound_expr = cast->value;
    if (const auto* variable = bound_expr.as<VarNode>()) {
      scalar_parameter_bound = scalar_parameter_vars_.count(variable) != 0;
    }
  }
  stream << (small_static_bound ? "#pragma unroll 8\n"
                                : (scalar_parameter_bound ? "#pragma unroll\n"
                                                            : "#pragma unroll 1\n"));
  PrintIndent();
  stream << "while (1) {\n";
  int while_scope = BeginScope();
  std::string cond = PrintExpr(op->condition);
  PrintIndent();
  stream << "if (!(" << cond << ")) { break; }\n";
  PrintStmt(op->body);
  this->EndScope(while_scope);
  PrintIndent();
  stream << "}\n";
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

The same simplifier exposes a plain unpredicated `add.u32` on local registers
as unsigned addition. It preserves 32-bit wraparound and leaves predicated,
saturating, mixed-type, and other-width forms untouched. This lets the reference's
ordinary address-plus-16 expression reach normal address lowering instead of
remaining opaque inline assembly; it does not change the instruction contract.

### By-value parameter buffers

The reference's `layout/sym_buffer.cuh::SymBuffer::map` reads a rank offset
from its by-value kernel-parameter array. The port previously staged the
offsets into an extra shared-memory cache. That staging is not present in
DeepGEMM and is removed. The port now declares one 72-element rank-offset
array, matching the reference rather than exposing 72 scalar parameters or
requiring a scalar-selection pattern to be recognized.

`ByValueParameterBuffers` in `src/target/build_common.h` recognizes fixed,
flat, read-only `local.param` buffers. CUDA codegen emits a naturally aligned
`const __grid_constant__` array struct; normal indexed reads lower to the
reference's constant-parameter loads. The host binder requires a CPU tensor
with the declared dtype, shape, and contiguous layout. Stores and escaped
parameter addresses are rejected. Supported elements are scalar signed and
unsigned 32-/64-bit integers and 32-/64-bit floats, with 1–1024 elements.

`src/runtime/pack_args.h::PackFuncVoidAddr` passes the CPU data address to the
driver's by-value argument copy, including after library export/reload. Calls
with at most 32 arguments use bounded stack scratch; larger calls retain
dynamic scratch. The superseded four-scalar grouping experiment is removed;
its serialized tags are rejected with an instruction to rebuild, not silently
interpreted as another ABI.
Exported modules carry `tirx.by_value_parameters_v1` in their launch metadata.
Older runtimes reject that requirement before launching rather than mistaking
the new argument tag for an ordinary pointer.

The high-level Python kernel call remains unchanged. Direct PrimFunc calls and
NumSim bindings now provide `symm_rank_offsets` as one array. NumSim ABI 42
retains the by-value binding names in its host manifest and cache identity,
then snapshots those arrays independently of aliased caller buffers for every
rank. Numerical execution, Racecheck, and Synccheck use the same snapshots.
The final low-level IR contract runs after declaring the parameter scope;
ordinary global/shared loads remain forbidden by that contract.

Compiler and runtime must both come from the patched build. This introduces
no newer CUDA instruction or toolkit requirement than the CUDA 13.2 reference.

## Evidence

The four-rank `dsv3_t1` case uses one token per rank, hidden size 7168,
intermediate size 2048, 256 routed experts, top-8, and one shared expert.
Captures use CUDA/NVRTC 13.2.78 and matched reference launch configurations.
Corrections restoring reference parameter storage and synchronization are
identified below. Counts include predicated instructions.

| Capture | SASS instructions | BF16 accumulations | Registers | Static shared bytes |
|---|---:|---:|---:|---:|
| DeepGEMM reference | 5024 | 112 | 128 | 1024 |
| Unpatched TIRx lowering | 5944 | 224 | 128 | 1024 |
| Generic serial-loop lowering prototype | 4264 | 112 | 128 | 1024 |
| Loop and coordinate lowering | 4104 | 112 | 128 | 1024 |
| Loop, coordinate, and scalar-copy lowering | 4096 | 112 | 128 | 1024 |
| With parameter-table lowering and reference parameter reads | 4080 | 112 | 128 | 1024 |

The port under the unpatched lowering contains one combine body in generated
CUDA; its two copies appear during CUDA compilation. Suppressing unrolling only on serial loops
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
| Unpatched | 100.457 | 97.072 | 1.034871 | Bitwise equal |
| Prototype | 99.711 | 97.141 | 1.026456 | Bitwise equal |

The loop-only prototype improves latency but still fails the 99% gate. After
restoring the reference's mask-guarded combine-load synchronization, the
loop-and-coordinate prototype passes two independent 11-round comparisons:
98.138/97.452 microseconds (1.007039x) and 98.917/98.713 microseconds (1.002067x),
both with bitwise-equal output and statistics. The rebuilt compiler's complete
first 14-shape matrix passes, with a worst ratio of 1.0092574x (`dsv3_t1`). The
independent one-token repeat is 98.196/97.157 microseconds (1.0106940x), just
outside the 99% gate, so that build did not establish full parity. Further
lowering investigation was required; the other kernels' matrices now pass.
Reduced static instruction count alone does not establish any unmeasured gate.

Adding transparent scalar copies passes two prototype comparisons at 1.0059873x
and 1.0042761x, both numerically exact. The corresponding rebuilt IR simplifier
produces 4096 instructions as well; eight independent constant-register moves
have a different assembly order, so those timings are not substituted for the
rebuilt compiler's own matrix, which uses an immutable compiler snapshot and
a separate prepared-kernel cache.

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
failed comparisons or replace the independent full-matrix gate.

An audit of the other benchmark kernels covers 456 GEMM/all-reduce rank/config
functions, 274 all-gather functions (including auxiliary kernels), and 72
two-shot rank functions. None contains `tirx.ptx.mov`; the copy simplification
does not match them. Their loop/coordinate compiler-regression matrices pass
against the frozen preceding build.

The same 802-function audit finds no four-parameter table candidates in those
three other benchmarks. The new parameter-based Mega MoE port passes the
four-rank numerical/race/sync examples and its first `dsv3_t1` GPU comparison:
98.726/98.530 microseconds (1.001989x), with bitwise-equal output/statistics.
All 14 shapes were prepared under a separate immutable compiler snapshot.
This first result was not a full-parity claim.

The scalar-group variant's first matrix passes all 14 shapes, with a worst ratio of
1.0044118x (`dsv3_t1`) and bitwise-equal output/statistics. The independent
one-token repeat is 98.183/96.880 microseconds (1.0134496x), outside the
unchanged 99% gate. That is a performance miss, not the separate rendezvous
port collision; its row is retained and needs further lowering analysis.
The miss remains part of the historical record below.

An old/stack/stack/old comparison uses one four-GPU lease, the unchanged
11-round benchmark, and separate worker invocations:

| Argument packing | TIRx microseconds | DeepGEMM microseconds | Ratio |
|---|---:|---:|---:|
| Dynamic, first | 99.028 | 98.527 | 1.005085 |
| Stack, first | 96.322 | 96.183 | 1.001445 |
| Stack, repeat | 99.213 | 98.379 | 1.008477 |
| Dynamic, repeat | 97.983 | 97.075 | 1.009354 |

All four comparisons are numerically exact. These observations do not erase
the earlier 1.0134496x miss or prove that allocation cost alone explains it.
The stack-packing build's matrix uses separate result files.

The stack build's first tiny `port_t2_h1024_e8_k1` result is
35.943/35.436 microseconds (1.0143075x), with exact output/statistics. It still
misses the gate; passing one-token diagnostics do not close full parity.
The other three distributed kernels' independent compiler-regression rounds
are complete and pass on every retained configuration, as recorded in the
[validation audit](../MULTI_GPU_VALIDATION.md#current-evidence).

The replacement by-value-array implementation removes the scalar-group
experiment entirely. Its first prototype one-token comparison is
100.861/99.806 microseconds (1.0105705x), a retained miss; its independent
repeat is 100.598/100.182 (1.0041524x). Those prototype results are not
substituted for the final source's matrix.

The final source (`06dfdd4b74d34861`) also exposes plain unsigned register
addition to normal lowering. Its one-token cubin contains 4064 instructions
without statistics and 4080 with statistics, using 128 registers, 32 stack
bytes, and 1024 static shared bytes. Both complete 14-shape matrices pass
with bitwise-equal output/statistics. The worst ratios are 1.0085679x
(`dsv3_t1`, first) and 1.0034897x (`dsv3_t8192`, repeat), below the unchanged
`1 / 0.99` limit. Every declared shape retains its independent measurements:

| Shape | First TIRx / reference microseconds | First ratio | Repeat TIRx / reference microseconds | Repeat ratio |
|---|---:|---:|---:|---:|
| `deepgemm_default` | 2702.000 / 2713.000 | 0.9959454 | 2705.000 / 2715.000 | 0.9963168 |
| `dsv3_t1` | 99.351 / 98.507 | 1.0085679 | 99.052 / 99.301 | 0.9974925 |
| `dsv3_t1024` | 441.220 / 440.034 | 1.0026952 | 440.787 / 440.367 | 1.0009537 |
| `dsv3_t16` | 258.430 / 258.578 | 0.9994276 | 258.088 / 258.654 | 0.9978117 |
| `dsv3_t2048` | 721.167 / 723.630 | 0.9965963 | 721.116 / 723.144 | 0.9971956 |
| `dsv3_t256` | 308.764 / 309.197 | 0.9985996 | 307.996 / 308.423 | 0.9986155 |
| `dsv3_t4096` | 1434.000 / 1431.000 | 1.0020964 | 1436.000 / 1434.000 | 1.0013947 |
| `dsv3_t64` | 293.210 / 293.265 | 0.9998125 | 292.742 / 292.627 | 1.0003930 |
| `dsv3_t8192` | 2587.000 / 2578.000 | 1.0034911 | 2588.000 / 2579.000 | 1.0034897 |
| `port_t1024_e384_k6` | 642.645 / 643.884 | 0.9980757 | 643.691 / 644.214 | 0.9991882 |
| `port_t256_e384_k6` | 575.398 / 575.314 | 1.0001460 | 575.683 / 576.125 | 0.9992328 |
| `port_t2_h1024_e8_k1` | 36.158 / 38.252 | 0.9452578 | 36.249 / 37.062 | 0.9780638 |
| `port_t64_e384_k6` | 556.477 / 556.605 | 0.9997700 | 554.103 / 553.822 | 1.0005074 |
| `port_t8192_e384_k6` | 2395.000 / 2396.000 | 0.9995826 | 2392.000 / 2396.000 | 0.9983306 |

Those rounds pass their full gate, unlike the preceding failed prototypes.
A subsequent revalidation with the final runtime records a new one-token miss:
**98.793/97.277 microseconds (1.0155844x)**, with exact output/statistics.
The final 28 CUDA images are byte-identical to the passing build, so static
instruction count alone does not explain this discrepancy. This new miss is
retained, and full parity remains open while the complete final-runtime matrix
and independent repeat run.

A separate 802-function audit of the other three kernels compares generated
CUDA, host LLVM IR, and serialized device/launch metadata between their
measured compiler and the final compiler. Every corresponding file is
byte-identical; the later array/register passes do not change those ports.

### Restoring reference shared-address forms

The pinned DeepGEMM CUTLASS header at
`f3fde58372d33e9a5650ba7b80fc48b3b49d40c8` uses
`mapa.shared::cluster.u32` followed by `mbarrier.arrive[.expect_tx].shared::cluster.b64`
in `ClusterBarrier::arrive` and `ClusterTransactionBarrier::arrive_and_expect_tx`.
Four port helpers instead widened the mapped address to 64 bits and used
generic-address arrivals. They now retain the reference's 32-bit shared forms.
This restores existing reference instructions; it adds no new instruction,
release scope, barrier, work partition, or scheduling pattern.

The new source stamp is `979bd32f26952708`. The one-token cubin falls from
4064 to 4000 instructions without statistics, and from 4080 to 4016 with
statistics. All seven `SR_SWINHI` reads disappear; registers, stack, and static
shared memory remain 128, 32 bytes, and 1024 bytes. These are disassembly
observations, not hardware stall or utilization claims. All 82 focused
fidelity/peer numerical/race/sync checks pass, including emitted-IR checks for
32-bit mapping and explicit shared arrival spaces.

The unchanged 11-round GPU comparison gives:

| Shape | First TIRx / reference microseconds | First ratio | Repeat TIRx / reference microseconds | Repeat ratio |
|---|---:|---:|---:|---:|
| `dsv3_t1` | 97.348 / 97.345 | 1.0000308 | 98.569 / 98.251 | 1.0032366 |
| `port_t2_h1024_e8_k1` | 33.241 / 33.604 | 0.9891977 | 33.001 / 32.177 | 1.0256084 |

Every output and statistic matches exactly. The tiny-shape repeat still
fails the unchanged 99% gate, so this correction does not close full parity.
The old misses remain recorded; further lowering investigation is required.

### Current source revalidation

The current source stamp `979bd32f26952708` was rebuilt into a fresh cache and
run across all 14 declared shapes with the patched generic compiler. Every
output and cumulative statistic matched bitwise. The ratios were:

The compiler patch now allows bounded literal `while` conditions with an upper
bound of at most eight to use `#pragma unroll 8`; unbounded and runtime-bound
loops retain `#pragma unroll 1`. A loop whose upper bound is a scalar kernel
parameter uses CUDA's uncounted `#pragma unroll`, matching the origin's token
loop form. The exact rebuilt compiler passes `port_t2_h1024_e8_k1` at 0.991
in a 31-round run. The full 14-shape matrix has exact outputs/statistics and
13 ratios at or below 1.004; the remaining tiny-shape row is 1.011 in that
sequence. dsv3 measurements vary between the passing full-matrix result and
1.011 in an independent run, so the final gate remains explicitly open.

| Shape group | Result |
|---|---:|
| 12 of 14 shapes | at most 1.004211 |
| `dsv3_t1` | 1.011856 in the full matrix |
| `dsv3_t8192` | 1.004211 |

The one-token result is sensitive to the preceding GPU state. An isolated
31-round `dsv3_t1` run is 1.007077; after `port_t8192_e384_k6` and a 10-second
cooldown it is 1.000252. The same cooldown protocol leaves
`port_t2_h1024_e8_k1` at 1.048116. These results keep the full gate open.

The relevant lowering sites remain generic: `Dispatch_(const ForNode*)`
emits `#pragma unroll 1` for dynamic serial loops or serial loops containing a
`while`, and `BindThreadIndex` caches native thread/block/cluster coordinates
with their launch bounds. `Dispatch_(const WhileNode*)` retains the dynamic
loop guard. These paths do not inspect Mega MoE names or shapes. The tiny
shape's remaining tail is therefore not attributed to extra static SASS; the
fresh diagnostics require another lowering investigation before claiming
parity.

### Timeline evidence

Four-rank IKET captures retain the original work assignment. The cold capture
uses the benchmark's existing 8 GB cache flush, 20-million-cycle sleep, and
stream all-reduce before launch. Its CUDA-event wrapper is diagnostic only;
the uninstrumented Kineto benchmark remains the performance gate. The warm
capture separately verifies output/statistics against DeepGEMM. Trace files
remain outside Git under the caller's artifact directory.

1. **Critical path:** on the rank with the longest MMA-role span, that role
   ends at 83.936 microseconds, phase-two NVLink synchronization at 89.088,
   and the epilogue at 95.584, all relative to that rank's first warp start.
   L1-full waits reach 23.328 microseconds. These are exposed dependency
   waits, not evidence that fewer static instructions alone close the gate.
2. **Overlap and bubbles:** loader, MMA, and epilogue roles begin within
   1.344–1.632 microseconds, so the roles overlap rather than execute serially.
   Their long completion/publication waits expose gaps; no hardware-utilization
   or stall-counter conclusion is inferred from these timestamps.
3. **Balance:** the four MMA-role endpoints range from 53.888 to 83.936
   microseconds. Earlier-finishing ranks spend up to 41.344 microseconds in
   phase-two NVLink synchronization. This capture shows a rank-dependent tail;
   redistributing the work would violate the port-fidelity requirement and is
   not used as a fix.

## Regression tests

The patch adds code-generation checks for a dynamic serial loop, a static
serial loop containing a dynamic inner loop, explicit unroll policies, and a
small static loop whose existing compiler choice must remain intact. Coordinate
tests cover 17-, 32-, and 64-thread launches and cluster-coordinate mapping.

On the unpatched installed compiler, the two default-policy regressions fail
and the four explicit-policy/static-loop controls pass. On the rebuilt patched
compiler, the complete non-GPU `test_codegen_cuda.py` suite reports 89 passed
and one skipped. Both PTX dispatch/helper-golden regressions pass. Two GPU
round trips preserve all bits of 32-/64-bit NaNs, signed zeros, infinities, and
subnormals; five same-kernel partial-CTA GPU cases and all 59 simplifier tests
also pass. Invalid width/type modifiers remain visible for normal rejection.
Fifteen focused GPU tests pass: two special-bit copy cases, unsigned-add
wraparound, and six by-value-array types at extents 4 and 72. The array cases
exercise direct calls and exported/reloaded libraries, preserve floating
special-value bits, and assert the serialized runtime requirement. A separate
cross-runtime probe confirms the current runtime executes that module and an
older runtime rejects it before launch. Read-only and address-escape negative
controls pass in codegen and NumSim.

Three layout counterexamples reject strided, multidimensional, and nonzero-offset
parameter declarations before flattening can hide their original geometry.
They fail on the preceding compiler and pass after the validation fix.

With ABI 42, 106 combined peer numerical/race/sync, parameter, ABI, and fidelity
checks pass, including all four complete Mega MoE cases. The final snapshot/cache
run passes 44 tests, and 67 wait/flag/diagnostic/canonical checks pass after
updating stale strong-flag and pool-alias expectations. Both debug and release
Rust suites pass 890 unit,
four checker-import, and three public-surface tests. The earlier ABI 41 run
passes 943 multimem/peer numerical, ordering-rule, and registry/ABI tests;
all 94 live multimem GPU microtests pass with no skips under a four-GPU socket
lease. The final ABI 42 feature rerun passes 972 tests; the final compiler's
99-case GPU rerun covers all 94 multimem cases and five partial-CTA cases,
with no skips. Source-to-wheel installation and numerical smoke checks pass
for both current and compatible-parent trees. The separate broad-suite
failures and unchanged CPU-budget misses remain in the validation audit.
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
