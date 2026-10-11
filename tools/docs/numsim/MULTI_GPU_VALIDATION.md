# Multi-GPU validation status and resume checklist

This is a draft integration of the multi-GPU NumSim, Racecheck, Synccheck,
kernel-port, and benchmark work. The memory rules and their positive/negative
tests come first in [multimem](MULTI_GPU_MULTIMEM.md) and
[peer memory](MULTI_GPU_PEER.md). This page distinguishes existing evidence
from the work needed to close the correctness and performance goal.

## Current evidence

The October 10, 2026 handoff preserves the prior GB200 result files and ports
on upstream `66681fb`, including the `tirx_harness/` to `tools/` directory
rename. Subsequent validation of runtime sources at `deda386` built the native
frontend and passed **499 CPU tests in 146.23 seconds**, using four pytest
workers with work stealing. This covered the three multimem/peer runtime
files and all ten new Racecheck/Synccheck test files listed in the feature
guides. Later commits through `0d2f86e` only change documentation.

That run used Rust 1.99 and Python 3.11, with the native extension staged
manually after the wheel rejected Python 3.11. It demonstrates those CPU
checks executing, but does not pass the installation gate: the package
requires Python 3.12 or 3.13, and the native tests must also be repeated in a
supported installed environment. The full Linux Rust and parent-package suites
remain pending.

The live GPU suite has not passed. Its first attempt lacked `cuda.bindings`;
after that dependency was installed, the retry reported failures before the
SSH connection was lost. The final diagnostics were not captured, so no
kernel or hardware cause is established. No new performance-equivalence
measurements completed. Continue the GPU checks under the shared socket lease
when an active worker is available; preserve the same kernel implementation.

The subsequent full-scale publication correction passes **29 local tests**
under Python 3.13.2, TVM 0.27.0.post1, and TIRx-kernels 0.1.2.post1:
19 IR contract tests cover system-scope arrivals/polls, alias fences,
one/two-CTA builds, fixed/runtime rank, and explicit origin-ordering opt-in;
10 benchmark-evidence tests reject stale/missing/incorrect or slow results.
These do not execute the kernel's memory effects. The new full-kernel
NumSim/Racecheck/Synccheck tests and GPU comparisons remain unrun.

The subsequent engine checks reproduced and corrected acceptance of invalid
multimem specializations and stale provenance after unnamed writes. The nine
multimem Rust tests pass. The complete Rust suite on macOS reports 770 passing
and 113 failing tests; an untouched `6b24e8c` engine snapshot reports 763 passing
and the exact same 113 failures. All seven added regression tests pass. The
failure-set comparison introduces no new failures, but is not a successful
Linux gate: macOS execution is blocked by the engine's unsupported worker
floating-point environment. The installed native frontend still lowers all
28 existing multimem instruction cases. The new bulk-copy provenance kernel
also lowers; execution of it and the complete Mega MoE case remains pending.

The ABI/support-matrix tests and the 29 IR/evidence tests pass together:
**32 passed in 7.62 seconds** with four pytest workers. The broader registry
run exposed the stale ABI header, now corrected; its other 16 failures require
generated native modules that could not link CPython symbols on macOS. Those
execution-dependent checks still need the supported Linux environment.

Local handoff checks passed for all 52 changed Python files' syntax, one JSON
result file and 11,950 JSONL records, 37 local documentation links, 32 named
rule-to-test references, documentation code-block syntax, and patch whitespace.
Sphinx is not installed locally, so a rendered documentation build is also
pending. None of these static checks substitutes for the native or GPU gates.

An offline audit of the checked-in measurements found the following worst
latency ratios. A ratio is `TIRx latency / origin latency`; performance of at
least 99% requires **ratio ≤ 1 / 0.99**, approximately 1.010101. “All shapes”
means every entry in the corresponding benchmark module's `SHAPES`, with the
documented dtypes and world size 4, not every possible kernel shape or device.

| Kernel | Shape coverage | Worst original / repeat ratio | Existing numerical evidence |
|---|---|---|---|
| Two-shot all-reduce | 18 shapes, CUTLASS | 1.002734 / no independent repeat file | Benchmark checks against NCCL and reset flags. |
| GEMM + all-reduce | 19 shapes × CUTLASS and FlashInfer | 1.007572 / 1.007864 | Every retained comparison row reports bitwise equality with its origin; the runner also checks a float64 reference with the stated output-rounding tolerance. |
| All-gather GEMM | 17 shapes, fastest measured supported origin per shape | 1.006654 / 0.986765 | Comparison runs validate output on every rank before timing. The small `cutlass_default` case has CUTLASS only; the other 16 have CUTLASS, cake, and cuTile. |
| Mega MoE | 14 shapes, DeepGEMM | 1.003392 / 1.003785 | Every latest row reports zero maximum output difference; the runner checks the output and cumulative statistics before timing. |

The complete per-shape latency tables and measurement details are in the
[multimem results](MULTI_GPU_MULTIMEM.md#results) and
[peer results](MULTI_GPU_PEER.md#results). Raw data lives in
[`benchmarks/multimem/results`](../../benchmarks/multimem/results/) and
[`benchmarks/peer/results`](../../benchmarks/peer/results/). Recomputing these
ratios is an audit of recorded data, not new GPU correctness or timing evidence.
Result files do not replace fresh validation of the final commit.

The previous session reported 245 peer tests, 348 multimem tests including
GPU microtests, 885 engine tests, and 15 frontend tests passing. These counts
are historical reports; they have not been reproduced on this branch.

## Correctness limits to resolve

- **Full-scale GEMM + all-reduce needs validation of the corrected default.**
  The port now defaults to `ordering="sys_fenced"` for both origin schedules:
  complete stores, system-scope release/acquire, and alias fences on the
  publication path. Its new direct NumSim/Racecheck/Synccheck tests have not
  run yet and are not included in the 499-pass result. `ordering="origin"`
  remains an explicit negative control. Run those tests, then measure the
  corrected kernel across the complete shape matrix. Comparison rows now
  record the ordering mode, and the gate cannot reuse old signaling results.
- **Mega MoE's provenance correction needs an end-to-end rerun.** The prior
  result was `review` for one `alias_stale_read` advisory. Focused Rust
  regressions now verify invalidation of overwritten named provenance,
  including partial and sparse writes, without discarding earlier findings.
  The full-kernel test now requires `clean`, but has not run with this change.
  The [provenance explanation](MULTI_GPU_PEER.md#shared-pool-alias-advisory) describes
  the completed unnamed bulk-copy write and correction. Rerun the Linux alias
  tests, peer kernels, wiki corpus, and relevant performance gates before
  recording a clean result or revising corpus advisory goldens.
- **Coverage is bounded.** Async multimem copy/reduction/store families and
  fp8 forms are not modeled. The supported-operation table and rejection
  paths must stay consistent. The numerical model has GB200-specific floating
  reduction behavior, with one-ulp tolerance for half sums. A green test set
  is not a claim to cover arbitrary PTX forms, inputs, or GPU architectures.
- **Full-kernel test execution is still pending.** The multimem ports now live
  in `ported/cutlass/`; tests own only bindings, references, and focused rule
  kernels. Two-shot retains its existing checker coverage. The new full-scale
  GEMM tests complement the small model, but adding them does not establish
  their result or replace the final GPU validation.

## Resume gates

1. Restore the compatible harness environment on a four-GPU Linux host, initialize
   the pinned `thirdparty/tvm-rust-ext` submodule, and install this branch.
   Follow [`tests/AGENTS.md`](../../tests/AGENTS.md); do not substitute an
   unrelated TVM wheel. Record the branch commit, GPU/driver/toolkit versions,
   and origin revisions used by each benchmark.
2. Run the CPU rule tests from both feature documents, including every
   positive control and negative variant, and the engine's
   `cargo test --all-features`. Include rank-local topology, address misuse,
   release/acquire scope, proxy paths, reduction chains, and failed-access
   cleanup. Run the registry/ABI gates and the parent package suite. Compare
   failures with a pristine current-upstream worktree, using identical
   commands serially, before calling a failure pre-existing.
3. Run all live multimem microtests with `-n 1`. A skip due to missing GPUs or
   multicast support is missing validation, not a pass. Keep numerical
   comparisons against both NumSim and the independent reference, and keep
   intentionally invalid ordering cases separate from clean examples.
4. Recheck the final full-scale ports and resolve the correctness limits
   above. Retain source witnesses and all simultaneous report statuses;
   unexpected `incomplete`, deadlock, or race results block completion.
5. Run every declared benchmark shape, then an independent repeat. Check
   host CPU count/load/idle pressure and GPU activity before measurement;
   run no validation or other benchmarks concurrently. Preserve the idle
   interval used to avoid power-history bias, identical timing methods for
   each implementation, origin fixes/exclusions, correctness checks on every
   rank, and raw configuration results. Recalculate each shape's ratio
   without dropping missing or failing shapes, and require ≥99% parity for
   the **correct final kernel**. The prior four-shape multimem idle spot check
   is not a full-matrix revalidation.
6. Update the complete latency tables, exact test commands/results, source
   revisions, and this status page. Only then mark the correctness/performance
   goal complete or move the PR out of draft.

From the repository root, after restoring the environment:

```bash
git submodule update --init thirdparty/tvm-rust-ext
python -m pip install --no-deps --no-build-isolation .
cd tools
python -m pytest -q -n 16 --dist=worksteal -m "not numsim_gpu"
python -m pytest -q -n 1 tests/numsim/microtests/test_multimem.py
(cd src/tirx_harness/numsim/engine-rs && cargo test --all-features)
```

Use the focused commands and benchmark commands in the feature documents
while iterating. The full-suite run above also includes performance tests;
apply the host-idleness preflight required by the test guide first.
