# Lazy blossom contraction and frontier inheritance

This directory benchmarks the algorithmic contraction change on this branch against its exact
`master` base. The branch retains Sparse Blossom's existing primal matching and nested-blossom
representation, but removes the repeated whole-area geometric work that made deeply nested,
same-dual blossoms expensive.

## Algorithm

Before this change, every blossom contraction performed two whole-area traversals:

1. `wrap_into_blossom` recursively rewrote the top-region pointer and wrapped-radius cache of every
   descendant region and detector node.
2. `create_blossom` rebuilt the event frontier at every detector in the new blossom.

A forced sequence of nested tight odd cycles therefore revisited the accumulated detector area at
every contraction. A chain containing `k` two-vertex ears incurred quadratic geometry work.

The implementation replaces this with three invariants:

- `GraphFillRegion::blossom_parent` is authoritative. Region and detector top-owner fields are
  caches. Contracting a child changes its immediate parent in constant time.
- A detector whose cached top has since been nested resolves the parent chain only when touched.
  Every crossed child is frozen, so its radius is folded exactly into the detector's cached
  wrapped-radius offset. The ordinary non-nested path is an inline pointer test.
- A child that was growing before contraction has exactly the same detector-local radius trajectory
  after it is frozen into a newly growing blossom. Its queued event frontier remains valid. Only
  children whose slope changes are rescanned.

For a growing child `C` frozen at dual time `tau` into a new growing blossom `B`,

```text
C_frozen + B(t) = C(tau) + (t - tau) = C(t).
```

When a blossom shatters, affected detector and region caches are repaired before the old blossom
storage returns to the arena. Lazy pointers therefore never outlive an arena object.

This obtains the contraction-side asymptotic benefit sought by Sparse Cherry Closure without first
replacing PyMatching's complete primal matcher with a separate cherry-forest implementation.

## Correctness validation

The final production sources passed:

- **40,000 randomized differential cases** against the unmodified base: 20,000 positive-weight and
  20,000 zero-weight/erasure-style graphs. Success/failure and optimum weight agreed in every case.
- **5,000 independent exact-oracle cases.** An expanded `(detector, observable-mask)` shortest-path
  solver plus subset matching DP verified the optimum weight and that the returned logical mask was
  one of the minimum-weight masks.
- **7,500 ASAN/UBSAN cases**, including 4,000 zero-weight cases, with leak detection enabled.
- Focused unit tests for lazy radius resolution, frontier inheritance, cache repair during shatter,
  and a depth-32 tight-triangle chain.
- The checked-in distance-13 surface-code DEM parser reproduces the repository's first five
  reference solution weights to the displayed precision before timing either implementation.

Equal-weight instances can have several optimum corrections. The base and branch sometimes select
different observable masks in those ties; the exact oracle verifies both are optimal. Matching
weight checksums agreed in every benchmark pair. The recorded surface-code workload also had
identical observable checksums.

`run_validation.sh` rebuilds and repeats the differential, oracle, and sanitizer checks. Case counts
can be reduced for a smoke test through its documented environment variables.

## Benchmark environment and method

- AMD EPYC 9V74 under KVM; 5 vCPUs
- Linux 6.18.35, x86-64
- GCC 14.2.0
- `-std=c++20 -O3 -DNDEBUG -march=native`
- Timed processes pinned to CPU 2
- Seven alternating base/branch pairs per synthetic scenario and chain size
- Fourteen alternating pairs for the recorded surface-code workload
- Tables report the median time of each implementation; quartiles and paired-median ratios are in
  the CSV files
- Graph construction is outside the timed region; one reusable `Mwpm` and its allocator arenas are
  used across all timed shots

Occasional whole-process pauses from the shared host produced isolated multi-millisecond outliers.
Medians, quartiles, and alternating order make these visible without letting them determine the
reported result.

## Representative workloads

| Scenario | Nodes | Shots | Mean detections | Base µs | Lazy µs | Speedup |
|---|---:|---:|---:|---:|---:|---:|
| Recorded rotated surface code, `d=13`, `p=0.01` | 2,184 | 1,000 | 219.990 | 96.378 | 102.056 | **0.944×** |
| Mixed-weight 28×28 grid, low density | 784 | 400 | 15.780 | 45.246 | 47.958 | **0.943×** |
| Mixed-weight 28×28 grid, high density | 784 | 250 | 195.452 | 188.993 | 174.066 | **1.086×** |
| Equal-weight 28×28 grid, high density | 784 | 250 | 195.880 | 177.489 | 153.563 | **1.156×** |
| 28×28 grid with zero-weight islands | 784 | 180 | 157.211 | 174.891 | 162.202 | **1.078×** |
| 600-node triangle-rich sparse graph | 600 | 180 | 209.289 | 338.783 | 223.215 | **1.518×** |
| 600-node graph with zero-weight edges | 600 | 150 | 180.440 | 234.819 | 205.116 | **1.145×** |

The result is intentionally not summarized as an unconditional speedup. The extra stale-owner test
costs about **6%** in ordinary low-contraction workloads, including the recorded below-threshold
surface-code sample. Once blossom contraction becomes material, the same implementation is
**8–52% faster** on these finite workloads.

Detailed medians, quartiles, run counts, and paired ratios are in [`results.csv`](results.csv).

## Forced tight-triangle chain

The adversarial family is a chain of tight triangles. Traditional eager contraction repeatedly
walks the growing nested blossom. Lazy contraction changes parent links and repairs detector state
only on demand.

| Ears `k` | Base µs | Lazy µs | Speedup |
|---:|---:|---:|---:|
| 10 | 5.349 | 4.844 | **1.10×** |
| 20 | 15.243 | 9.140 | **1.67×** |
| 40 | 65.984 | 21.120 | **3.12×** |
| 80 | 405.812 | 66.542 | **6.10×** |
| 160 | 2,257.270 | 214.164 | **10.54×** |
| 320 | 22,816.600 | 775.541 | **29.42×** |
| 640 | 229,678.000 | 2,751.080 | **83.49×** |

The increasing ratio is the intended algorithmic result: eager nested-area maintenance grows
quadratically on this family, whereas the new geometric maintenance is linear in the newly added
area plus lazy accesses. Detailed values are in [`chain_scaling.csv`](chain_scaling.csv).

## Reproducing

Compile the identical harness against separate base and branch checkouts:

```bash
benchmarks/sparse_blossom_lazy_contraction/compile.sh /path/to/base /tmp/bench-base baseline
benchmarks/sparse_blossom_lazy_contraction/compile.sh /path/to/branch /tmp/bench-lazy lazy

/tmp/bench-base grid_highp_equal 2
/tmp/bench-lazy grid_highp_equal 2
/tmp/bench-base chain 320 80
/tmp/bench-lazy chain 320 80
```

The optional number after a named scenario multiplies its built-in loop count. Available synthetic
scenarios are `grid_lowp_mixed`, `grid_highp_mixed`, `grid_highp_equal`,
`grid_erasure_zero`, `random_blossom`, and `random_zero`.

Prepare and run the checked-in surface-code sample:

```bash
benchmarks/sparse_blossom_lazy_contraction/prepare_dem_graph.py \
    data/surface_code_rotated_memory_x_13_0.01.dem /tmp/d13.pmgraph

/tmp/bench-base surface /tmp/d13.pmgraph \
    data/surface_code_rotated_memory_x_13_0.01_1000_shots.b8 10
/tmp/bench-lazy surface /tmp/d13.pmgraph \
    data/surface_code_rotated_memory_x_13_0.01_1000_shots.b8 10
```

Run the complete standalone validation suite:

```bash
benchmarks/sparse_blossom_lazy_contraction/run_validation.sh \
    /path/to/base /path/to/branch /tmp/lazy-validation
```
