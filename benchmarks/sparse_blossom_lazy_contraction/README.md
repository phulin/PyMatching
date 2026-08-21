# Lazy blossom contraction benchmark

This benchmark covers the lazy geometry and frontier-inheritance implementation in this branch.
It compares the branch against its `master` base using the same graph instances, syndromes,
compiler, and benchmark harness.

## Algorithm

Before this change, contracting a blossom performed two whole-area operations:

1. `wrap_into_blossom` recursively rewrote the top-region pointer and wrapped-radius cache of
   every descendant region and detector node.
2. `create_blossom` then recomputed the event frontier at every detector in the new blossom.

A forced sequence of nested tight odd cycles therefore revisited the accumulated detector area
at every contraction. A chain containing `k` two-vertex ears incurred quadratic geometry work.

The implementation replaces this with three rules:

- `blossom_parent` is the authoritative region hierarchy. Top-region pointers are caches.
  Contracting a child into a blossom changes its immediate parent in constant time.
- A detector whose cached top has since been nested resolves the parent chain only when touched.
  Every crossed region is frozen, so its radius is folded exactly into the detector's cached
  wrapped-radius offset. The common non-nested path remains an inline pointer check.
- A child that was already growing has the same local-radius trajectory after it is frozen into
  a newly growing blossom. Its queued frontier remains valid. Only children whose slope changes
  have their detector frontier rebuilt.

When a blossom shatters, affected detector and region caches are repaired eagerly before the old
blossom storage is returned to the arena. This prevents lazy pointers from outliving their parent.

For a growing child `C` frozen at dual time `tau` into a new blossom `B`,

```
C_frozen + B(t) = C(tau) + (t - tau) = C(t).
```

Thus frontier inheritance changes neither collision times nor dual trajectories. Lazy resolution
only reassociates the same sum of frozen child radii and the varying top radius.

## Validation

The branch was checked with:

- 40,000 randomized differential cases against the unmodified base: 20,000 positive-weight and
  20,000 zero-weight/erasure-style graphs. Success/failure and optimum weight agreed in every case.
- 5,000 independent exact-oracle cases. An expanded `(detector, observable-mask)` shortest-path
  solver plus subset matching DP verified both optimum weight and that the returned logical mask
  was one of the minimum-weight masks.
- 7,500 ASAN/UBSAN cases, including 4,000 zero-weight cases, with leak detection enabled.
- Focused unit tests for lazy radius resolution, frontier inheritance, shatter cache repair, and a
  depth-32 tight-triangle chain.

Equal-weight instances can have several optimum corrections. The base and this branch sometimes
select different observable masks in those ties; the exact oracle confirms that both are optimal.
Weight checksums agreed in every benchmark scenario.

## Benchmark environment

- AMD EPYC 9V74 under KVM, 5 vCPUs
- Linux 6.18.35, x86-64
- GCC 14.2.0
- `-std=c++20 -O3 -DNDEBUG -march=native`
- Five alternating base/branch runs; tables report the median per-decode time
- A reusable `Mwpm` instance and allocator arenas were used across shots

## Mixed scenarios

| Scenario | Nodes | Shots | Mean detections | Base µs | Lazy µs | Speedup |
|---|---:|---:|---:|---:|---:|---:|
| Mixed-weight 28×28 grid, low density | 784 | 400 | 15.780 | 46.254 | 47.492 | 0.974× |
| Mixed-weight 28×28 grid, high density | 784 | 250 | 195.452 | 188.584 | 175.116 | 1.077× |
| Equal-weight 28×28 grid, high density | 784 | 250 | 195.880 | 183.834 | 155.115 | 1.185× |
| 28×28 grid with zero-weight islands | 784 | 180 | 157.211 | 175.300 | 169.306 | 1.035× |
| 600-node triangle-rich sparse graph | 600 | 180 | 209.289 | 327.462 | 224.374 | 1.459× |
| 600-node graph with zero-weight edges | 600 | 150 | 180.440 | 227.349 | 213.147 | 1.067× |

The low-density mixed graph is the Sparse Blossom sweet spot and has almost no contraction work;
the additional lazy-cache branch costs about 2.7% there. Benefits rise with cycle density and equal
or zero weights, where repeated contraction becomes material.

Detailed values are in [`results.csv`](results.csv).

## Forced tight-triangle chain

The adversarial family is a chain of tight triangles. Traditional eager contraction repeatedly
walks the growing nested blossom. Lazy contraction changes only parent links and repairs detector
state on demand.

| Ears `k` | Base µs | Lazy µs | Speedup |
|---:|---:|---:|---:|
| 10 | 4.931 | 4.103 | 1.20× |
| 20 | 14.605 | 9.197 | 1.59× |
| 40 | 63.179 | 21.268 | 2.97× |
| 80 | 371.264 | 68.522 | 5.42× |
| 160 | 2,304.160 | 217.966 | 10.57× |
| 320 | 19,490.100 | 750.949 | 25.95× |
| 640 | 220,551.000 | 2,785.900 | 79.17× |

The scaling data are in [`chain_scaling.csv`](chain_scaling.csv).

## Reproducing

Create separate base and branch checkouts, then compile the same harness against each:

```bash
benchmarks/sparse_blossom_lazy_contraction/compile.sh /path/to/base /tmp/bench-base baseline
benchmarks/sparse_blossom_lazy_contraction/compile.sh /path/to/branch /tmp/bench-lazy lazy

/tmp/bench-base grid_highp_equal
/tmp/bench-lazy grid_highp_equal
/tmp/bench-base chain 320 8
/tmp/bench-lazy chain 320 8
```

Available scenarios are `grid_lowp_mixed`, `grid_highp_mixed`, `grid_highp_equal`,
`grid_erasure_zero`, `random_blossom`, and `random_zero`.
