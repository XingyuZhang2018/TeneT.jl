# Cross-System Performance Comparison

Benchmark results across different supercomputers for MPI multi-GPU iPEPS optimization.

## Systems

| | BSC MareNostrum5 | JSC Jupiter | Sofia VUB |
|--|------------------|-------------|-----------|
| GPU | NVIDIA H100 64GB | NVIDIA GH200 120GB | NVIDIA H200 141GB |
| Architecture | x86_64 | ARM aarch64 (Grace Hopper) | x86_64 (AMD Zen4) |
| GPU/node | 4 | 4 | **8** |
| Intra-node | NVLink | NVLink | NVSwitch |
| Inter-node | InfiniBand | InfiniBand | InfiniBand (untested) |
| MPI | OpenMPI 4.1.5 | OpenMPI 5.0.8 | OpenMPI 5.0.7 |
| UCX | 1.16.0 | default | 1.18.0 (UCX-CUDA module) |
| Julia | 1.11.3 | 1.11.1 | 1.11.3 |
| CUDA libs | CUDA.jl artifacts | CUDA.jl artifacts (strip NVHPC math_libs) | CUDA.jl artifacts + `LD_PRELOAD=/usr/lib64/libcuda.so.1` |

## Full fg Benchmark (D=10 χ=400, Plaquette VUMPS)

### Forward (leading_boundary only)

| GPU | BSC H100 | JSC GH200 | Sofia H200 |
|-----|----------|-----------|------------|
| 1   | 85.2s    | 59.2s     | 164.1s ⚠   |
| 2   | 44.1s    | 31.1s     |  86.0s ⚠   |
| 4   | 23.4s    | 19.5s     |  49.3s ⚠   |
| 8   | 13.6s    | 11.8s     |  35.7s ⚠   |

> *⚠ Sofia "Forward" numbers include a cold-start tail. BSC/JSC Forward column
> reports `leading_boundary` time at iteration 5+ of the `optimise_ipeps` loop
> (env warm-started from previous iter; VUMPS converges in ~1-3 iters). Sofia
> `bench.jl` primes the env with one warmup call then measures; this gives
> ~10% warmup benefit (cold 182s → warm 164s at 1 GPU) but still runs VUMPS
> ~25+ iters because `update!(rt, rt′)` outside the optimise_ipeps loop
> doesn't stabilise as tightly. Per-iteration compute is comparable to JSC
> (see Part 2 FLmap table below); the 3× forward-column gap is steady-state
> vs cold-ish methodology, not GPU speed.*

### Full fg (forward + backward)

| GPU | BSC H100 | JSC GH200 | Sofia H200 | Sofia/JSC |
|-----|----------|-----------|------------|-----------|
| 1   | 923s     | 831s      | 670s       | **1.24× faster** |
| 2   | 555s     | 457s      | 369s       | **1.24× faster** |
| 4   | 408s     | 324s      | 232s       | **1.39× faster** |
| 8   | 354s     | 231s      | 203s       | **1.14× faster** |

> *Sofia `fg` (forward + backward) aggregates over the full pullback and is
> apples-to-apples with JSC/BSC. Sofia H200 is consistently faster than JSC
> GH200 on full fg, with the largest gap at 4 GPU (1.39×). Breakdown: Sofia's
> backward at 1 GPU is 515s vs JSC's ~772s (33% faster), dominating the fg
> win; this is consistent with H200's HBM3e at 4.8 TB/s vs GH200's HBM3 at
> 4.0 TB/s.*

### fg Speedup (vs 1 GPU)

| GPU | BSC H100 | JSC GH200 | Sofia H200 |
|-----|----------|-----------|------------|
| 1   | 1×       | 1×        | 1×         |
| 2   | 1.66×    | 1.82×     | 1.82×      |
| 4   | 2.26×    | 2.56×     | 2.89×      |
| 8   | 2.61×    | 3.60×     | 3.30×      |

> *Sofia scales better than JSC at 4 GPU (2.89× vs 2.56×) — all-intra-node
> NVSwitch avoids the 4→8 GPU inter-node IB penalty. But at 8 GPU, JSC's
> 2-node hierarchical layout wins 3.60× vs Sofia's 8-GPU single-node 3.30×
> because the hand-rolled `allgatherv_p2p!` does N(N-1)/2 pairwise transfers
> that don't fully saturate the NVSwitch fabric.*

### Key Observations

- GH200 forward is 15-44% faster than H100 (higher memory bandwidth: 4TB/s vs 3.35TB/s).
- GH200 vs H100 multi-GPU fg scaling: 3.60× vs 2.61× at 8 GPU. The gap widens
  with more GPUs (1-GPU 11% faster → 8-GPU 53% faster). GH200 backward benefits
  from larger memory (120GB vs 64GB, less checkpoint memory pressure).
- **Sofia H200 is the fastest system on full fg across 1/2/4/8 GPU** —
  1.24× / 1.24× / 1.39× / 1.14× faster than JSC GH200, dominated by backward
  throughput (H200 HBM3e 4.8 TB/s vs GH200 HBM3 4.0 TB/s).
- Sofia has 141 GB VRAM (largest of the three), enabling larger D/χ without
  checkpoint pressure, and 8 GPU/node vs 4 on BSC/JSC. Sofia's 8-GPU test is
  intra-node only (NVSwitch); BSC/JSC 8-GPU spans 2 nodes with IB. JSC's
  hierarchical 2×4 layout actually scales better at 8 GPU (3.60× vs 3.30×)
  because `allgatherv_p2p! / allreduce_p2p!` don't saturate the single-node
  NVSwitch fabric with 28 pairwise transfers.

## FLmap_parallel Forward Comparison (selected sizes)

| D | χ | BSC H100 1GPU | JSC GH200 1GPU | Sofia H200 1GPU | Sofia/BSC | BSC 8GPU | JSC 8GPU | Sofia 8GPU |
|---|---|---------------|----------------|-----------------|-----------|----------|----------|------------|
| 8 | 512 | 356ms  | 243ms | 487ms  | 0.73x | 55ms  | 50ms  | 121ms |
| 10 | 512 | 682ms  | 447ms | 454ms  | 1.50x | 100ms | 75ms  | 142ms |
| 10 | 1024 | 3558ms | 2552ms | 2325ms | 1.53x | 624ms | 375ms | 429ms |
| 12 | 512 | 1151ms | 812ms | 979ms  | 1.18x | 168ms | 132ms | 199ms |
| 12 | 1024 | 5858ms | 4541ms | 5107ms | 1.15x | 850ms | 692ms | 795ms |

- At 1 GPU, Sofia H200 matches or slightly beats JSC GH200 at large sizes (D=10 χ=1024);
  trails at smaller sizes (D=8 χ=512) likely because its 8-GPU allocation keeps `forloop_iter=128`
  while smaller sizes are dominated by kernel-launch overhead.
- At 8 GPU (intra-node NVSwitch vs BSC/JSC inter-node IB), Sofia was SLOWER
  pre-2026-04-24 because the old `allgatherv_p2p!` / `allreduce_p2p!` did
  concurrent Isend + sequential Recv and a binary-tree reduce; with the
  bandwidth-optimal ring refactor landed on `feat/p2p-collectives-ring` the
  8-GPU intra-node numbers are now the fastest across the three systems —
  see below.

## MPI Collectives Comparison (p2p, 128MB)

| | BSC 2GPU | JSC 2GPU | Sofia 2GPU | BSC 4GPU | JSC 4GPU | Sofia 4GPU | BSC 8GPU | JSC 8GPU | Sofia 8GPU |
|--|---------|---------|------------|---------|---------|------------|---------|---------|------------|
| Allgatherv | 2.89ms | 2.72ms | **0.37ms** | 5.30ms | 6.66ms | **0.63ms** | 18.1ms | 13.2ms | **0.84ms** |
| Allreduce | 7.66ms | 10.77ms | **0.67ms** | 19.3ms | 21.6ms | **1.36ms** | 29.6ms | 37.3ms | **1.84ms** |

Sofia numbers are post-refactor (commit set on `feat/p2p-collectives-ring`,
job 1000628). Pre-refactor Sofia was 3.23/8.56/21.30ms Allgatherv and
7.24/20.02/48.59ms Allreduce — see `Sofia_VUB_H200.md` Part 1 for the
before/after table.

- Sofia beats BSC/JSC at every GPU count, primarily because BSC/JSC 8-GPU
  spans 2 nodes and hits IB latency. Sofia's 8 intra-node ranks on NVSwitch
  with the new ring algorithms cross an order of magnitude lower than IB
  round-trip costs.
- BSC/JSC could in principle get similar factor-of-10 improvements from the
  same refactor on any full-intra-node subset. The old algorithms were
  already close to optimal for the 2-node traffic pattern BSC/JSC actually
  measured, but the intra-node pairwise contention was hiding headroom even
  there.

## Environment Configuration Summary

| Setting | BSC | JSC | Sofia |
|---------|-----|-----|-------|
| `CUDA_VISIBLE_DEVICES` | `$OMPI_COMM_WORLD_LOCAL_RANK` | `$SLURM_LOCALID` | `$OMPI_COMM_WORLD_LOCAL_RANK` |
| `UCX_MEMTYPE_CACHE` | `n` (auto by MPI.jl) | `n` (must set explicitly) | `n` (must set explicitly) |
| `UCX_TLS` | default | `rc_x,self,sm,cuda_copy` | `rc_x,self,sm,cuda_copy,cuda_ipc` |
| `CUDA_LAUNCH_BLOCKING` | not needed | `1` (ARM sync worker bug) | `1` (H200 sync worker bug, same class) |
| `LD_PRELOAD` | not needed | not needed | `/usr/lib64/libcuda.so.1` (MANDATORY — see Sofia UCX_CUDA_ISSUE.md) |
| LD_LIBRARY_PATH | clean (no NVHPC) | strip NVHPC math_libs | EasyBuild-stacked (no strip needed) |
| MPI launcher | `mpirun` | `srun` | `mpirun` |
| `using` order | any | **CUDA before MPI** | any (with `LD_PRELOAD`) |
