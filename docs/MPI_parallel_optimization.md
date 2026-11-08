# MPI Multi-GPU Parallel Optimization for TeneT.jl

## Overview

This document summarizes the MPI multi-GPU parallelization work for iPEPS tensor network gradient optimization (`fg`) in TeneT.jl, tested on BSC MareNostrum5 (H100 64GB GPUs, OpenMPI 4.1.5, UCX 1.16.0).

**Problem size**: D=10, chi=400, Plaquette VUMPS, J1-J2 Heisenberg model on Square lattice.

## Final Performance Results

| GPU | Forward | fg (forward+backward) | fg Speedup | Energy | gnorm |
|-----|---------|----------------------|------------|--------|-------|
| 1   | 85.2s   | 923s                 | 1x         | -0.250016 | 0.01150 |
| 2   | 44.1s   | 555s                 | 1.67x      | -0.250016 | 0.01150 |
| 4   | 23.4s   | 408s                 | 2.26x      | -0.250016 | 0.01150 |
| 8 (cross-node) | 13.6s | 354s        | 2.61x      | -0.250016 | 0.01150 |

Forward scaling is near-ideal (6.26x on 8 GPUs). Full fg scaling is limited by backward MPI communication overhead (allreduce calls in AD rrule).

## Key Changes

### 1. CUDA_VISIBLE_DEVICES Isolation (Critical)

**Problem**: OpenMPI/UCX uses CUDA IPC (`cuIpcGetMemHandle`/`cuIpcOpenMemHandle`) for intra-node GPU communication. This conflicts with CUDA's stream-ordered memory allocator (default in CUDA.jl), causing the CUDA driver to reserve ~33GB per GPU for IPC mappings, leaving only ~30GB for the computation pool. This made 2+ GPU `fg` impossible (OOM).

**Solution**: Restrict each MPI process to see only its own GPU:
```bash
mpirun -np $N bash -c 'export CUDA_VISIBLE_DEVICES=$OMPI_COMM_WORLD_LOCAL_RANK; exec julia --project script.jl'
```
```julia
CUDA.device!(0)  # Each process sees only 1 GPU as device 0
```

This eliminates cuIPC mappings entirely. Works for any GPU count, single or multi-node.

**Impact**: Solved the 33GB memory leak AND improved forward performance (93s -> 44s for 2-GPU, because cuIPC overhead was eliminated).

### 2. Custom MPI Collectives with Pre-allocated Buffers

**Problem**: Standard `MPI.Allgatherv!` and `MPI.Allreduce!` are slow on CuArrays with `UCX_MEMTYPE_CACHE=n` (required by MPI.jl to avoid crashes). Standard MPI collectives use host-staging internally, 5-12x slower than direct p2p.

**Solution**: Custom implementations using `MPI.Isend`/`MPI.Recv!` with pre-allocated GPU buffers:

- **`allgatherv_p2p!`**: Star-pattern Isend from pre-allocated sendbuf + sequential Recv into pre-allocated recvbuf, then copyto target.
- **`allreduce_p2p!`**: Hierarchical tree reduce (intra-node via `Comm_split_type` + inter-node via `Comm_split` leaders), all phases using pre-allocated buffers.

Pre-allocated buffers (`_comm_sendbuf`, `_comm_recvbuf`) are essential to avoid GPU memory registration accumulation from MPI touching different CuArray addresses across calls.

**Benchmark (122MB, D=10 chi=400 tensor size)**:

| Method | 4 GPU | 8 GPU | 16 GPU |
|--------|-------|-------|--------|
| MPI.Allreduce! | 93.7ms | 98.5ms | 101.7ms |
| Custom hierarchical | **19.0ms** | **29.6ms** | **48.1ms** |
| Custom flat tree | 19.1ms | 45.4ms | 88.9ms |

Hierarchical is 1.8x faster than flat at 16 GPUs due to reduced cross-node communication.

### 3. Checkpointing Architecture (3 layers, all required)

For D=10 chi=400 on H100 64GB, all three checkpoint layers are necessary to fit the AD tape in memory:

1. **Outer**: `checkpoint(vumps_step, ...)` in `vumps_itr` — saves per-step tape for the AD iteration loop (maxiter_ad steps).
2. **Middle**: `checkpoint(_power_iter_segment, f, v, seg)` in `simple_eig` — saves intermediate eigenvector values for long power iterations (power_iter_obs=40, checkpoint every 5 steps).
3. **Inner**: `checkpoint(FLmap/FRmap/ACmap/Cmap, ...)` inside `f()` closures — saves `@tensor` intermediate tape within each map evaluation.

Removing any layer causes OOM on single GPU. The cost is ~4x backward/forward ratio per eigsolve iteration.

### 4. Bug Fix: Cenv Missing `ifcheckpoint`

`Cenv()` in `general.jl` used `ifcheckpoint` in the `f()` closure but never declared it from `alg.ifcheckpoint`. Added the missing declaration. This fix contributed ~55s speedup in fg (990s -> 935s).

## Environment Configuration

### Required for BSC (OpenMPI + UCX + CUDA)

```bash
# In sbatch script - CRITICAL for multi-GPU
export CUDA_VISIBLE_DEVICES=$OMPI_COMM_WORLD_LOCAL_RANK

# Already set by MPI.jl - DO NOT change
# UCX_MEMTYPE_CACHE=n  (avoids cudaMalloc interception crash)
```

### What NOT to do

- `JULIA_CUDA_MEMORY_POOL=none` — 2x slower (162s vs 85s forward)
- `JULIA_CUDA_MEMORY_POOL=simple` — 1.7x slower (143s vs 85s forward)
- `UCX_TLS=rc,cuda_copy,cuda_ipc,gdr_copy` — makes Allgatherv 53x slower
- `UCX_MEMTYPE_CACHE=y` — crashes with Julia/CUDA.jl

### Julia Script Device Selection

```julia
# With CUDA_VISIBLE_DEVICES set, each process sees only 1 GPU
CUDA.device!(0)
```

## Files Modified

| File | Changes |
|------|---------|
| `src/contraction/forloop_parallel_MPI.jl` | `allgatherv_p2p!` (pre-alloc recv), `allreduce_p2p!` (hierarchical + pre-alloc broadcast), pre-allocated buffer infrastructure |
| `src/autodiff/rules.jl` | `parallel()` rrule with allgatherv/allreduce in backward |
| `src/boundary_algorithm/vumps/general.jl` | `checkpoint(Map, ...)` inside f() for leftenv/rightenv/ACenv/Cenv; Cenv `ifcheckpoint` bug fix |
| `src/boundary_algorithm/vumps/plaquette.jl` | Same checkpoint pattern for ACenv_plaq/Cenv_plaq |
| `src/boundary_algorithm/vumps/c4v.jl` | Same checkpoint pattern for leftenv_c4v/ACenv_c4v/Cenv_c4v |
| `src/utils/misc.jl` | `simple_eig` with `_power_iter_segment` segmented checkpoint |

## Benchmark Details

### Forward Scaling (leading_boundary only, no AD)

| GPU | Time | Speedup | Efficiency |
|-----|------|---------|------------|
| 1   | 85.2s | 1.00x  | 100%       |
| 2   | 44.1s | 1.93x  | 96%        |
| 4   | 23.4s | 3.64x  | 91%        |
| 8   | 13.6s | 6.26x  | 78%        |

### Custom p2p vs Standard MPI (with CUDA_VISIBLE_DEVICES)

Allgatherv (122MB):

| GPU | Custom p2p | MPI.Allgatherv! | Speedup |
|-----|-----------|-----------------|---------|
| 4   | 4.6ms     | 85.9ms          | 18.7x   |
| 8   | 11.9ms    | 265.9ms         | 22.3x   |

Allreduce (122MB):

| GPU | Custom hier | MPI.Allreduce! | Speedup |
|-----|------------|----------------|---------|
| 4   | 19.0ms     | 93.7ms         | 4.9x    |
| 8   | 29.6ms     | 98.5ms         | 3.3x    |
| 16  | 48.1ms     | 101.7ms        | 2.1x    |

### Hierarchical vs Flat Allreduce

| GPU | Hierarchical | Flat | Hier/Flat |
|-----|-------------|------|-----------|
| 4 (1 node)  | 19.0ms | 19.1ms | 1.0x |
| 8 (2 nodes) | 29.6ms | 45.4ms | 1.5x |
| 12 (3 nodes)| 42.1ms | 75.7ms | 1.8x |
| 16 (4 nodes)| 48.1ms | 88.9ms | 1.8x |

Hierarchical approach is essential for cross-node performance.
