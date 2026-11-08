# BSC MareNostrum5 Benchmark Results

- **Date**: 2026-04-15
- **GPU**: NVIDIA H100 64GB
- **Interconnect**: NVLink (intra-node), InfiniBand (inter-node)
- **Software**: OpenMPI 4.1.5, UCX 1.16.0, CUDA.jl, Julia
- **Config**: `CUDA_VISIBLE_DEVICES=$OMPI_COMM_WORLD_LOCAL_RANK`, `UCX_MEMTYPE_CACHE=n`

## Part 1: MPI Collectives (allgatherv_p2p! / allreduce_p2p!)

### Allgatherv

| Size | 1 GPU | 2 GPU | 4 GPU | 8 GPU (2 nodes) |
|------|-------|-------|-------|-----------------|
| 8KB  | 0 (no-op) | 0.06ms | 0.11ms | 0.21ms |
| 8MB  | 0 | 0.20ms (37.6 GB/s) | 0.40ms (19.2 GB/s) | 2.13ms (3.6 GB/s) |
| 128MB | 0 | 2.89ms (42.2 GB/s) | 5.30ms (23.0 GB/s) | 18.12ms (6.7 GB/s) |

### Allreduce

| Size | 1 GPU | 2 GPU | 4 GPU | 8 GPU (2 nodes) |
|------|-------|-------|-------|-----------------|
| 8KB  | 0 (no-op) | 0.12ms | 0.22ms | 0.36ms |
| 8MB  | 0 | 0.55ms (13.8 GB/s) | 1.21ms (6.3 GB/s) | 2.01ms (3.8 GB/s) |
| 128MB | 0 | 7.66ms (15.9 GB/s) | 19.29ms (6.3 GB/s) | 29.58ms (4.1 GB/s) |

## Part 2: FLmap_parallel Forward

| D | χ | Size | 1 GPU | 2 GPU | 4 GPU | 8 GPU | 2x speedup | 4x speedup | 8x speedup |
|---|---|------|-------|-------|-------|-------|-------------|-------------|-------------|
| 8 | 256 | 32MB | 135.8ms | 59.4ms | 30.8ms | 18.1ms | 2.29x | 4.41x | 7.50x |
| 8 | 512 | 128MB | 356.3ms | 182.4ms | 92.9ms | 54.7ms | 1.95x | 3.84x | 6.52x |
| 8 | 1024 | 512MB | 1605ms | 817ms | 417ms | 297ms | 1.97x | 3.85x | 5.40x |
| 10 | 256 | 50MB | 161.1ms | 85.7ms | 56.7ms | 41.6ms | 1.88x | 2.84x | 3.87x |
| 10 | 512 | 200MB | 681.9ms | 346.7ms | 176.9ms | 99.9ms | 1.97x | 3.86x | 6.83x |
| 10 | 1024 | 800MB | 3558ms | 1877ms | 1027ms | 624ms | 1.90x | 3.46x | 5.70x |
| 12 | 256 | 72MB | 249.8ms | 128.6ms | 67.1ms | 37.4ms | 1.94x | 3.72x | 6.68x |
| 12 | 512 | 288MB | 1151ms | 584ms | 297ms | 168ms | 1.97x | 3.88x | 6.85x |
| 12 | 1024 | 1152MB | 5858ms | 2953ms | 1494ms | 850ms | 1.98x | 3.92x | 6.89x |

## Part 2: FLmap_parallel Backward

| D | χ | Size | 1 GPU | 2 GPU | 4 GPU | 8 GPU | 2x speedup | 4x speedup | 8x speedup |
|---|---|------|-------|-------|-------|-------|-------------|-------------|-------------|
| 8 | 256 | 32MB | 282ms | 212ms | 182ms | 123ms | 1.33x | 1.55x | 2.29x |
| 8 | 512 | 128MB | 658ms | 384ms | 316ms | 267ms | 1.71x | 2.08x | 2.47x |
| 8 | 1024 | 512MB | 3323ms | 1757ms | 1105ms | 883ms | 1.89x | 3.01x | 3.76x |
| 10 | 256 | 50MB | 392ms | 236ms | 232ms | 166ms | 1.66x | 1.69x | 2.37x |
| 10 | 512 | 200MB | 1588ms | 845ms | 569ms | 434ms | 1.88x | 2.79x | 3.66x |
| 10 | 1024 | 800MB | 8589ms | 4432ms | 2567ms | 1664ms | 1.94x | 3.35x | 5.16x |
| 12 | 256 | 72MB | 634ms | 365ms | 297ms | 243ms | 1.74x | 2.13x | 2.61x |
| 12 | 512 | 288MB | 2883ms | 1508ms | 912ms | 665ms | 1.91x | 3.16x | 4.34x |
| 12 | 1024 | 1152MB | 16879ms | 8631ms | 4712ms | 2982ms | 1.96x | 3.58x | 5.66x |

## Full fg Benchmark (D=10 χ=400, Plaquette VUMPS, with checkpoint)

| GPU | Forward | fg (fwd+bwd) | fg Speedup |
|-----|---------|-------------|------------|
| 1   | 85.2s   | 923s        | 1x         |
| 2   | 44.1s   | 555s        | 1.67x      |
| 4   | 23.4s   | 408s        | 2.26x      |
| 8 (2 nodes) | 13.6s | 354s | 2.61x      |
