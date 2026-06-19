# BSC NCCL Smoke Workflow

This is the BSC MareNostrum5 H100 workflow that passed as Slurm job `42077015`
on 2026-06-19. It tests the current `src/contraction/parallel/nccl_wrapper.jl`
without loading the full TeneT package, using only CUDA.jl and MPI.jl.

The smoke runs 2 MPI ranks on 2 H100 GPUs and verifies:

- NCCL allreduce
- NCCL allgather
- NCCL reduce-scatter

## Prewarm

If the BSC compute/login environment cannot download Julia packages directly,
prewarm the minimal project on the login node through OfflineHPC:

```bash
# Copy OfflineHPC.jl/scripts/connect.jl into this directory first.
OFFLINEHPC_PORT=18080 julia --startup-file=no prewarm_nccl_smoke.jl
```

The expected success line is:

```text
BSC_NCCL_SMOKE_PREWARM_OK
```

## Submit

```bash
sbatch submit_test.sh
```

The expected success lines include:

```text
PROJECT_LOAD_OK
rank=0 size=2 visible=0 device=NVIDIA H100
rank=1 size=2 visible=1 device=NVIDIA H100
BSC 2-GPU NCCL smoke passed
```
