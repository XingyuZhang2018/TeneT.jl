# Sofia MPI+CUDA debugging log (resolved)

**User**: vsc48503 (Xingyu Zhang)
**Project**: bsofia_pilot_2026_0002
**Date opened**: 2026-04-23
**Date resolved**: 2026-04-24
**Partition**: zen4_h200

## TL;DR

`LD_PRELOAD=/usr/lib64/libcuda.so.1` on the per-rank env fixes the
`UCX WARN transports 'cuda_copy','cuda_ipc' are not available` + subsequent
`ibv_reg_mr Bad address` that every multi-GPU Julia MPI run was hitting.
See [`sofia_submit_test_v17.sh`](sofia_submit_test_v17.sh) for the canonical
submit template. Part 1 of `test_MPI_config.jl` (MPI collective allgatherv /
allreduce on CuArray) now passes clean on 2/4/8 GPUs on a single zen4_h200
node (acc0XX).

## Original symptom (still useful as an example)

After loading `UCX-CUDA/1.18.0-GCCcore-14.2.0-CUDA-12.8.0` +
`GDRCopy/2.4.4-GCCcore-14.2.0` + `OpenMPI/5.0.7-GCC-14.2.0`, every multi-GPU
Julia MPI run emitted:

```
UCX  WARN  transports 'cuda_copy','cuda_ipc' are not available, please use
one or more of: cma, dc, ..., self, shm, sm, sysv, tcp, ...
```

Then any MPI collective / Isend / Recv on a `CuArray` failed with:

```
ib_md.c:282  UCX  ERROR ibv_reg_mr(address=0x320000000, length=1024,
                access=0x10000f) failed: Bad address
ucp_mm.c:76  UCX  ERROR failed to register address 0x320000000 (host)
                length 1024 on md[1]=mlx5_0: Input/output error
                (md supports: host|cuda)
pml_ucx.c:934  Error: ucx send failed: Input/output error
```

## Root cause (what I had wrong before)

The previous theory — that `libuct_cuda_gdrcopy.so.0` was missing from the
UCX-CUDA module dir — was **stale**. That file was installed on Apr 20 and
is present in both `zen4-ib` and `zen5-ib` software trees. `ucx_info -d`
run standalone on a compute node (inside sbatch, after `module load`) lists
both `cuda_copy` and `cuda_ipc` transports just fine. The `"ignoring
'ucs_module_global_init'"` debug line from the old log is harmless: UCX
loads both the plain and the gdrcopy variant (gdrcopy NEEDs plain per
`readelf -d`) and picks gdrcopy — CUDA transports still register.

The actual cause is a **libcuda conflict specific to Julia**:

- Julia's `CUDA_Driver_jll` ships its own `libcuda.so` (a forwards-compat
  shim) from an artifact at `~/.julia/artifacts/abf7998d.../lib/libcuda.so`.
  On Sofia with Julia's default CUDA 13.2 artifact, this gets loaded by
  `using CUDA` with `RTLD_GLOBAL` visibility.
- UCX-CUDA 1.18 was built against the system `/usr/lib64/libcuda.so.1`
  (the actual NVIDIA kernel driver). When `mpirun`→`libmpi`→`ucp_init`
  pulls in `libuct_cuda.so.0`, its `cuInit` / `cuDeviceGetCount` symbols
  resolve **into Julia's artifact libcuda**, not the system driver.
- The artifact libcuda lacks the private UCT interfaces UCX-CUDA expects,
  so the `cuda_copy` and `cuda_ipc` transports fail to register in the
  UCP context. UCX then emits the "transports not available" warning.
- Any subsequent MPI op on a device pointer falls back to the IB (mlx5)
  path, which tries `ibv_reg_mr` on CUDA device memory, which the mlx5
  driver rejects as an invalid host address → `Bad address`.

A pure-C MPI program (`mpicc`) compiled in the same environment never
triggers any of this because it never loads Julia's artifact libcuda —
`mpirun → /tmp/mpi_cuda_test` with 8 ranks × Allreduce on CuArray-like
device buffers completed cleanly in every `UCX_TLS` configuration tested.

## The fix

Add `LD_PRELOAD=/usr/lib64/libcuda.so.1` to the per-rank env inside the
`bash -c "..."` block after `mpirun -x ...`. This forces the system driver
to be loaded first (matching soname `libcuda.so.1`), so when CUDA.jl later
`dlopen`s its artifact copy the linker reuses the already-loaded system
driver and UCX-CUDA's symbol resolution lands on the right library.

Canonical submit template:

```bash
module load GDRCopy/2.4.4-GCCcore-14.2.0
module load UCX-CUDA/1.18.0-GCCcore-14.2.0-CUDA-12.8.0
module load OpenMPI/5.0.7-GCC-14.2.0

ENVS="export CUDA_VISIBLE_DEVICES=\$OMPI_COMM_WORLD_LOCAL_RANK; \
export UCX_TLS=rc_x,self,cuda_copy,cuda_ipc; \
export UCX_MEMTYPE_CACHE=n; \
export UCX_WARN_UNUSED_ENV_VARS=n; \
export LD_PRELOAD=/usr/lib64/libcuda.so.1"

mpirun -np $N -x UCX_MODULE_DIR -x LD_LIBRARY_PATH -x PATH \
  bash -c "$ENVS; exec $JULIA --project=... /path/to/test.jl"
```

## Verification (job 1000452, `sofia_submit_test_v17.sh`)

All three configurations (N=2,4,8 GPUs on acc004) now show:

```
─── Part 1: MPI Collectives ───
  small 8KB       Allgatherv:  0.05ms  (0.1 GB/s) ✓   Allreduce:  0.11ms  (0.1 GB/s) ✓
  medium 8MB      Allgatherv:  0.42ms (18.3 GB/s) ✓   Allreduce:  0.88ms  (8.7 GB/s) ✓
  large 128MB     Allgatherv:  5.83ms (20.9 GB/s) ✓   Allreduce: 11.92ms (10.2 GB/s) ✓
```

(Numbers shown for 2 GPUs; 4/8 GPU cases also all ✓.)

Part 2 (`FLmap_parallel`) shows `FAIL: FLmap forward` at `rtol=1e-4` — this
is a **separate, non-MPI** issue (parallel vs serial reduction order
changes the result beyond the chosen tolerance). All backward tests pass.

## Things that are NOT the fix (things already tried)

- UCX_MODULE_DIR overlays (plain only, gdrcopy only, combined) — all
  produce the same behavior because the UCX-CUDA module dir is fine as
  shipped.
- Per-rank `module load` inside `bash -c` — unnecessary once the login
  node's module env is propagated via `-x LD_LIBRARY_PATH -x PATH`.
- `UCX_TLS` permutations — irrelevant, the transports weren't missing
  because of a TLS filter, they failed to register in the first place.
- `UCX_MEMTYPE_CACHE=n` — correct variable name, but it doesn't help if
  CUDA transports never registered. Keep it set to n regardless.
- `--mca pml ob1` / `--mca coll_cuda_priority 0` — same underlying
  libcuda conflict, doesn't address root cause.
- `LD_PRELOAD` of UCX-CUDA's own plugin .so files — wrong target; the
  conflict is at the driver layer, not the plugin.

## Orthogonal gotcha: `Allreduce!(IN_PLACE, d_buf, ...)`

Even with the LD_PRELOAD fix, `MPI.Allreduce!(MPI.IN_PLACE, buf, +, comm)`
on a `CuArray` still segfaults inside `mca_coll_cuda_allreduce` via
`non_overlap_accelerator_copy_content_same_ddt` → host memcpy on a CUDA
pointer. This is an OpenMPI 5.0.7 `coll_cuda` bug for the IN_PLACE path.
The TeneT `allreduce_p2p!` / `allgatherv_p2p!` routines avoid it because
they use `MPI.Isend` / `MPI.Recv!` through the UCX PML path, which does
work. If you ever need `MPI.Allreduce!` directly on CuArrays, pass
separate `sendbuf` / `recvbuf` rather than `IN_PLACE`, or launch with
`--mca coll_cuda_priority 0`.
