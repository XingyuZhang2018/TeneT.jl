┌ Error: You are using CUDA 13.0.0, but CUDA.jl was precompiled for CUDA 13.2.0.
│ This is unexpected; please file an issue.
└ @ CUDA C:\Users\xingzhan\.julia\packages\CUDA\Il00B\src\initialization.jl:148
┌ Warning: HIP library is unavailable, HIP integration will be disabled.
└ @ AMDGPU C:\Users\xingzhan\.julia\packages\AMDGPU\Z4UN5\src\AMDGPU.jl:202
┌ Warning: rocBLAS is unavailable, functionality will be disabled.
└ @ AMDGPU C:\Users\xingzhan\.julia\packages\AMDGPU\Z4UN5\src\AMDGPU.jl:213
┌ Warning: rocSPARSE is unavailable, functionality will be disabled.
└ @ AMDGPU C:\Users\xingzhan\.julia\packages\AMDGPU\Z4UN5\src\AMDGPU.jl:213
┌ Warning: rocSOLVER is unavailable, functionality will be disabled.
└ @ AMDGPU C:\Users\xingzhan\.julia\packages\AMDGPU\Z4UN5\src\AMDGPU.jl:213
┌ Warning: rocRAND is unavailable, functionality will be disabled.
└ @ AMDGPU C:\Users\xingzhan\.julia\packages\AMDGPU\Z4UN5\src\AMDGPU.jl:213
┌ Warning: rocFFT is unavailable, functionality will be disabled.
└ @ AMDGPU C:\Users\xingzhan\.julia\packages\AMDGPU\Z4UN5\src\AMDGPU.jl:213
┌ Warning: MIOpen is unavailable, functionality will be disabled.
└ @ AMDGPU C:\Users\xingzhan\.julia\packages\AMDGPU\Z4UN5\src\AMDGPU.jl:213
====================================================================================================
# VUMPS{General} 2D Ising bench at χ=128 β=0.43 tol=1.0e-6
Hardware: NVIDIA GeForce RTX 4090
Date: 2026-05-07T11:42:46.548
====================================================================================================
Onsager exact f = -2.1285104706

────────────────────────────────────────────────────────────────────────────────────────────────────
Phase 1: power_iter sweep + KrylovKit native (no GPUKrylov override yet)
────────────────────────────────────────────────────────────────────────────────────────────────────
[warmup chi=8]
[warmup chi=128 both backends]

Method                          Time     err          f               Δf       Status
----------------------------------------------------------------------------------------------------
simple_eig pi=5                 t=1.13s  err=9.39e-07  f=-2.1285104706  Δf=2.90e-13  ✓conv
simple_eig pi=10                t=0.61s  err=7.35e-07  f=-2.1285104706  Δf=2.18e-12  ✓conv
simple_eig pi=20                t=0.58s  err=4.43e-07  f=-2.1285104706  Δf=5.60e-13  ✓conv
simple_eig pi=50                t=0.65s  err=9.32e-08  f=-2.1285104706  Δf=3.06e-14  ✓conv
simple_eig pi=100               t=1.07s  err=4.18e-07  f=-2.1285104706  Δf=9.19e-14  ✓conv
KrylovKit (native)              t=8.76s  err=8.33e-08  f=-2.1285104706  Δf=1.15e-14  ✓conv

────────────────────────────────────────────────────────────────────────────────────────────────────
Phase 2: register GPUKrylov type-piracy and bench it
────────────────────────────────────────────────────────────────────────────────────────────────────
[warmup chi=128 with GPUKrylov type-piracy]

GPUKrylov                       t=0.68s  err=7.54e-08  f=-2.1285104706  Δf=1.15e-14  ✓conv

====================================================================================================
Summary  (χ=128, β=0.43, tol=1.0e-6)
====================================================================================================
Onsager  f = -2.1285104706
Method                            Time (s)                f        |Δf|
simple_eig pi=5                       1.13    -2.1285104706    2.90e-13
simple_eig pi=10                      0.61    -2.1285104706    2.18e-12
simple_eig pi=20                      0.58    -2.1285104706    5.60e-13
simple_eig pi=50                      0.65    -2.1285104706    3.06e-14
simple_eig pi=100                     1.07    -2.1285104706    9.19e-14
KrylovKit (native)                    8.76    -2.1285104706    1.15e-14
GPUKrylov                             0.68    -2.1285104706    1.15e-14
