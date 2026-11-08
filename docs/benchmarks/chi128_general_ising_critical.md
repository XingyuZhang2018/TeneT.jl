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
# VUMPS{General} 2D Ising bench at χ=128 β=0.4406867935 tol=1.0e-6
Hardware: NVIDIA GeForce RTX 4090
Date: 2026-05-07T11:20:59.979
====================================================================================================
Onsager exact f = -2.1096511525

────────────────────────────────────────────────────────────────────────────────────────────────────
Phase 1: power_iter sweep + KrylovKit native (no GPUKrylov override yet)
────────────────────────────────────────────────────────────────────────────────────────────────────
[warmup chi=8]
[warmup chi=128 both backends]

Method                          Time     err          f               Δf       Status
----------------------------------------------------------------------------------------------------
simple_eig pi=5                 t=12.05s  err=8.27e-05  f=-2.1096511516  Δf=9.17e-10  ⚠no-conv
simple_eig pi=10                t=16.01s  err=1.90e-05  f=-2.1096511438  Δf=8.65e-09  ⚠no-conv
simple_eig pi=20                t=23.98s  err=8.31e-06  f=-2.1096511445  Δf=7.94e-09  ⚠no-conv
simple_eig pi=50                t=29.02s  err=9.99e-07  f=-2.1096511444  Δf=8.05e-09  ✓conv
simple_eig pi=100               t=18.76s  err=8.93e-07  f=-2.1096511441  Δf=8.37e-09  ✓conv
KrylovKit (native)              t=1069.23s  err=8.44e-07  f=-2.1096511446  Δf=7.90e-09  ✓conv

────────────────────────────────────────────────────────────────────────────────────────────────────
Phase 2: register GPUKrylov type-piracy and bench it
────────────────────────────────────────────────────────────────────────────────────────────────────
[warmup chi=128 with GPUKrylov type-piracy]

GPUKrylov                       t=25.25s  err=8.81e-07  f=-2.1096511446  Δf=7.90e-09  ✓conv

====================================================================================================
Summary  (χ=128, β=0.4406867935, tol=1.0e-6)
====================================================================================================
Onsager  f = -2.1096511525
Method                            Time (s)                f        |Δf|
simple_eig pi=5                      12.05    -2.1096511516    9.17e-10
simple_eig pi=10                     16.01    -2.1096511438    8.65e-09
simple_eig pi=20                     23.98    -2.1096511445    7.94e-09
simple_eig pi=50                     29.02    -2.1096511444    8.05e-09
simple_eig pi=100                    18.76    -2.1096511441    8.37e-09
KrylovKit (native)                 1069.23    -2.1096511446    7.90e-09
GPUKrylov                            25.25    -2.1096511446    7.90e-09
