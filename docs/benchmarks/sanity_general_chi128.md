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
# Sanity: VUMPS{General} 2D Ising free energy vs Onsager
Date: 2026-05-07T10:59:51.534
Hardware: NVIDIA GeForce RTX 4090
Settings: χ=128 β=0.43 tol=1.0e-10
====================================================================================================
Onsager f_exact at β=0.430:  -2.1285104706

Method                          VUMPS_err  λ_AC          λ_C           Z_per_site  f          Δf       Time
----------------------------------------------------------------------------------------------------
simple_eig pi=5                 err=7.80e-11  λ_AC=7.8373e-02  λ_C=3.1382e-02  Z_per_site=2.497423  f=-2.1285104706  Δf=9.77e-15  t=1.98s
simple_eig pi=10                err=7.77e-11  λ_AC=7.8403e-02  λ_C=3.1393e-02  Z_per_site=2.497423  f=-2.1285104706  Δf=1.15e-14  t=1.28s
simple_eig pi=20                err=4.59e-11  λ_AC=7.8550e-02  λ_C=3.1452e-02  Z_per_site=2.497423  f=-2.1285104706  Δf=1.15e-14  t=1.05s
simple_eig pi=50                err=2.79e-11  λ_AC=8.2997e-02  λ_C=3.3233e-02  Z_per_site=2.497423  f=-2.1285104706  Δf=8.88e-15  t=1.11s
simple_eig pi=100               err=9.15e-11  λ_AC=8.4167e-02  λ_C=3.3702e-02  Z_per_site=2.497423  f=-2.1285104706  Δf=1.15e-14  t=1.89s
GPUKrylov                       err=3.21e-11  λ_AC=-8.3557e-02  λ_C=-3.3457e-02  Z_per_site=2.497423  f=-2.1285104706  Δf=1.33e-15  t=2.21s

====================================================================================================
Onsager:  f = -2.1285104706
Δf shows |f_VUMPS - f_Onsager|. Should be small for converged methods.
