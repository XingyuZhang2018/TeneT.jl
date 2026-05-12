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
Date: 2026-05-07T10:43:35.308
Hardware: NVIDIA GeForce RTX 4090
Settings: χ=128 β=0.43 tol=1.0e-8
====================================================================================================
Onsager f_exact at β=0.430:  -2.1285104706

Method                          VUMPS_err  λ_AC          λ_C           Z_per_site  f          Δf       Time
----------------------------------------------------------------------------------------------------
simple_eig pi=5                 err=1.11e-10  λ_AC=2.7024e+00  λ_C=5.6460e-01  Z_per_M=4.786323  f=-1.8206539871  Δf=3.08e-01  t=0.11s
simple_eig pi=10                err=1.10e-10  λ_AC=2.4033e+00  λ_C=5.0213e-01  Z_per_M=4.786323  f=-1.8206539871  Δf=3.08e-01  t=0.09s
simple_eig pi=20                err=1.11e-10  λ_AC=1.8641e+00  λ_C=3.8946e-01  Z_per_M=4.786323  f=-1.8206539871  Δf=3.08e-01  t=0.07s
simple_eig pi=50                err=2.22e-16  λ_AC=1.6513e+00  λ_C=3.4501e-01  Z_per_M=4.786323  f=-1.8206539871  Δf=3.08e-01  t=0.12s
simple_eig pi=100               err=1.12e-16  λ_AC=1.6520e+00  λ_C=3.4514e-01  Z_per_M=4.786323  f=-1.8206539871  Δf=3.08e-01  t=0.22s
GPUKrylov                       err=2.26e-16  λ_AC=2.7943e+00  λ_C=5.8382e-01  Z_per_M=4.786323  f=-1.8206539871  Δf=3.08e-01  t=0.98s

====================================================================================================
Onsager:  f = -2.1285104706
Δf shows |f_VUMPS - f_Onsager|. Should be small for converged methods.
