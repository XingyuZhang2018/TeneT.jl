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
==========================================================================================
# Sanity: VUMPS{C4v} 2D Ising free energy vs Onsager exact
Date: 2026-05-07T10:25:50.220
Hardware: NVIDIA GeForce RTX 4090
Settings: χ=128 β=0.43 tol=1.0e-6
==========================================================================================
Onsager exact free energy at β=0.430:  f = -2.1285104706

Method                        VUMPS_err  λ_AC      f(λ_AC)/β     f(λ_AC)/2β    |Δf v1| |Δf v2|  Time
------------------------------------------------------------------------------------------
simple_eig pi=5                 err=2.25e-10  λ_AC=4.741171e+00  f(λ)=-3.6192656982  f(λ)/2=-1.8096328491  Δf=1.49e+00 | 3.19e-01  t=0.06s
simple_eig pi=10                err=2.19e-08  λ_AC=3.666031e+00  f(λ)=-3.0211851023  f(λ)/2=-1.5105925511  Δf=8.93e-01 | 6.18e-01  t=0.22s
simple_eig pi=20                err=1.25e-15  λ_AC=6.070641e+00  f(λ)=-4.1941027944  f(λ)/2=-2.0970513972  Δf=2.07e+00 | 3.15e-02  t=0.11s
simple_eig pi=50                err=1.56e-12  λ_AC=3.672698e+00  f(λ)=-3.0254103570  f(λ)/2=-1.5127051785  Δf=8.97e-01 | 6.16e-01  t=0.24s
simple_eig pi=100               err=2.41e-15  λ_AC=9.950642e-01  f(λ)=0.0115070753  f(λ)/2=0.0057535376  Δf=2.14e+00 | 2.13e+00  t=0.23s
GPUKrylov                       err=4.91e-15  λ_AC=2.011656e+01  f(λ)=-6.9803334477  f(λ)/2=-3.4901667239  Δf=4.85e+00 | 1.36e+00  t=2.62s

==========================================================================================
Onsager: f_exact = -2.1285104706
If a method is correct, |Δf| should be small.
If method's |Δf v1| ≈ |Δf v2|·factor, see which is closer to 0 to identify the doubling convention.
