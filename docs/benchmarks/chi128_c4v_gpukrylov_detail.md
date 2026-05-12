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
==============================================================================
# C4v VUMPS bench (GPUKrylov substitute): 2D classical Ising
Date: 2026-05-07T10:15:59.036
Hardware: NVIDIA GeForce RTX 4090
Settings: chi=128, beta=0.43, tol=1.0e-6, maxiter=500
==============================================================================

[warmup chi=8]
[warmup chi=128, both eig backends]

[simple_eig power_iter=5]
  chi=128  beta=0.430  ifsimple_eig=true  power_iter=5  tol=1e-06
[ Info: Start C4v VUMPS iteration without AD...
[ Info: C4vVUMPS conv@step:    4	err = 4.992e-07	time = 0.370 sec
[ Info: Start Plaquette VUMPS iteration with AD...
[ Info: C4vVUMPS conv@step:    1	err = 2.672e-08	time = 0.770 sec
  Time: 0.77 s   err=2.67e-08  ✓

[simple_eig power_iter=10]
  chi=128  beta=0.430  ifsimple_eig=true  power_iter=10  tol=1e-06
[ Info: Start C4v VUMPS iteration without AD...
[ Info: C4vVUMPS conv@step:    7	err = 7.905e-07	time = 0.695 sec
[ Info: Start Plaquette VUMPS iteration with AD...
[ Info: C4vVUMPS conv@step:    1	err = 6.998e-08	time = 0.712 sec
  Time: 0.71 s   err=7.00e-08  ✓

[simple_eig power_iter=20]
  chi=128  beta=0.430  ifsimple_eig=true  power_iter=20  tol=1e-06
[ Info: Start C4v VUMPS iteration without AD...
[ Info: C4vVUMPS conv@step:    3	err = 5.347e-07	time = 0.348 sec
[ Info: Start Plaquette VUMPS iteration with AD...
[ Info: C4vVUMPS conv@step:    1	err = 5.214e-09	time = 0.363 sec
  Time: 0.36 s   err=5.21e-09  ✓

[simple_eig power_iter=50]
  chi=128  beta=0.430  ifsimple_eig=true  power_iter=50  tol=1e-06
[ Info: Start C4v VUMPS iteration without AD...
[ Info: C4vVUMPS conv@step:   59	err = 4.736e-07	time = 16.327 sec
[ Info: Start Plaquette VUMPS iteration with AD...
[ Info: C4vVUMPS conv@step:    1	err = 5.823e-09	time = 16.342 sec
  Time: 16.34 s   err=5.82e-09  ✓

[simple_eig power_iter=100]
  chi=128  beta=0.430  ifsimple_eig=true  power_iter=100  tol=1e-06
[ Info: Start C4v VUMPS iteration without AD...
[ Info: C4vVUMPS conv@step:    1	err = 2.955e-07	time = 0.751 sec
[ Info: Start Plaquette VUMPS iteration with AD...
[ Info: C4vVUMPS conv@step:    1	err = 1.286e-15	time = 0.793 sec
  Time: 0.79 s   err=1.29e-15  ✓

[GPUKrylov eigsolve]
  chi=128  beta=0.430  ifsimple_eig=false  power_iter=1  tol=1e-06
[ Info: Start C4v VUMPS iteration without AD...
[ Info: C4vVUMPS conv@step:    1	err = 2.492e-15	time = 0.309 sec
[ Info: Start Plaquette VUMPS iteration with AD...
[ Info: C4vVUMPS conv@step:    1	err = 7.664e-16	time = 0.373 sec
  Time: 0.37 s   err=7.66e-16  ✓

==============================================================================
Summary  (chi=128, beta=0.43, tol=1.0e-6)
==============================================================================
Method                            Time (s)     VUMPS err  Converged
simple_eig power_iter=5               0.77      2.67e-08  yes
simple_eig power_iter=10              0.71      7.00e-08  yes
simple_eig power_iter=20              0.36      5.21e-09  yes
simple_eig power_iter=50             16.34      5.82e-09  yes
simple_eig power_iter=100             0.79      1.29e-15  yes
GPUKrylov eigsolve                    0.37      7.66e-16  yes
