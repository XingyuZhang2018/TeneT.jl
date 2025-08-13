using TeneT
using TeneT: _arraytype, set_device_id!
using TeneT: StructArray, ISA
using TeneT: qrpos,lqpos,left_canonical,right_canonical,leftenv,FLmap,rightenv,FRmap,ACenv,ACmap,Cenv,Cmap,LRtoC,ALCtoAC,ACCtoALAR,error
using TeneT: _to_front, _to_tail, permute_fronttail
using TeneT: project_AL!, project_AR!, retract!, project_AL, project_AR, permute_fronttail
using TeneT: set_device_id!, to_N_device, FLmap_parallel, FRmap_parallel, ACmap_parallel
using Test
using LinearAlgebra
using AMDGPU
using CUDA
using Random
using Test
using OMEinsum
using KrylovKit
using Zygote
CUDA.allowscalar(false)

# test_type = [Array, ROCArray];
# χ, D, d = 4, 3, 2;
# set_device_id!(test_type[2], 1)
# test_As =  [randSA(atype, [1 2; 2 1], [(χ, D, χ), (χ, D, χ)]) for atype in test_type];
# test_Ms =  [randSA(atype, [1 2; 2 1], [(D, D, D, D), (D, D, D, D)]) for atype in test_type];
# test_S1s = [randSA(atype, [1 2; 2 1], [(χ, D, χ, χ, D, χ), (χ, D, χ, χ, D, χ)]) for atype in test_type];
# test_S2s = [randSA(atype, [1 2; 2 1], [(χ, χ, χ, χ), (χ, χ, χ, χ)]) for atype in test_type];

# function num_grad(f, K; δ::Real=1e-5)
#     if eltype(K) == ComplexF64
#         (f(K + δ / 2) - f(K - δ / 2)) / δ + 
#             (f(K + δ / 2 * 1.0im) - f(K - δ / 2 * 1.0im)) / δ * 1.0im
#     else
#         (f(K + δ / 2) - f(K - δ / 2)) / δ
#     end
# end

# function num_grad(f, a::AbstractArray; δ::Real=1e-5)
#     b = Array(copy(a))
#     df = map(CartesianIndices(b)) do i
#         foo = x -> (ac = copy(b); ac[i] = x; f(_arraytype(a)(ac)))
#         num_grad(foo, b[i], δ=δ)
#     end
#     return _arraytype(a)(df)
# end

# function num_grad(f, a::StructArray; δ::Real=1e-5)
#     b = copy(a)
#     df = map(1:length(b.data)) do i
#         foo = x -> (ac = copy(b); ac[i] = x; f(ac))
#         num_grad(foo, b[i], δ=δ)
#     end
#     return df
# end