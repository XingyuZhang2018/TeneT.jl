using VUMPS
using VUMPS: getsymmetry, _arraytype, Iinitial, zerosZ2, bulkdims
using CUDA
using Random
using Test
using OMEinsum
CUDA.allowscalar(false)

@testset "symmetry $(symmetry) $atype{$dtype}" for atype in [Array, CuArray], dtype in [Float64, ComplexF64], symmetry in [:none, :Z2]
    A = randinitial(Val(symmetry), atype, dtype, 2,2,2)
    @test getsymmetry(A) == symmetry
    @test _arraytype(A) == (symmetry == :none ? atype : Z2tensor)
    a = randinitial(Val(symmetry), atype, dtype, 4,4)
end