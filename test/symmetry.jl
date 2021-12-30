using VUMPS
using VUMPS: getsymmetry, _arraytype, Iinitial,zerosZ2
using CUDA
using Random
using Test
CUDA.allowscalar(false)

@testset "symmetry $(symmetry) $atype{$dtype}" for atype in [Array, CuArray], dtype in [Float64, ComplexF64], symmetry in [:none, :Z2]
    A = randinitial(Val(symmetry), atype, dtype, 2,2,2)
    @test getsymmetry(A) == Val(symmetry)
    @test _arraytype(A) == (symmetry == :none ? atype : Z2tensor)
    a = randinitial(Val(symmetry), atype, dtype, 4,4)
end

@testset begin
    a = randinitial(Val(:none), Array, Float64, 4, 4, 4, 4)
    a = parity_conserving(a)
    b = tensor2Z2tensor(a)
    c = Z2tensor2tensor(b)
    d = tensor2Z2tensor(c)
    @test a == c && b == d
end