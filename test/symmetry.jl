using TeneT
using TeneT: getsymmetry, _arraytype, Iinitial,  bulkdims
using CUDA
using Random
using Test
using OMEinsum
CUDA.allowscalar(false)

@testset "symmetry $(symmetry) $atype{$dtype}" for atype in [Array, CuArray], dtype in [Float64, ComplexF64], symmetry in [:none, :Z2]
    A = randinitial(Val(symmetry), atype, dtype, 2,2,2)
    @test getsymmetry(A) == symmetry
    @test _arraytype(A) == (symmetry == :none ? atype : Z2Array)
    a = randinitial(Val(symmetry), atype, dtype, 4,4)
end

@testset "symmetryreshape asSymmetryArray $(symmetry) $atype{$dtype}" for atype in [Array], dtype in [ComplexF64], symmetry in [:none, :Z2, :U1]
    a = randinitial(Val(symmetry), Array, Float64, 3,3,3,3,3,3,3,3; dir = [1,-1,-1,1,-1,1,1,-1])
    atensor = asArray(a)
    symmetry == :U1 ? ((rea, choosesilces, chooseinds) = symmetryreshape(a, 9,9,9,9)) : (rea = symmetryreshape(a, 9,9,9,9))
    rea2 = asSymmetryArray(reshape(atensor, 9,9,9,9), Val(symmetry); dir =  [-1,1,1,-1])
    @test rea !== rea2
    symmetry == :U1 ? (rerea = symmetryreshape(rea, 3,3,3,3,3,3,3,3; choosesilces = choosesilces, chooseinds = chooseinds, reqn = a.qn, redims = a.dims)) : (rerea = symmetryreshape(rea, 3,3,3,3,3,3,3,3))
    @test rerea ≈ a
end
