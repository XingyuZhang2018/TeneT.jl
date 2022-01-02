using VUMPS
using VUMPS: getsymmetry, _arraytype, Iinitial, zerosZ2, bulkdims
using CUDA
using Random
using Test
using OMEinsum
CUDA.allowscalar(false)

@testset "symmetry $(symmetry) $atype{$dtype}" for atype in [Array, CuArray], dtype in [Float64, ComplexF64], symmetry in [:none, :Z2]
    A = randinitial(Val(symmetry), atype, dtype, 2,2,2)
    @test getsymmetry(A) == Val(symmetry)
    @test _arraytype(A) == (symmetry == :none ? atype : Z2tensor)
    a = randinitial(Val(symmetry), atype, dtype, 4,4)
end

@testset "parity_conserving and tensor2Z2tensor tensor2Z2tensor compatibility" begin
    a = randinitial(Val(:none), Array, Float64, 3, 2)
    a = parity_conserving(a)
    b = tensor2Z2tensor(a)
    c = Z2tensor2tensor(b)
    d = tensor2Z2tensor(c)
    @test a == c && b == d
end

@testset "reshape compatibility" begin
    a = randinitial(Val(:none), Array, Float64, 3, 8, 3)
    a = parity_conserving(a)
    a = reshape(a,3,2,4,3)
    b = tensor2Z2tensor(a)
    c = Z2tensor2tensor(b)
    @test a == c
end

@testset "reshape compatibility" begin
    aZ2 = randinitial(Val(:Z2), Array, Float64, 2,2,2,2,2,2,2,2)
    bZ2 = Z2reshape(aZ2,4,4,4,4)
    cZ2 = Z2reshape(bZ2,2,2,2,2,2,2,2,2)
    @test aZ2 == cZ2

    a = Z2tensor2tensor(aZ2)
    b = reshape(a, 4,4,4,4) 
    bZ2t = tensor2Z2tensor(b)
    @test bZ2t == bZ2
    c = reshape(b, 2,2,2,2,2,2,2,2)
    cZ2t = tensor2Z2tensor(c)
    @test cZ2t == cZ2

    aZ2 = randinitial(Val(:Z2), Array, Float64, 10,2,2,10)
    bZ2 = Z2reshape(aZ2,10,4,10)
    cZ2 = Z2reshape(bZ2,10,2,2,10)
    @test aZ2 == cZ2

    a = Z2tensor2tensor(aZ2)
    b = reshape(a, 10,4,10) 
    bZ2t = tensor2Z2tensor(b)
    @test bZ2t == bZ2
    c = reshape(b, 10,2,2,10)
    cZ2t = tensor2Z2tensor(c)
    @test cZ2t == cZ2
end