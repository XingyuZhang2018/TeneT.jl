using VUMPS
using VUMPS: qrpos,lqpos,sysvd!,_arraytype,zerosZ2, AbstractArray
using CUDA
using KrylovKit
using LinearAlgebra
using OMEinsum
using Random
using Test
using BenchmarkTools
CUDA.allowscalar(false)

@testset "parity_conserving" for atype in [Array], dtype in [ComplexF64]
    Random.seed!(100)
    D = 2
    T = atype(rand(dtype,D,D,D))
	T = parity_conserving(T)
	s = 0
	for i in 1:D, j in 1:D, k in 1:D
		(((i + j + k) - 3) % 2 != 0) && (s += T[i,j,k])
	end
	@test s == 0
end

@testset "parity_conserving and asZ2Array asZ2Array compatibility" begin
    # a = randinitial(Val(:none), Array, Float64, 3, 8, 3)
    # a = parity_conserving(a)
    # b = asZ2Array(a)
    # c = asArray(b)
    # d = asZ2Array(c)
    # @test a == c && b == d

    using BitBasis
    function swapgatedD(d::Int, D::Int)
        S = ein"ij,kl->ikjl"(Matrix{ComplexF64}(I,d,d),Matrix{ComplexF64}(I,D,D))
        for j = 1:D, i = 1:d
            sum(bitarray(i-1,Int(ceil(log2(d)))))%2 != 0 && sum(bitarray(j-1,Int(ceil(log2(D)))))%2 != 0 && (S[i,j,:,:] .= -S[i,j,:,:])
        end
        return S
    end
    AI = ein"ij,kl->ikjl"(Matrix{ComplexF64}(I,2,2),Matrix{ComplexF64}(I,2,2))
    @show asZ2Array(swapgatedD(4, 3)).tensor[1]
    # @test AI == -asZ2Array(swapgatedD(4, 4)).tensor[8]
end

@testset "Z2 Tensor with $atype{$dtype}" for atype in [Array, CuArray], dtype in [Float64, ComplexF64]
	Random.seed!(100)
	@test Z2Array <: AbstractSymmetricArray <: AbstractArray

	A = randZ2(atype, dtype, 3,3,4)
	Atensor = asArray(A)
	@test A isa Z2Array

	# permutedims
	@test permutedims(Atensor,[3,2,1]) == asArray(permutedims(A,[3,2,1]))

	## reshape
	@test reshape(Atensor,(9,4)) == reshape(asArray(reshape(reshape(A,9,4),3,3,4)),(9,4))
end

@testset "reshape parity_conserving compatibility" begin
    a = randinitial(Val(:none), Array, Float64, 3, 8, 3)
    a = parity_conserving(a)
    a = reshape(a,3,2,4,3)
    b = asZ2Array(a)
    c = asArray(b)
    @test a == c
end

@testset "reshape compatibility" begin
    aZ2 = randinitial(Val(:Z2), Array, Float64, 2,2,2,2,2,2,2,2)
    bZ2 = Z2reshape(aZ2,4,4,4,4)
    cZ2 = Z2reshape(bZ2,2,2,2,2,2,2,2,2)
    @test aZ2 == cZ2

    a = asArray(aZ2)
    b = reshape(a, 4,4,4,4) 
    bZ2t = asZ2Array(b)
    @test bZ2t == bZ2
    c = reshape(b, 2,2,2,2,2,2,2,2)
    cZ2t = asZ2Array(c)
    @test cZ2t == cZ2

    aZ2 = randinitial(Val(:Z2), Array, Float64, 10,2,2,10)
    bZ2 = Z2reshape(aZ2,10,4,10)
    cZ2 = Z2reshape(bZ2,10,2,2,10)
    @test aZ2 == cZ2

    a = asArray(aZ2)
    b = reshape(a, 10,4,10) 
    bZ2t = asZ2Array(b)
    @test bZ2t == bZ2
    c = reshape(b, 10,2,2,10)
    cZ2t = asZ2Array(c)
    @test cZ2t == cZ2
end

@testset "general flatten reshape" begin
    # (D,D,D,D,D,D,D,D)->(D^2,D^2,D^2,D^2)
    a = randinitial(Val(:Z2), Array, Float64, 3,3,3,3,3,3,3,3)
    atensor = asArray(a)
    rea = Z2reshape(a, 9,9,9,9)
    rea2 = asZ2Array(reshape(atensor, 9,9,9,9))
    @test rea !== rea2
    rerea = Z2reshape(rea, 3,3,3,3,3,3,3,3)
    @test rerea ≈ a

    # (χ,D,D,χ) -> (χ,D^2,χ)
    a = randinitial(Val(:Z2), CuArray, Float64, 5, 3, 3, 5)
    rea = Z2reshape(a, 5, 9, 5)
    rerea = Z2reshape(rea, 5, 3, 3, 5)
    @test rerea ≈ a
end

@testset "OMEinsum Z2 with $atype{$dtype}" for atype in [Array], dtype in [Float64]
	Random.seed!(100)
	A = randZ2(atype, dtype, 3,3,4)
	B = randZ2(atype, dtype, 4,3)
	Atensor = asArray(A)
	Btensor = asArray(B)

	## binary contraction
	@test ein"abc,cd -> abd"(Atensor,Btensor) ≈ asArray(ein"abc,cd -> abd"(A,B))
	@test ein"abc,db -> adc"(Atensor,Btensor) ≈ asArray(ein"abc,db -> adc"(A,B))
	@test ein"cba,dc -> abd"(Atensor,Btensor) ≈ asArray(ein"cba,dc -> abd"(A,B))
	@test ein"abc,cb -> a"(Atensor,Btensor) ≈ asArray(ein"abc,cb -> a"(A,B))
	@test ein"bac,cb -> a"(Atensor,Btensor) ≈ asArray(ein"bac,cb -> a"(A,B))
	@test ein"cba,ab -> c"(Atensor,Btensor) ≈ asArray(ein"cba,ab -> c"(A,B))

	## NestedEinsum
	@test ein"(abc,cd),ed -> abe"(Atensor,Btensor,Btensor) ≈ asArray(ein"abd,ed -> abe"(ein"abc,cd -> abd"(A,B),B)) ≈ asArray(ein"(abc,cd),ed -> abe"(A,B,B))

	## constant
	@test Array(ein"abc,abc ->"(Atensor,Atensor))[] ≈ Array(ein"abc,abc ->"(A,A))[]

	## tr
	B = randZ2(atype, dtype, 4,4)
	Btensor = asArray(B)
	@test Array(ein"aa ->"(Btensor))[] ≈ Array(ein"aa ->"(B))[]

	B = randZ2(atype, dtype, 2,2,2,2)
	Btensor = asArray(B)
	@test Array(ein"abab -> "(Btensor))[] ≈ tr(reshape(B,4,4))
	@test Array(ein"aabb -> "(Btensor))[] ≈ Array(ein"aabb-> "(B))[]

	## VUMPS unit
	d = 2
    D = 5
    AL = randZ2(atype, dtype, D, d, D)
    M = randZ2(atype, dtype, d, d, d, d)
    FL = randZ2(atype, dtype, D, d, D)
    tAL, tM, tFL = map(asArray,[AL, M, FL])
	tFL = ein"((adf,abc),dgeb),fgh -> ceh"(tFL,tAL,tM,conj(tAL))
	FL = ein"((adf,abc),dgeb),fgh -> ceh"(FL,AL,M,conj(AL))
    @test tFL ≈ asArray(FL) 

	# # autodiff test
	# D,d = 3,2
	# FL = randZ2(atype, dtype, D, d, D)
	# S = randZ2(atype, dtype, D, d, D, D, d, D)
	# FLtensor = asArray(FL)
	# Stensor = asArray(S)
	# @test ein"(abc,abcdef),def ->"(FL, S, FL)[] ≈ ein"(abc,abcdef),def ->"(FLtensor, Stensor, FLtensor)[]
end

@testset "inplace function with $symmetry $atype{$dtype}" for atype in [Array], dtype in [ComplexF64], symmetry in [:Z2]
    Random.seed!(100)
    d = 2
    χ = 2

    ## rmul!
    A = randinitial(Val(symmetry), atype, dtype, χ, χ)
    Acopy = copy(A)
    @test A*2.0 == rmul!(A, 2.0)
    @test A.tensor != Acopy.tensor

    ## lmul!
    A = randinitial(Val(symmetry), atype, dtype, χ, χ)
    B = randinitial(Val(symmetry), atype, dtype, χ, χ)
    Bcopy = copy(B)
    @test A*B == lmul!(A, B) 
    @test B.tensor != Bcopy.tensor

    ## mul!
    A = randinitial(Val(symmetry), atype, dtype, χ, χ)
    Y = similar(A)
    Ycopy = copy(Y)
    @test A*2.0 == mul!(Y, A, 2.0)
    @test Y.tensor != Ycopy.tensor

    ## axpy!
    A = randinitial(Val(symmetry), atype, dtype, χ, χ)
    B = randinitial(Val(symmetry), atype, dtype, χ, χ)
    Bcopy = copy(B)
    @test A*2.0 + B == axpy!(2.0, A, B)
    @test B.tensor != Bcopy.tensor
end

@testset "KrylovKit with $atype{$dtype}" for atype in [Array], dtype in [ComplexF64]
    Random.seed!(100)
    d = 3
    D = 5
    AL = randZ2(atype, dtype, D, d, D)
    M = randZ2(atype, dtype, d, d, d, d)
    FL = randZ2(atype, dtype, D, d, D)
    tAL, tM, tFL = map(asArray,[AL, M, FL])
    λs, FLs, info = eigsolve(FL -> ein"((adf,abc),dgeb),fgh -> ceh"(FL,AL,M,conj(AL)), FL, 1, :LM; ishermitian = false)
    tλs, tFLs, info = eigsolve(tFL -> ein"((adf,abc),dgeb),fgh -> ceh"(tFL,tAL,tM,conj(tAL)), tFL, 1, :LM; ishermitian = false)
    @test λs ≈ tλs
    @test asArray(FLs[1]) ≈ tFLs[1]

    λl,FL = λs[1], FLs[1]
    dFL = randZ2(atype, dtype, D, d, D)
    dFL -= Array(ein"abc,abc ->"(conj(FL), dFL))[] * FL
    ξl, info = linsolve(FR -> ein"((ceh,abc),dgeb),fgh -> adf"(FR, AL, M, conj(AL)), conj(dFL), -λl, 1)
    tλl,tFL = tλs[1], tFLs[1]
    tdFL = asArray(dFL)
    tξl, info = linsolve(tFR -> ein"((ceh,abc),dgeb),fgh -> adf"(tFR, tAL, tM, conj(tAL)), conj(tdFL), -tλl, 1)
    @test asArray(ξl) ≈ tξl
end

@testset "Z2 qr with $atype{$dtype}" for atype in [Array], dtype in [Float64]
    Random.seed!(100)
    A = randZ2(atype, dtype, 5, 3, 5)
	Atensor = asArray(A)
	A = reshape(A, 15, 4) 
	Atensor = reshape(Atensor, 15, 5)
	Q, R = qrpos(A)
    Qtensor, Rtensor = qrpos(Atensor)
    @test Qtensor*Rtensor ≈ Atensor
	@test Q*R ≈ A
	@test asArray(reshape(Q, 5, 3, 5)) ≈ reshape(Qtensor, 5, 3, 5)
	@test asArray(R) ≈ Rtensor
end

@testset "Z2 lq with $atype{$dtype}" for atype in [Array], dtype in [Float64]
    Random.seed!(100)
    A = randZ2(atype, dtype, 4, 3, 4)
	Atensor = asArray(A)
	A = reshape(A, 4, 12)
	Atensor = reshape(Atensor, 4, 12)
	L, Q = lqpos(A)
    Ltensor, Qtensor = lqpos(Atensor)
    @test Ltensor*Qtensor ≈ Atensor
	@test L*Q ≈ A
	@test asArray(L) ≈ Ltensor
	@test asArray(reshape(Q, 4, 3, 4)) ≈ reshape(Qtensor, 4, 3, 4)
end

@testset "Z2 svd with $atype{$dtype}" for atype in [Array, CuArray], dtype in [Float64, ComplexF64]
    Random.seed!(100)
    A = randZ2(atype, dtype, 7,4)
	Atensor = asArray(A)
	U, S, V = sysvd!(copy(A))
    Utensor, Stensor, Vtensor = sysvd!(copy(Atensor))
    @test Utensor * Diagonal(Stensor) * Vtensor ≈ Atensor
	@test U * Diagonal(S) * V ≈ A
end