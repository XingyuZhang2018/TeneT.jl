using TeneT
using TeneT: qrpos,lqpos,sysvd!,_arraytype,zerosZ2, AbstractArray,index_to_parity
using CUDA
using KrylovKit
using LinearAlgebra
using OMEinsum
using Random
using Test
using BenchmarkTools
CUDA.allowscalar(false)

@testset "parityconserving" for atype in [Array], dtype in [ComplexF64], siteinds in [:electron, :tJ]
    Random.seed!(100)
    D = 5
    T = atype(rand(dtype,D,D,D))
    T = parityconserving(T, Val(siteinds))
    s = 0
    for i in 1:D, j in 1:D, k in 1:D
        sum(index_to_parity.([i,j,k], Val(siteinds))) % 2 != 0 && (s += T[i,j,k])
    end
    @test s == 0
end

@testset "Z2 Tensor with $atype{$dtype} and $siteinds siteinds" for atype in [Array], dtype in [ComplexF64], siteinds in [:electron, :tJ]
	Random.seed!(100)
	@test Z2Array <: AbstractSymmetricArray <: AbstractArray

	A = randZ2(atype, dtype, Val(siteinds), 3,3,4)
    @test A isa Z2Array
	Atensor = asArray(A, Val(siteinds))

	## permutedims
	@test permutedims(Atensor,[3,2,1]) == asArray(permutedims(A,[3,2,1]), Val(siteinds))

	## reshape
	@test reshape(Atensor,(9,4)) == reshape(asArray(reshape(reshape(A,9,4),3,3,4),Val(siteinds)),(9,4))
end

@testset "parityconserving and asZ2Array, asArray compatibility" for atype in [Array, CuArray], dtype in [ComplexF64], siteinds in [:electron, :tJ]
    Random.seed!(100)
    a = randinitial(Val(:none), atype, dtype, 4, 4)
    a = parityconserving(a, Val(siteinds))
    b = asZ2Array(a, Val(siteinds))
    c = asArray(b, Val(siteinds))
    d = asZ2Array(c, Val(siteinds))
    # @show a c
    @test a == c && b == d
end


@testset "general flatten reshape" for atype in [Array], dtype in [ComplexF64], siteinds in [:electron, :tJ]
    # (D,D,D,D,D,D,D,D)->(D^2,D^2,D^2,D^2)
    D = 6
    indims = [[4, 2],[4, 2],[2, 2],[4, 2],[4, 2]]
    a = randZ2(atype, ComplexF64, Val(siteinds), D, D, 4, D, D; indims = indims)
    a = ein"abcde, fgchi -> gbhdiefa"(a, conj(a))
    @test collect.(size.(a.tensor))[[1,2,3]] == a.dims[[1,2,3]]
    
    indims = [[4, 2] for _ in 1:8]
    rea, reinfo1 = Z2reshape(a, Val(siteinds), D^2,D^2,D^2,D^2; reinfo = (nothing, nothing, indims, nothing, nothing))

    rerea1 = Z2reshape(rea, Val(siteinds), D,D,D,D,D,D,D,D; reinfo = reinfo1)[1]
    @test rerea1 ≈ a

    reinfo2 = Z2reshapeinfo((D^2,D^2,D^2,D^2), (D,D,D,D,D,D,D,D), Val(siteinds), indims)
    rerea2 = Z2reshape(rea, Val(siteinds), D,D,D,D,D,D,D,D; reinfo = reinfo2)[1]
    @test rerea2 ≈ a

    # (χ,D,D,χ) -> (χ,D^2,χ)
    D, χ = 2, 6
    indims = [[4, 2], [1, 1], [1, 1], [4, 2]]
    a = randZ2(atype, ComplexF64, Val(siteinds), χ,D,D,χ; indims = indims)
    rea, reinfo1 = Z2reshape(a, Val(siteinds), χ,D^2,χ; reinfo = (nothing, nothing, indims, nothing, nothing))
    rerea = Z2reshape(rea, Val(siteinds), χ,D,D,χ; reinfo = reinfo1)[1]
    @test rerea ≈ a

    reinfo2 = Z2reshapeinfo((χ,D^2,χ), (χ,D,D,χ), Val(siteinds), indims)
    rerea2 = Z2reshape(rea, Val(siteinds), χ,D,D,χ; reinfo = reinfo2)[1]
    @test rerea2 ≈ a
end

@testset "OMEinsum Z2 with $atype{$dtype} and $siteinds siteinds" for atype in [Array], dtype in [Float64], siteinds in [:electron, :tJ]
	Random.seed!(100)
	A = randZ2(atype, dtype, Val(siteinds), 3,3,4)
	B = randZ2(atype, dtype, Val(siteinds), 4,3)
	Atensor = asArray(A, Val(siteinds))
	Btensor = asArray(B, Val(siteinds))

	## binary contraction
	@test ein"abc,cd -> abd"(Atensor,Btensor) ≈ asArray(ein"abc,cd -> abd"(A,B), Val(siteinds))
	@test ein"abc,db -> adc"(Atensor,Btensor) ≈ asArray(ein"abc,db -> adc"(A,B), Val(siteinds))
	@test ein"cba,dc -> abd"(Atensor,Btensor) ≈ asArray(ein"cba,dc -> abd"(A,B), Val(siteinds))
	@test ein"abc,cb -> a"(Atensor,Btensor) ≈ asArray(ein"abc,cb -> a"(A,B), Val(siteinds))
	@test ein"bac,cb -> a"(Atensor,Btensor) ≈ asArray(ein"bac,cb -> a"(A,B), Val(siteinds))
	@test ein"cba,ab -> c"(Atensor,Btensor) ≈ asArray(ein"cba,ab -> c"(A,B), Val(siteinds))

	## NestedEinsum
	@test ein"(abc,cd),ed -> abe"(Atensor,Btensor,Btensor) ≈ asArray(ein"abd,ed -> abe"(ein"abc,cd -> abd"(A,B),B),Val(siteinds)) ≈ asArray(ein"(abc,cd),ed -> abe"(A,B,B),Val(siteinds))

	## constant
	@test Array(ein"abc,abc ->"(Atensor,Atensor))[] ≈ Array(ein"abc,abc ->"(A,A))[]

	## tr
	B = randZ2(atype, dtype, Val(siteinds), 4,4)
	Btensor = asArray(B, Val(siteinds))
	@test Array(ein"aa ->"(Btensor))[] ≈ Array(ein"aa ->"(B))[]

	B = randZ2(atype, dtype, Val(siteinds), 2,2,2,2)
	Btensor = asArray(B, Val(siteinds))
	@test Array(ein"abab -> "(Btensor))[] ≈ tr(reshape(B,4,4))
	# @test Array(ein"aabb -> "(Btensor))[] ≈ Array(ein"aabb-> "(B))[] # not supported for _compactify!

	## TeneT unit
	d = 2
    D = 5
    AL = randZ2(atype, dtype, Val(siteinds), D, d, D)
    M = randZ2(atype, dtype, Val(siteinds), d, d, d, d)
    FL = randZ2(atype, dtype, Val(siteinds), D, d, D)
    tAL, tM, tFL = map(x->asArray(x, Val(siteinds)),[AL, M, FL])
	tFL = ein"((adf,abc),dgeb),fgh -> ceh"(tFL,tAL,tM,conj(tAL))
	FL = ein"((adf,abc),dgeb),fgh -> ceh"(FL,AL,M,conj(AL))
    @test tFL ≈ asArray(FL, Val(siteinds)) 

	## autodiff test
	D,d = 3,2
	FL = randZ2(atype, dtype, Val(siteinds), D, d, D)
	S = randZ2(atype, dtype, Val(siteinds), D, d, D, D, d, D)
	FLtensor = asArray(FL, Val(siteinds))
	Stensor = asArray(S, Val(siteinds))
	@test ein"(abc,abcdef),def ->"(FL, S, FL)[] ≈ ein"(abc,abcdef),def ->"(FLtensor, Stensor, FLtensor)[]
end

@testset "inplace function with $symmetry $atype{$dtype} and $siteinds siteinds" for atype in [Array], dtype in [ComplexF64], symmetry in [:Z2], siteinds in [:electron, :tJ]
    Random.seed!(100)
    d = 2
    χ = 2

    ## rmul!
    A = randinitial(Val(symmetry), atype, dtype, Val(siteinds), χ, χ)
    Acopy = copy(A)
    @test A*2.0 == rmul!(A, 2.0)
    @test A.tensor != Acopy.tensor

    ## lmul!
    A = randinitial(Val(symmetry), atype, dtype, Val(siteinds), χ, χ)
    B = randinitial(Val(symmetry), atype, dtype, Val(siteinds), χ, χ)
    Bcopy = copy(B)
    @test A*B == lmul!(A, B) 
    @test B.tensor != Bcopy.tensor

    ## mul!
    A = randinitial(Val(symmetry), atype, dtype, Val(siteinds), χ, χ)
    Y = similar(A)
    Ycopy = copy(Y)
    @test A*2.0 == mul!(Y, A, 2.0)
    @test Y.tensor != Ycopy.tensor

    ## axpy!
    A = randinitial(Val(symmetry), atype, dtype, Val(siteinds), χ, χ)
    B = randinitial(Val(symmetry), atype, dtype, Val(siteinds), χ, χ)
    Bcopy = copy(B)
    @test A*2.0 + B == axpy!(2.0, A, B)
    @test B.tensor != Bcopy.tensor
end

@testset "KrylovKit with $atype{$dtype} and $siteinds siteinds" for atype in [Array], dtype in [ComplexF64], siteinds in [:electron, :tJ]
    Random.seed!(100)
    d = 3
    D = 5
    AL = randZ2(atype, dtype, Val(siteinds), D, d, D)
    M = randZ2(atype, dtype, Val(siteinds), d, d, d, d)
    FL = randZ2(atype, dtype, Val(siteinds), D, d, D)
    tAL, tM, tFL = map(x->asArray(x, Val(siteinds)), [AL, M, FL])
    λs, FLs, info = eigsolve(FL -> ein"((adf,abc),dgeb),fgh -> ceh"(FL,AL,M,conj(AL)), FL, 1, :LM; ishermitian = false)
    tλs, tFLs, info = eigsolve(tFL -> ein"((adf,abc),dgeb),fgh -> ceh"(tFL,tAL,tM,conj(tAL)), tFL, 1, :LM; ishermitian = false)
    @test λs ≈ tλs
    @test asArray(FLs[1], Val(siteinds)) ≈ tFLs[1]

    λl,FL = λs[1], FLs[1]
    dFL = randZ2(atype, dtype, Val(siteinds), D, d, D)
    dFL -= Array(ein"abc,abc ->"(conj(FL), dFL))[] * FL
    ξl, info = linsolve(FR -> ein"((ceh,abc),dgeb),fgh -> adf"(FR, AL, M, conj(AL)), conj(dFL), -λl, 1)
    tλl,tFL = tλs[1], tFLs[1]
    tdFL = asArray(dFL, Val(siteinds))
    tξl, info = linsolve(tFR -> ein"((ceh,abc),dgeb),fgh -> adf"(tFR, tAL, tM, conj(tAL)), conj(tdFL), -tλl, 1)
    @test asArray(ξl, Val(siteinds)) ≈ tξl
end

@testset "Z2 qr with $atype{$dtype} and $siteinds siteinds" for atype in [Array], dtype in [Float64], siteinds in [:electron, :tJ]
    Random.seed!(100)
    A = randZ2(atype, dtype, Val(siteinds), 5, 3, 5)
	Atensor = asArray(A, Val(siteinds))
	A = reshape(A, 15, 4) 
	Atensor = reshape(Atensor, 15, 5)
	Q, R = qrpos(A)
    Qtensor, Rtensor = qrpos(Atensor)
    @test Qtensor*Rtensor ≈ Atensor
	@test Q*R ≈ A
	@test asArray(reshape(Q, 5, 3, 5), Val(siteinds)) ≈ reshape(Qtensor, 5, 3, 5)
	@test asArray(R, Val(siteinds)) ≈ Rtensor
end

@testset "Z2 lq with $atype{$dtype} and $siteinds siteinds" for atype in [Array], dtype in [Float64], siteinds in [:electron, :tJ]
    Random.seed!(100)
    A = randZ2(atype, dtype, Val(siteinds), 4, 3, 4)
	Atensor = asArray(A, Val(siteinds))
	A = reshape(A, 4, 12)
	Atensor = reshape(Atensor, 4, 12)
	L, Q = lqpos(A)
    Ltensor, Qtensor = lqpos(Atensor)
    @test Ltensor*Qtensor ≈ Atensor
	@test L*Q ≈ A
	@test asArray(L, Val(siteinds)) ≈ Ltensor
	@test asArray(reshape(Q, 4, 3, 4), Val(siteinds)) ≈ reshape(Qtensor, 4, 3, 4)
end

@testset "Z2 svd with $atype{$dtype} and $siteinds siteinds" for atype in [Array], dtype in [Float64, ComplexF64], siteinds in [:electron, :tJ]
    Random.seed!(100)
    A = randZ2(atype, dtype, Val(siteinds), 7,4)
	Atensor = asArray(A, Val(siteinds))
	U, S, V = sysvd!(copy(A))
    Utensor, Stensor, Vtensor = sysvd!(copy(Atensor))
    @test Utensor * Diagonal(Stensor) * Vtensor ≈ Atensor
	@test U * Diagonal(S) * V ≈ A
end