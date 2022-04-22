using VUMPS
using VUMPS: u1bulkdims, randU1, zerosU1, IU1, qrpos, lqpos, sysvd!, initialA
using CUDA
using KrylovKit
using LinearAlgebra
using OMEinsum
using SparseArrays
using Random
using Test
using BenchmarkTools
CUDA.allowscalar(false)

@testset "U1 Tensor with $atype{$dtype}" for atype in [Array], dtype in [Float64]
	Random.seed!(100)
	@test U1Array <: AbstractSymmetricArray <: AbstractArray

    # u1bulkdims division
    # @test u1bulkdims(2,4) == ([1,1], [1,2,1])
    # @test u1bulkdims(5,8) == ([1,3,1], [1,3,3,1])
    # @test u1bulkdims(3,3,4) == ([1,2], [1,2], [1,2,1])
    # for a = 5:8, b = 5:8
    #     @test sum(u1bulkdims(a,b)[1]) == a
    #     @test sum(u1bulkdims(a,b)[2]) == b
    # end

    # initial 
    dir = [1,1,-1]
    @test randU1(atype, dtype, 6,7,5; dir = dir).size == (6,7,5)
    @test zerosU1(atype, dtype, 6,7,5, dir = dir).size == (6,7,5)
    @test IU1(atype, dtype, 6, dir = [1,-1]).size == (6,6)

    # # asU1Array and asArray
	A = randU1(atype, dtype, 4,4,5, dir = dir)
    @test A isa U1Array
	Atensor = asArray(A)
    AA = asU1Array(Atensor, dir = dir)
    AAtensor = asArray(AA)
    @test A ≈ AA
    @test Atensor ≈ AAtensor

	# permutedims
	@test permutedims(Atensor,[3,2,1]) == asArray(permutedims(A,[3,2,1]))

	# # reshape
	@test reshape(Atensor,(16,5)) == reshape(asArray(reshape(reshape(A,16,5),4,4,5)),(16,5))
end


@testset "OMEinsum U1 with $atype{$dtype}" for atype in [Array], dtype in [Float64]
	Random.seed!(100)
	A = randU1(atype, dtype, 3,3,4; dir = [1,1,-1], q=[0])
	B = randU1(atype, dtype, 4,3; dir = [1,-1], q=[0])
	Atensor = asArray(A)
	Btensor = asArray(B)

	# binary contraction
	@test ein"abc,cd -> abd"(Atensor,Btensor) ≈ asArray(ein"abc,cd -> abd"(A,B))
	@test ein"abc,db -> adc"(Atensor,Btensor) ≈ asArray(ein"abc,db -> adc"(A,B))
	@test ein"cba,dc -> abd"(Atensor,Btensor) ≈ asArray(ein"cba,dc -> abd"(A,B))
	@test ein"abc,cb -> a"(Atensor,Btensor) ≈ asArray(ein"abc,cb -> a"(A,B))
	@test ein"bac,cb -> a"(Atensor,Btensor) ≈ asArray(ein"bac,cb -> a"(A,B))
	@test ein"cba,ab -> c"(Atensor,Btensor) ≈ asArray(ein"cba,ab -> c"(A,B))
    a = randU1(atype, dtype, 3,7,5; dir = [1,-1,1])
    b = randU1(atype, dtype, 7,5,3; dir = [1,-1,-1])
    c = ein"abc,bcd->ad"(a,b)
    # @show a b c
    atensor = asArray(a)
    btensor = asArray(b)
    ctensor = asArray(c)
    @test ctensor ≈ ein"abc,bcd->ad"(atensor,btensor)

	# NestedEinsum
    C = randU1(atype, dtype, 4,3; dir = [-1,1])
    Ctensor = asArray(C)
	@test ein"(abc,cd),ed -> abe"(Atensor,Btensor,Ctensor) ≈ asArray(ein"abd,ed -> abe"(ein"abc,cd -> abd"(A,B),C)) ≈ asArray(ein"(abc,cd),ed -> abe"(A,B,C))

	# constant
    D = randU1(atype, dtype, 3,3,4; dir = [-1,-1,1])
    Dtensor = asArray(D)
	@test Array(ein"abc,abc ->"(Atensor,Dtensor))[] ≈ Array(ein"abc,abc ->"(A,D))[]

	# tr
	B = randU1(atype, dtype, 4,4; dir = [1,-1], q=[0])
	Btensor = asArray(B)
	@test Array(ein"aa ->"(Btensor))[] ≈ Array(ein"aa ->"(B))[]

	B = randU1(atype, dtype, 4,4,4,4; dir = [-1,-1,1,1])
	Btensor = asArray(B)
	@test Array(ein"abab -> "(Btensor))[] ≈ dtr(B)

	# VUMPS unit
	d = 4
    D = 10
    AL = randU1(atype, dtype, D,d,D; dir = [-1,1,1], q=[0])
    M = randU1(atype, dtype, d,d,d,d; dir = [-1,1,1,-1], q=[0])
    FL = randU1(atype, dtype, D,d,D; dir = [1,1,-1], q=[0])
    tAL, tM, tFL = map(asArray,[AL, M, FL])
	tFL = ein"((adf,abc),dgeb),fgh -> ceh"(tFL,tAL,tM,conj(tAL))
	FL = ein"((adf,abc),dgeb),fgh -> ceh"(FL,AL,M,conj(AL))
    @test tFL ≈ asArray(FL) 

	# autodiff test
	D,d = 4,3
	FL = randU1(atype, dtype, D, d, D; dir = [1,1,1], q=[0])
	S = randU1(atype, dtype, D, d, D, D, d, D; dir = [-1,-1,-1,-1,-1,-1])
	FLtensor = asArray(FL)
	Stensor = asArray(S)
	@test ein"(abc,abcdef),def ->"(FL, S, FL)[] ≈ ein"(abc,abcdef),def ->"(FLtensor, Stensor, FLtensor)[]
end

@testset "inplace function with $symmetry $atype{$dtype}" for atype in [Array], dtype in [ComplexF64], symmetry in [:U1]
    Random.seed!(100) 
    d = 2
    χ = 5

    # rmul!
    A = randinitial(Val(symmetry), atype, dtype, χ, χ; dir=[1,-1], q=[0])
    Acopy = copy(A)
    @test A*2.0 == rmul!(A, 2.0)
    @test A.tensor != Acopy.tensor

    # lmul!
    A = randinitial(Val(symmetry), atype, dtype, χ, χ; dir = [1, -1])
    B = randinitial(Val(symmetry), atype, dtype, χ, χ; dir = [1, -1])
    Bcopy = copy(B)
    @test A*B == lmul!(A, B) 
    @test B.tensor != Bcopy.tensor

    # mul!
    A = randinitial(Val(symmetry), atype, dtype, χ, χ; dir = [1, -1])
    Y = similar(A)
    Ycopy = copy(Y)
    @test A*2.0 == mul!(Y, A, 2.0)
    @test Y.tensor != Ycopy.tensor

    # axpy!
    A = randinitial(Val(symmetry), atype, dtype, χ, χ; dir=[1, -1])
    B = randinitial(Val(symmetry), atype, dtype, χ, χ; dir=[1, -1])
    Bcopy = copy(B)
    At = asArray(A)
    Bt = asArray(B)
    Bcopyt = asArray(Bcopy)
    @test A*2.0 + B == axpy!(2.0, A, B) == B
    @test B.tensor != Bcopy.tensor
    @test Bt + 2.0*At == axpy!(2.0, At, Bt) == asArray(axpy!(2.0, A, Bcopy))
end

@testset "KrylovKit with $atype{$dtype}" for atype in [Array], dtype in [ComplexF64]
    Random.seed!(100)
    d = 3
    D = 5
    AL = randU1(atype, dtype, D,d,D; dir = [-1,1,1], q = [0])
    M = randU1(atype, dtype, d,d,d,d; dir = [-1,1,1,-1], q = [0])
    FL = randU1(atype, dtype, D,d,D; dir = [1,1,-1], q = [0])
    # @show AL
    tAL, tM, tFL = map(asArray, [AL, M, FL])
    λs, FLs, info = eigsolve(FL -> ein"((adf,abc),dgeb),fgh -> ceh"(FL,AL,M,conj(AL)), FL, 1, :LM; ishermitian = false)
    tλs, tFLs, info = eigsolve(tFL -> ein"((adf,abc),dgeb),fgh -> ceh"(tFL,tAL,tM,conj(tAL)), tFL, 1, :LM; ishermitian = false)
    @test λs[1] ≈ tλs[1]
    @test asArray(FLs[1]) ≈ tFLs[1] 

    λl,FL = λs[1], FLs[1]
    dFL = randU1(atype, dtype, D, d, D; dir = [1,1,-1], q = [0])
    dFL -= Array(ein"abc,abc ->"(conj(FL), dFL))[] * FL
    ξl, info = linsolve(FR -> ein"((ceh,abc),dgeb),fgh -> adf"(FR, AL, M, conj(AL)), conj(dFL), -λl, 1) 
    tλl,tFL = tλs[1], tFLs[1]
    tdFL = asArray(dFL)
    tξl, info = linsolve(tFR -> ein"((ceh,abc),dgeb),fgh -> adf"(tFR, tAL, tM, conj(tAL)), conj(tdFL), -tλl, 1)
    @test asArray(ξl) ≈ tξl
end

@testset "general flatten reshape" begin
    # (D,D,D,D,D,D,D,D)->(D^2,D^2,D^2,D^2)
    D, χ = 2, 10
    # a = randinitial(Val(:U1), Array, ComplexF64, D,D,D,D,D,D,D,D; dir = [1,-1,-1,1,-1,1,1,-1])
    a = randinitial(Val(:U1), Array, ComplexF64, D,D,4,D,D; dir = [-1,-1,1,1,1], q=[0])
    a = ein"abcde, fgchi -> gbhdiefa"(a, conj(a))
    
    atensor = asArray(a)
    rea, reinfo = U1reshape(a, D^2,D^2,D^2,D^2)
    rea2 = asU1Array(reshape(atensor, D^2,D^2,D^2,D^2); dir =  [-1,1,1,-1])
    @test rea !== rea2
    rerea = U1reshape(rea, D,D,D,D,D,D,D,D; reinfo = reinfo)[1]
    @test asArray(rerea) ≈ asArray(a)

    # # (χ,D,D,χ) -> (χ,D^2,χ)
    a = randinitial(Val(:U1), Array, ComplexF64, χ,D,D,χ; dir = [-1,1,-1,1])
    atensor = asArray(a)
    rea, reinfo  = U1reshape(a, χ,D^2,χ)
    rea2 = asU1Array(reshape(atensor, χ,D^2,χ); dir =  [-1,-1,1])
    @test rea !== rea2
    rerea = U1reshape(rea, χ,D,D,χ; reinfo = reinfo)[1]
    @test rerea ≈ a
end

@testset "U1 qr with $atype{$dtype}" for atype in [Array], dtype in [Float64]
    Random.seed!(100)
    χ, D = 10, 4
    A = randU1(atype, dtype, χ,D,χ; dir = [-1,1,1])
	Atensor = asArray(A)
	A = reshape(A, χ*D,χ) 
	Atensor = reshape(Atensor, χ*D,χ)
	Q, R = qrpos(A)
    Qtensor, Rtensor = qrpos(Atensor)
    @test Qtensor*Rtensor ≈ Atensor
	@test Q*R ≈ A

    @test Qtensor'*Qtensor ≈ I(χ)
    M = ein"cda,cdb -> ab"(reshape(Q, χ,D,χ),conj(reshape(Q, χ,D,χ)))
    @test asArray(M) ≈ I(χ)

	@test asArray(reshape(Q, χ,D,χ)) ≈ reshape(Qtensor, χ,D,χ)
	@test asArray(R) ≈ Rtensor
end

@testset "U1 lq with $atype{$dtype}" for atype in [Array], dtype in [Float64]
    Random.seed!(100)
    χ, D = 10, 4
    A = randU1(atype, dtype,  χ,D,χ; dir = [-1,1,1])
	Atensor = asArray(A)
	A = reshape(A, χ, χ*D)
	Atensor = reshape(Atensor, χ, χ*D)
	L, Q = lqpos(A)
    Ltensor, Qtensor = lqpos(Atensor)
    @test Ltensor*Qtensor ≈ Atensor
	@test L*Q ≈ A

    @test Qtensor*Qtensor' ≈ I(χ)
    M = ein"acd,bcd -> ab"(reshape(Q, χ,D,χ),conj(reshape(Q, χ,D,χ)))
    @test asArray(M) ≈ I(χ)

	@test asArray(L) ≈ Ltensor
	@test asArray(reshape(Q,  χ,D,χ)) ≈ reshape(Qtensor,  χ,D,χ)
end

@testset "U1 svd with $atype{$dtype}" for atype in [Array], dtype in [Float64]
    Random.seed!(100)
    A = randU1(atype, dtype, 40,10; dir = [-1,1])
	Atensor = asArray(A)
	U, S, V = sysvd!(copy(A))
    Utensor, Stensor, Vtensor = sysvd!(copy(Atensor))
    @test Utensor * Diagonal(Stensor) * Vtensor ≈ Atensor
	@test U * Diagonal(S) * V ≈ A
end