using VUMPS
using VUMPS: u1bulkdims, randU1, zerosU1, IU1
using CUDA
using KrylovKit
using LinearAlgebra
using OMEinsum
using SparseArrays
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

@testset "parity_conserving and tensor2U1tensor tensor2U1tensor compatibility" begin
    # a = randinitial(Val(:none), Array, Float64, 3, 8, 3)
    # a = parity_conserving(a)
    # b = tensor2U1tensor(a)
    # c = U1tensor2tensor(b)
    # d = tensor2U1tensor(c)
    # @test a == c && b == d
end

@testset "U1 Tensor with $atype{$dtype}" for atype in [Array], dtype in [Float64]
	Random.seed!(100)
	@test U1tensor <: AbstractU1Array <: AbstractArray

    # u1bulkdims division
    @test u1bulkdims(2,4) == ([1,1], [1,2,1])
    @test u1bulkdims(5,8) == ([1,3,1], [1,3,3,1])
    @test u1bulkdims(3,3,4) == ([1,2], [1,2], [1,2,1])
    for a = 5:8, b = 5:8
        @test sum(u1bulkdims(a,b)[1]) == a
        @test sum(u1bulkdims(a,b)[2]) == b
    end

    # initial 
    oi = [1,1,-1]
    @test randU1(atype, dtype, oi, 6, 7, 5).size == (6,7,5)
    @test zerosU1(atype, dtype, oi, 6, 7, 5).size == (6,7,5)
    @test IU1(atype, dtype, [-1, 1], 6).size == (6,6)

    # tensor2U1tensor and U1tensor2tensor
	A = randU1(atype, dtype, [1,1,-1], 3,3,4)
    @test A isa U1tensor
	Atensor = U1tensor2tensor(A)
    AA = tensor2U1tensor(Atensor, [1,1,-1])
    @test A ≈ AA

	# permutedims
	@test permutedims(Atensor,[3,2,1]) == U1tensor2tensor(permutedims(A,[3,2,1]))

	# reshape
	@test reshape(Atensor,(9,4)) == reshape(U1tensor2tensor(reshape(reshape(A,9,4),3,3,4)),(9,4))
end

# @testset "reshape parity_conserving compatibility" begin
#     a = randinitial(Val(:none), Array, Float64, 3, 8, 3)
#     a = parity_conserving(a)
#     a = reshape(a,3,2,4,3)
#     b = tensor2U1tensor(a)
#     c = U1tensor2tensor(b)
#     @test a == c
# end

# @testset "reshape compatibility" begin
#     aU1 = randinitial(Val(:U1), Array, Float64, 2,2,2,2,2,2,2,2)
#     bU1 = U1reshape(aU1,4,4,4,4)
#     cU1 = U1reshape(bU1,2,2,2,2,2,2,2,2)
#     @test aU1 == cU1

#     a = U1tensor2tensor(aU1)
#     b = reshape(a, 4,4,4,4) 
#     bU1t = tensor2U1tensor(b)
#     @test bU1t == bU1
#     c = reshape(b, 2,2,2,2,2,2,2,2)
#     cU1t = tensor2U1tensor(c)
#     @test cU1t == cU1

#     aU1 = randinitial(Val(:U1), Array, Float64, 10,2,2,10)
#     bU1 = U1reshape(aU1,10,4,10)
#     cU1 = U1reshape(bU1,10,2,2,10)
#     @test aU1 == cU1

#     a = U1tensor2tensor(aU1)
#     b = reshape(a, 10,4,10) 
#     bU1t = tensor2U1tensor(b)
#     @test bU1t == bU1
#     c = reshape(b, 10,2,2,10)
#     cU1t = tensor2U1tensor(c)
#     @test cU1t == cU1
# end

# @testset "general flatten reshape" begin
#     # (D,D,D,D,D,D,D,D)->(D^2,D^2,D^2,D^2)
#     a = randinitial(Val(:U1), Array, Float64, 3,3,3,3,3,3,3,3)
#     atensor = U1tensor2tensor(a)
#     rea = U1reshape(a, 9,9,9,9)
#     rea2 = tensor2U1tensor(reshape(atensor, 9,9,9,9))
#     @test rea !== rea2
#     rerea = U1reshape(rea, 3,3,3,3,3,3,3,3)
#     @test rerea ≈ a

#     # (χ,D,D,χ) -> (χ,D^2,χ)
#     a = randinitial(Val(:U1), CuArray, Float64, 5, 3, 3, 5)
#     rea = U1reshape(a, 5, 9, 5)
#     rerea = U1reshape(rea, 5, 3, 3, 5)
#     @test rerea ≈ a
# end

@testset "OMEinsum U1 with $atype{$dtype}" for atype in [Array], dtype in [Float64]
	Random.seed!(100)
	A = randU1(atype, dtype, [1,1,-1], 3,3,4)
	B = randU1(atype, dtype, [1,-1], 4,3)
	Atensor = U1tensor2tensor(A)
	Btensor = U1tensor2tensor(B)

	# binary contraction
	@test ein"abc,cd -> abd"(Atensor,Btensor) ≈ U1tensor2tensor(ein"abc,cd -> abd"(A,B))
	@test ein"abc,db -> adc"(Atensor,Btensor) ≈ U1tensor2tensor(ein"abc,db -> adc"(A,B))
	@test ein"cba,dc -> abd"(Atensor,Btensor) ≈ U1tensor2tensor(ein"cba,dc -> abd"(A,B))
	@test ein"abc,cb -> a"(Atensor,Btensor) ≈ U1tensor2tensor(ein"abc,cb -> a"(A,B))
	@test ein"bac,cb -> a"(Atensor,Btensor) ≈ U1tensor2tensor(ein"bac,cb -> a"(A,B))
	@test ein"cba,ab -> c"(Atensor,Btensor) ≈ U1tensor2tensor(ein"cba,ab -> c"(A,B))
    a = randU1(atype, dtype, [1,-1,1], 4,4,4)
    b = randU1(atype, dtype, [1,-1,-1], 4,4,4)
    c = ein"abc,bcd->ad"(a,b)
    atensor = U1tensor2tensor(a)
    btensor = U1tensor2tensor(b)
    ctensor = U1tensor2tensor(c)
    @test ctensor ≈ ein"abc,bcd->ad"(atensor,btensor)

	# NestedEinsum
    C = randU1(atype, dtype, [-1,1], 4,3)
    Ctensor = U1tensor2tensor(C)
	@test ein"(abc,cd),ed -> abe"(Atensor,Btensor,Ctensor) ≈ U1tensor2tensor(ein"abd,ed -> abe"(ein"abc,cd -> abd"(A,B),C)) ≈ U1tensor2tensor(ein"(abc,cd),ed -> abe"(A,B,C))

	# constant
    D = randU1(atype, dtype, [-1,-1,1], 3,3,4)
    Dtensor = U1tensor2tensor(D)
	@test Array(ein"abc,abc ->"(Atensor,Dtensor))[] ≈ Array(ein"abc,abc ->"(A,D))[]

	# tr
	B = randU1(atype, dtype, [1,-1], 4,4)
	Btensor = U1tensor2tensor(B)
	@test Array(ein"aa ->"(Btensor))[] ≈ Array(ein"aa ->"(B))[]

	B = randU1(atype, dtype, [1,1,-1,-1], 4,4,4,4)
	Btensor = U1tensor2tensor(B)
	@test Array(ein"abab -> "(Btensor))[] ≈ dtr(B)

	# VUMPS unit
	d = 4
    D = 5
    AL = randU1(atype, dtype, [-1,1,1], D,d,D)
    M = randU1(atype, dtype, [-1,1,1,-1], d,d,d,d)
    FL = randU1(atype, dtype, [1,1,-1], D,d,D)
    tAL, tM, tFL = map(U1tensor2tensor,[AL, M, FL])
	tFL = ein"((adf,abc),dgeb),fgh -> ceh"(tFL,tAL,tM,conj(tAL))
	FL = ein"((adf,abc),dgeb),fgh -> ceh"(FL,AL,M,conj(AL))
    @test tFL ≈ U1tensor2tensor(FL) 

	# autodiff test
	D,d = 4,3
	FL = randU1(atype, dtype, [1,1,1], D, d, D)
	S = randU1(atype, dtype, [-1,-1,-1,-1,-1,-1], D, d, D, D, d, D)
	FLtensor = U1tensor2tensor(FL)
	Stensor = U1tensor2tensor(S)
	@test ein"(abc,abcdef),def ->"(FL, S, FL)[] ≈ ein"(abc,abcdef),def ->"(FLtensor, Stensor, FLtensor)[]
end

 @testset "inplace function with $symmetry $atype{$dtype}" for atype in [Array], dtype in [ComplexF64], symmetry in [:U1]
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

@testset "KrylovKit with $atype{$dtype}" for atype in [CuArray], dtype in [ComplexF64]
    Random.seed!(100)
    d = 3
    D = 5
    AL = randU1(atype, dtype, D, d, D)
    M = randU1(atype, dtype, d, d, d, d)
    FL = randU1(atype, dtype, D, d, D)
    tAL, tM, tFL = map(U1tensor2tensor,[AL, M, FL])
    λs, FLs, info = eigsolve(FL -> ein"((adf,abc),dgeb),fgh -> ceh"(FL,AL,M,conj(AL)), FL, 1, :LM; ishermitian = false)
    tλs, tFLs, info = eigsolve(tFL -> ein"((adf,abc),dgeb),fgh -> ceh"(tFL,tAL,tM,conj(tAL)), tFL, 1, :LM; ishermitian = false)
    @test λs ≈ tλs
    @test U1tensor2tensor(FLs[1]) ≈ tFLs[1]

    λl,FL = λs[1], FLs[1]
    dFL = randU1(atype, dtype, D, d, D)
    ξl, info = linsolve(FR -> ein"((ceh,abc),dgeb),fgh -> adf"(AL, FR, M, conj(AL)), dFL, -λl, 1)

    tλl,tFL = tλs[1], tFLs[1]
    tdFL = U1tensor2tensor(dFL)
    tξl, info = linsolve(tFR -> ein"((ceh,abc),dgeb),fgh -> adf"(tAL, tFR, tM, conj(tAL)), tdFL, -tλl, 1)
    @test U1tensor2tensor(ξl) ≈ tξl
end

@testset "U1 qr with $atype{$dtype}" for atype in [Array], dtype in [Float64]
    Random.seed!(100)
    A = randU1(atype, dtype, 5, 3, 5)
	Atensor = U1tensor2tensor(A)
	A = reshape(A, 15, 4) 
	Atensor = reshape(Atensor, 15, 5)
	Q, R = qrpos(A)
    Qtensor, Rtensor = qrpos(Atensor)
    @test Qtensor*Rtensor ≈ Atensor
	@test Q*R ≈ A
	@test U1tensor2tensor(reshape(Q, 5, 3, 5)) ≈ reshape(Qtensor, 5, 3, 5)
	@test U1tensor2tensor(R) ≈ Rtensor
end

@testset "U1 lq with $atype{$dtype}" for atype in [Array], dtype in [Float64]
    Random.seed!(100)
    A = randU1(atype, dtype, 4, 3, 4)
	Atensor = U1tensor2tensor(A)
	A = reshape(A, 4, 12)
	Atensor = reshape(Atensor, 4, 12)
	L, Q = lqpos(A)
    Ltensor, Qtensor = lqpos(Atensor)
    @test Ltensor*Qtensor ≈ Atensor
	@test L*Q ≈ A
	@test U1tensor2tensor(L) ≈ Ltensor
	@test U1tensor2tensor(reshape(Q, 4, 3, 4)) ≈ reshape(Qtensor, 4, 3, 4)
end

@testset "U1 svd with $atype{$dtype}" for atype in [Array, CuArray], dtype in [Float64, ComplexF64]
    Random.seed!(100)
    A = randU1(atype, dtype, 7,4)
	Atensor = U1tensor2tensor(A)
	U, S, V = sysvd!(copy(A))
    Utensor, Stensor, Vtensor = sysvd!(copy(Atensor))
    @test Utensor * Diagonal(Stensor) * Vtensor ≈ Atensor
	@test U * Diagonal(S) * V ≈ A
end