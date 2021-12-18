using VUMPS
using VUMPS: AbstractZ2Array
using VUMPS: parity_conserving,Z2tensor,Z2tensor2tensor,qrpos,lqpos
using CUDA
using KrylovKit
using LinearAlgebra
using OMEinsum
using SparseArrays
using Random
using Test
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

@testset "Z2 Tensor with $atype{$dtype}" for atype in [Array, CuArray], dtype in [Float64, ComplexF64]
	Random.seed!(100)
	@test Z2tensor <: AbstractZ2Array <: AbstractArray

	A = randZ2(atype, dtype, 3,3,4)
	Atensor = Z2tensor2tensor(A)
	@test A isa Z2tensor

	# permutedims
	@test permutedims(Atensor,[3,2,1]) == Z2tensor2tensor(permutedims(A,[3,2,1]))

	## reshape
	@test reshape(Atensor,(9,4)) == reshape(Z2tensor2tensor(reshape(reshape(A,9,4),3,3,4)),(9,4))
end

@testset "OMEinsum Z2 with $atype{$dtype}" for atype in [CuArray], dtype in [Float64]
	Random.seed!(100)
	A = randZ2(atype, dtype, 3,3,4);
	B = randZ2(atype, dtype, 4,3);
	Atensor = Z2tensor2tensor(A)
	Btensor = Z2tensor2tensor(B)
	## binary contraction

	@test ein"abc,cd -> abd"(Atensor,Btensor) ≈ Z2tensor2tensor(ein"abc,cd -> abd"(A,B))
	@test ein"abc,db -> adc"(Atensor,Btensor) ≈ Z2tensor2tensor(ein"abc,db -> adc"(A,B))
	@test ein"cba,dc -> abd"(Atensor,Btensor) ≈ Z2tensor2tensor(ein"cba,dc -> abd"(A,B))
	@test ein"abc,cb -> a"(Atensor,Btensor) ≈ Z2tensor2tensor(ein"abc,cb -> a"(A,B))
	@test ein"bac,cb -> a"(Atensor,Btensor) ≈ Z2tensor2tensor(ein"bac,cb -> a"(A,B))
	@test ein"cba,ab -> c"(Atensor,Btensor) ≈ Z2tensor2tensor(ein"cba,ab -> c"(A,B))

	## NestedEinsum
	@test ein"(abc,cd),ed -> abe"(Atensor,Btensor,Btensor) ≈ Z2tensor2tensor(ein"abd,ed -> abe"(ein"abc,cd -> abd"(A,B),B)) ≈ Z2tensor2tensor(ein"(abc,cd),ed -> abe"(A,B,B))

	## constant
	@test Array(ein"abc,abc ->"(Atensor,Atensor))[] ≈ Array(ein"abc,abc ->"(A,A))[]

	## tr
	B = randZ2(atype, dtype, 4,4)
	Btensor = Z2tensor2tensor(B)
	@test Array(ein"aa ->"(Btensor))[] ≈ Array(ein"aa ->"(B))[]

	B = randZ2(atype, dtype, 2,2,2,2)
	Btensor = Z2tensor2tensor(B)
	@test Array(ein"abab -> "(Btensor))[] ≈ tr(reshape(B,4,4))
	@test Array(ein"aabb -> "(Btensor))[] ≈ Array(ein"aabb-> "(B))[]
	
	## VUMPS unit
	d = 2
    D = 10
    AL = randZ2(atype, dtype, D, d, D)
    M = randZ2(atype, dtype, d, d, d, d)
    FL = randZ2(atype, dtype, D, d, D)
    tAL, tM, tFL = map(Z2tensor2tensor,[AL, M, FL])
	tFL = ein"((adf,abc),dgeb),fgh -> ceh"(tFL,tAL,tM,conj(tAL))
	FL = ein"((adf,abc),dgeb),fgh -> ceh"(FL,AL,M,conj(AL))
    @test tFL ≈ Z2tensor2tensor(FL) 
end

@testset "KrylovKit with $atype{$dtype}" for atype in [CuArray], dtype in [ComplexF64]
    Random.seed!(100)
    d = 2
    D = 5
    AL = randZ2(atype, dtype, D, d, D)
    M = randZ2(atype, dtype, d, d, d, d)
    FL = randZ2(atype, dtype, D, d, D)
    tAL, tM, tFL = map(Z2tensor2tensor,[AL, M, FL])
    λs, FLs, info = eigsolve(FL -> ein"((adf,abc),dgeb),fgh -> ceh"(FL,AL,M,conj(AL)), FL, 1, :LM; ishermitian = false)
    tλs, tFLs, info = eigsolve(tFL -> ein"((adf,abc),dgeb),fgh -> ceh"(tFL,tAL,tM,conj(tAL)), tFL, 1, :LM; ishermitian = false)
    @test λs ≈ tλs
    @test Z2tensor2tensor(FLs[1]) ≈ tFLs[1]

    λl,FL = λs[1],FLs[1]
    dFL = randZ2(atype, dtype, D, d, D)
    ξl, info = linsolve(FR -> ein"((ceh,abc),dgeb),fgh -> adf"(AL, FR, M, conj(AL)), dFL, -λl, 1)

    tλl,tFL = tλs[1],tFLs[1]
    tdFL = Z2tensor2tensor(dFL)
    tξl, info = linsolve(tFR -> ein"((ceh,abc),dgeb),fgh -> adf"(tAL, tFR, tM, conj(tAL)), tdFL, -tλl, 1)
    @test Z2tensor2tensor(ξl) ≈ tξl
end

@testset "Z2 qr with $atype{$dtype}" for atype in [Array, CuArray], dtype in [Float64, ComplexF64]
    Random.seed!(100)
    A = randZ2(atype, dtype, 7,4)
	Atensor = Z2tensor2tensor(A)
	Q, R = qrpos(A)
    Qtensor, Rtensor = qrpos(Atensor)
    @test Array(Qtensor*Rtensor) ≈ Array(Atensor)
	@test Q*R ≈ A
	@test Z2tensor2tensor(Q) ≈ Qtensor
	@test Z2tensor2tensor(R) ≈ Rtensor
end

@testset "Z2 lq with $atype{$dtype}" for atype in [Array, CuArray], dtype in [Float64, ComplexF64]
    Random.seed!(100)
    A = randZ2(atype, dtype, 7,4)
	Atensor = Z2tensor2tensor(A)
	L, Q = lqpos(A)
    Ltensor, Qtensor = lqpos(Atensor)
    @test Array(Ltensor*Qtensor) ≈ Array(Atensor)
	@test L*Q ≈ A
	@test Z2tensor2tensor(L) ≈ Ltensor
	@test Z2tensor2tensor(Q) ≈ Qtensor
end