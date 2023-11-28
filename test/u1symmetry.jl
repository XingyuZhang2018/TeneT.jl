using TeneT
using TeneT: randU1, zerosU1, IU1, qrpos, lqpos, svd!, initialA, zerosinitial, randU1DiagMatrix, invDiagU1Matrix
using CUDA
using KrylovKit
using LinearAlgebra
using OMEinsum
using SparseArrays
using Random
using Test
using BenchmarkTools
CUDA.allowscalar(false)

@testset "indextoqn and getblockdims" begin
    @test electronPn <: AbstractSiteType
    @test electronSz <: AbstractSiteType
    @test electronZ2 <: AbstractSiteType
    @test tJSz <: AbstractSiteType
    @test tJZ2 <: AbstractSiteType

    @test [indextoqn(electronPn(), i) for i in 1:8] == [0, 1, 1, 2, 1, 2, 2, 3]
    @test getqrange(electronPn(), 8) == [[0,1,2,3]]
    @test getblockdims(electronPn(), 8) == [[1,3,3,1]]

    @test [indextoqn(electronSz(), i) for i in 1:8] == [0, 1, -1, 0, 1, 2, 0, 1]
    @test getqrange(electronSz(), 8) == [[-1,0,1,2]]
    @test getblockdims(electronSz(), 8) == [[1,3,3,1]]

    @test [indextoqn(electronZ2(), i) for i in 1:8] == [0, 1, 1, 0, 1, 0, 0, 1]
    @test getqrange(electronZ2(), 8) == [[0,1]]
    @test getblockdims(electronZ2(), 8) == [[4,4]]

    @test [indextoqn(tJSz(), i) for i in 1:8] == [0, 1, -1, 1, 2, 0, -1, 0]
    @test getqrange(tJSz(), 8) == [[-1,0,1,2]]
    @test getblockdims(tJSz(), 8) == [[2,3,2,1]]

    @test [indextoqn(tJZ2(), i) for i in 1:8] == [0, 1, 1, 1, 0, 0, 1, 0]
    @test getqrange(tJZ2(), 8) == [[0,1]]
    @test getblockdims(tJZ2(), 8) == [[4,4]]
end

@testset "U1Array with $sitetype $atype{$dtype}" for atype in [Array], dtype in [ComplexF64], sitetype in [electronPn(),electronZ2(),tJZ2()]
    Random.seed!(100)
    # initial
    @test U1Array <: AbstractSymmetricArray <: AbstractArray

    randinial = randU1(sitetype, atype, dtype, 4,4,5; dir=[-1,1,1])
    @test randinial isa U1Array
    zeroinial = zerosU1(sitetype, atype, dtype, 4,4,5; dir=[-1,1,1])
    Iinial = IU1(sitetype, atype, dtype, 3; dir = [-1,1])
    @test size(randinial) == (4,4,5)
    @test size(zeroinial) == (4,4,5)
    @test size(Iinial) == (3,3)

    # asU1Array and asArray
	A = randU1(sitetype, atype, dtype, 4,4,5; dir=[-1,1,1])
	Atensor = asArray(sitetype, A)
    AA = asU1Array(sitetype, Atensor; dir=[-1,1,1])
    AAtensor = asArray(sitetype, AA)
    @test A ≈ AA
    @test Atensor ≈ AAtensor

	# # permutedims
	@test permutedims(Atensor,[3,2,1]) == asArray(sitetype, permutedims(A,[3,2,1]))

	# # reshape
	@test reshape(Atensor,(16,5)) == reshape(asArray(sitetype, reshape(reshape(A,16,5),4,4,5)),(16,5))
end

@testset "OMEinsum U1 with $sitetype $atype{$dtype}" for atype in [Array], dtype in [ComplexF64], sitetype in [electronPn(),electronZ2(),tJZ2()]
    Random.seed!(100)
    A = randU1(sitetype, atype, dtype, 3,3,4; dir=[1,1,-1])
    B = randU1(sitetype, atype, dtype, 4,3; dir=[1,-1])
    Atensor = asArray(sitetype, A)
    Btensor = asArray(sitetype, B)

    # binary contraction
    @test ein"abc,cd -> abd"(Atensor,Btensor) ≈ asArray(sitetype, ein"abc,cd -> abd"(A,B))
    @test ein"abc,db -> adc"(Atensor,Btensor) ≈ asArray(sitetype, ein"abc,db -> adc"(A,B))
    @test ein"cba,dc -> abd"(Atensor,Btensor) ≈ asArray(sitetype, ein"cba,dc -> abd"(A,B))
    @test ein"abc,cb -> a"(Atensor,Btensor) ≈ asArray(sitetype, ein"abc,cb -> a"(A,B))
    @test ein"bac,cb -> a"(Atensor,Btensor) ≈ asArray(sitetype, ein"bac,cb -> a"(A,B))
    @test ein"cba,ab -> c"(Atensor,Btensor) ≈ asArray(sitetype, ein"cba,ab -> c"(A,B))
    a = randU1(sitetype, atype, dtype, 3,7,5; dir=[1,-1,1])
    b = randU1(sitetype, atype, dtype, 7,5,3; dir=[1,-1,-1])
    c = ein"abc,bcd->ad"(a,b)
    # @show a b c
    atensor = asArray(sitetype, a)
    btensor = asArray(sitetype, b)
    ctensor = asArray(sitetype, c)
    @test ctensor ≈ ein"abc,bcd->ad"(atensor,btensor)

    # NestedEinsum
    C = randU1(sitetype, atype, dtype, 4,3; dir = [-1,1])
    Ctensor = asArray(sitetype, C)
    @test ein"(abc,cd),ed -> abe"(Atensor,Btensor,Ctensor) ≈ asArray(sitetype, ein"abd,ed -> abe"(ein"abc,cd -> abd"(A,B),C)) ≈ asArray(sitetype, ein"(abc,cd),ed -> abe"(A,B,C))

    # constant
    D = randU1(sitetype, atype, dtype, 3,3,4; dir = [-1,-1,1])
    Dtensor = asArray(sitetype, D)
    @test Array(ein"abc,abc ->"(Atensor,Dtensor))[] ≈ Array(ein"abc,abc ->"(A,D))[]

    # tr
    B = randU1(sitetype, atype, dtype, 4,4; dir = [1,-1])
    Btensor = asArray(sitetype, B)
    @test Array(ein"aa ->"(Btensor))[] ≈ Array(ein"aa ->"(B))[] 
    B = randU1(sitetype, atype, dtype, 4,4,4,4; dir = [-1,-1,1,1])
    Btensor = asArray(sitetype, B)
    @test Array(ein"abab -> "(Btensor))[] ≈ dtr(B)  

    # TeneT unit
    d = 4
    D = 10
    AL = randU1(sitetype, atype, dtype, D,d,D; dir = [-1,1,1])
    M = randU1(sitetype, atype, dtype, d,d,d,d; dir = [-1,1,1,-1])
    FL = randU1(sitetype, atype, dtype, D,d,D; dir = [1,1,-1])
    tAL, tM, tFL = map(x->asArray(sitetype, x), [AL, M, FL])
    tFL = ein"((adf,abc),dgeb),fgh -> ceh"(tFL,tAL,tM,conj(tAL))
    FL = ein"((adf,abc),dgeb),fgh -> ceh"(FL,AL,M,conj(AL))
    @test tFL ≈ asArray(sitetype, FL)
         
    # autodiff test
    D,d = 4,3
    FL = randU1(sitetype, atype, dtype, D, d, D; dir = [1,1,1])
    S = randU1(sitetype, atype, dtype, D, d, D, D, d, D; dir = [-1,-1,-1,-1,-1,-1])
    FLtensor = asArray(sitetype, FL)
    Stensor = asArray(sitetype, S)
    @test ein"(abc,abcdef),def ->"(FL, S, FL)[] ≈ ein"(abc,abcdef),def ->"(FLtensor, Stensor, FLtensor)[]
end

@testset "inplace function with $sitetype $atype{$dtype}" for atype in [Array], dtype in [ComplexF64], sitetype in [electronPn(),electronZ2(),tJZ2()]
    Random.seed!(100) 
    d = 2
    χ = 5

    # rmul!
    A = randU1(sitetype, atype, dtype, χ,d,χ; dir = [-1,1,1])  
    Acopy = copy(A)
    @test A*2.0 == rmul!(A, 2.0)
    @test A.tensor != Acopy.tensor

    # lmul!
    # A = randU1(sitetype, atype, dtype, χ,χ; dir = [1,-1])  
    # B = randU1(sitetype, atype, dtype, χ,χ; dir = [1,-1])  
    # Bcopy = copy(B)
    # @test A*B == lmul!(A, B) 
    # @test B.tensor != Bcopy.tensor

    # mul!
    A = randU1(sitetype, atype, dtype, χ,χ; dir = [1,-1])  
    Y = similar(A)
    Ycopy = copy(Y)
    @test A*2.0 == mul!(Y, A, 2.0)
    @test Y.tensor != Ycopy.tensor

    # axpy!
    A = randU1(sitetype, atype, dtype, χ,χ; dir = [1,-1])
    B = randU1(sitetype, atype, dtype, χ,χ; dir = [1,-1])
    Bcopy = copy(B)
    At = asArray(sitetype, A)
    Bt = asArray(sitetype, B)
    Bcopyt = asArray(sitetype, Bcopy)
    @test A*2.0 + B == axpy!(2.0, A, B) == B
    @test B.tensor != Bcopy.tensor
    @test Bt + 2.0*At == axpy!(2.0, At, Bt) == asArray(sitetype, axpy!(2.0, A, Bcopy))
end

@testset "KrylovKit with $atype{$dtype}" for atype in [Array], dtype in [ComplexF64], sitetype in [electronPn(),electronZ2(),tJZ2()]
    Random.seed!(100)
    χ, d = 5, 3
    AL = randU1(sitetype, atype, dtype, χ,d,χ; dir = [-1,1,1])
    M = randU1(sitetype, atype, dtype, d,d,d,d; dir = [-1,1,1,-1])
    FL = randU1(sitetype, atype, dtype, χ,d,χ; dir = [1,1,-1])
    tAL = asArray(sitetype, AL)
    tM = asArray(sitetype, M)
    tFL = asArray(sitetype, FL)

    λs, FLs, info = eigsolve(FL -> ein"((adf,abc),dgeb),fgh -> ceh"(FL,AL,M,conj(AL)), FL, 1, :LM; ishermitian = false)
    tλs, tFLs, info = eigsolve(tFL -> ein"((adf,abc),dgeb),fgh -> ceh"(tFL,tAL,tM,conj(tAL)), tFL, 1, :LM; ishermitian = false)
    @test λs[1] ≈ tλs[1]
    @test asArray(sitetype, FLs[1]) ≈ tFLs[1] 

    λl,FL = λs[1], FLs[1]
    dFL = randU1(sitetype, atype, dtype, χ,d,χ; dir = [1,1,-1])
    dFL -= Array(ein"abc,abc ->"(conj(FL), dFL))[] * FL
    ξl, info = linsolve(FR -> ein"((ceh,abc),dgeb),fgh -> adf"(FR, AL, M, conj(AL)), conj(dFL), -λl, 1) 
    tλl, tFL = tλs[1], tFLs[1]
    tdFL = asArray(sitetype, dFL)
    tξl, info = linsolve(tFR -> ein"((ceh,abc),dgeb),fgh -> adf"(tFR, tAL, tM, conj(tAL)), conj(tdFL), -tλl, 1)
    @test asArray(sitetype, ξl) ≈ tξl
end

@testset "U1 order-3 tensor qr with $sitetype $atype{$dtype}" for atype in [Array], dtype in [ComplexF64], sitetype in [electronPn(),electronZ2(),tJZ2()]
    Random.seed!(100)
    χ, D = 10, 4
    A = randU1(sitetype, atype, dtype, χ, D, χ; dir = [-1, 1, 1])
	Atensor = asArray(sitetype, A)
	A = reshape(A, χ * D, χ) 
	Atensor = reshape(Atensor, χ * D, χ)
	Q, R = qrpos(A)
    @test Q.qn == sort(Q.qn) 
    @test R.qn == sort(R.qn)
    Qtensor, Rtensor = qrpos(Atensor)
    @test Qtensor * Rtensor ≈ Atensor
	@test Q * R ≈ A

    Q = reshape(Q, χ,D,χ)
    R = reshape(R, χ,χ)
    A = reshape(A, χ,D,χ)
    @test ein"abc,cd->abd"(Q, R) ≈ A

    @test Qtensor' * Qtensor ≈ I(χ)
    M = ein"cda,cdb -> ab"(reshape(Q, χ, D, χ), conj(reshape(Q, χ, D, χ)))
    @test asArray(sitetype, M) ≈ I(χ)

	@test asArray(sitetype, reshape(Q, χ,D,χ)) ≈ reshape(Qtensor, χ,D,χ)
	@test asArray(sitetype, R) ≈ Rtensor
end 
 

@testset "invDiagU1Matrix with $sitetype $atype{$dtype}" for atype in [Array], dtype in [ComplexF64], sitetype in [electronPn(),electronZ2(),tJZ2()]
    Random.seed!(100)
    χ = 4
    A = randU1DiagMatrix(sitetype, atype, dtype, χ; dir = [-1, 1])   
    invA = invDiagU1Matrix(A)
    @test A * invA ≈ IU1(sitetype, atype, dtype, χ; dir = [-1, 1])
end      

@testset "U1 order-N tensor qr with $sitetype $atype{$dtype}" for atype in [Array], dtype in [ComplexF64], sitetype in [electronPn()]
    Random.seed!(100)
    χ, D = 4, 4
    A = randU1(sitetype, atype, dtype, χ, D, D, χ; dir = [-1,1,1,1])
	Atensor = asArray(sitetype, A)
	A = reshape(A, χ*D, χ*D) 
	Atensor = reshape(Atensor, χ*D, χ*D)
    
	Q, R = qrpos(A)
    @test Q * R ≈ A

    Q = reshape(Q, χ, D, D*χ)
    R = reshape(R, D*χ, D, χ)
    A = reshape(A, χ, D, D, χ)
	@test ein"abc,cde->abde"(Q, R) ≈ A
end

@testset "U1 lq with $atype{$dtype}" for atype in [Array], dtype in [ComplexF64], sitetype in [electronPn(),electronZ2(),tJZ2()]
    Random.seed!(100)
    χ, D = 10, 4
    A = randU1(sitetype, atype, dtype, χ, D, χ; dir = [-1, 1, 1])
	Atensor = asArray(sitetype, A)
	A = reshape(A, χ, χ*D)
	Atensor = reshape(Atensor, χ, χ*D)
	L, Q = lqpos(A)
    @test Q.qn == sort(Q.qn) 
    @test L.qn == sort(L.qn)
    Ltensor, Qtensor = lqpos(Atensor)
    @test Ltensor*Qtensor ≈ Atensor
	@test L*Q ≈ A

    @test Qtensor*Qtensor' ≈ I(χ)
    M = ein"acd,bcd -> ab"(reshape(Q, χ,D,χ),conj(reshape(Q, χ,D,χ)))
    @test asArray(sitetype, M) ≈ I(χ)

	@test asArray(sitetype, L) ≈ Ltensor
	@test asArray(sitetype, reshape(Q,  χ,D,χ)) ≈ reshape(Qtensor,  χ,D,χ)
end 

@testset "U1 svd with $atype{$dtype} $sitetype" for atype in [Array], dtype in [Float64], sitetype in [electronPn(),electronZ2(),tJZ2()]
    Random.seed!(100)
    χ = 20
    A = randU1(sitetype, atype, dtype, χ, χ; dir = [-1, 1])
	Atensor = asArray(sitetype, A)
	U, S, V = svd!(copy(A))
    Utensor, Stensor, Vtensor = svd!(copy(Atensor))
    @test Utensor * Diagonal(Stensor) * Vtensor' ≈ Atensor
	@test U * Diagonal(S) * V' ≈ A

    U, S, V = svd!(copy(A); trunc=10)
    @test sum(S.dims) == [10, 10]
end

@testset "general flatten reshape" for ifZ2 in [false]
    using TeneT: blockdiv
    # (D,D,D,D,D,D,D,D)->(D^2,D^2,D^2,D^2)
    D, χ = 4, 2
    # a = randinitial(Val(:U1), Array, ComplexF64, D,D,D,D,D,D,D,D; dir = [1,-1,-1,1,-1,1,1,-1])
    indqn = [[-1, 0, 1] for _ in 1:5]
    indims = [[1, 2, 1] for _ in 1:5]
    a = randU1(Array, ComplexF64, D,D,4,D,D; dir=[-1,-1,1,1,1], indqn=indqn, indims=indims, ifZ2=ifZ2)
    a = ein"abcde, fgchi -> gbhdiefa"(a, conj(a))

    # @show size.(a.tensor)[[1,2,3]] a.dims[[1,2,3]]
    indqn = [[-1, 0, 1] for _ in 1:8]
    indims = [[1, 2, 1] for _ in 1:8]
    rea, reinfo = U1reshape(a, D^2,D^2,D^2,D^2; reinfo = (nothing, nothing, nothing, indqn, indims, nothing, nothing))
    rerea = U1reshape(rea, D,D,D,D,D,D,D,D; reinfo = reinfo)[1]
    @test rerea ≈ a

    reinfo2 = U1reshapeinfo((D^2,D^2,D^2,D^2), (D,D,D,D,D,D,D,D), a.dir, indqn, indims, ifZ2)
    rerea2 = U1reshape(rea, D,D,D,D,D,D,D,D; reinfo = reinfo2)[1]
    @test rerea2 ≈ a

    # (χ,D,D,χ) -> (χ,D^2,χ)
    D, χ = 2, 5
    indqn = [[-2, -1, 0, 1, 2], [0, 1], [0, 1], [-2, -1, 0, 1, 2]]
    indims = [[1, 1, 1, 1, 1], [1, 1], [1, 1], [1, 1, 1, 1, 1]]
    a = randU1(Array, ComplexF64, χ,D,D,χ; dir=[-1,1,-1,1], indqn=indqn, indims=indims, ifZ2=ifZ2)
    rea, reinfo  = U1reshape(a, χ,D^2,χ; reinfo = (nothing, nothing, nothing, indqn, indims, nothing, nothing))
    rerea = U1reshape(rea, χ,D,D,χ; reinfo = reinfo)[1]
    @test rerea ≈ a

    reinfo2 = U1reshapeinfo((χ,D^2,χ), (χ,D,D,χ), a.dir, indqn, indims, ifZ2)
    rerea2 = U1reshape(rea, χ,D,D,χ; reinfo = reinfo2)[1]
    @test rerea2 ≈ a
end

@testset "general flatten reshape" for ifZ2 in [true]
    using TeneT: blockdiv
    # (D,D,D,D,D,D,D,D)->(D^2,D^2,D^2,D^2)
    D, χ = 4, 2
    # a = randinitial(Val(:U1), Array, ComplexF64, D,D,D,D,D,D,D,D; dir = [1,-1,-1,1,-1,1,1,-1])
    indqn = [[0, 1] for _ in 1:5]
    indims = [[1, 3] for _ in 1:5]
    a = randU1(Array, ComplexF64, D,D,4,D,D; dir=[-1,-1,1,1,1], indqn=indqn, indims=indims, ifZ2=ifZ2)
    a = ein"abcde, fgchi -> gbhdiefa"(a, conj(a))

    # @show size.(a.tensor)[[1,2,3]] a.dims[[1,2,3]]
    indqn = [[0, 1] for _ in 1:8]
    indims = [[1, 3] for _ in 1:8]
    rea, reinfo = U1reshape(a, D^2,D^2,D^2,D^2; reinfo = (nothing, nothing, nothing, indqn, indims, nothing, nothing))
    rerea = U1reshape(rea, D,D,D,D,D,D,D,D; reinfo = reinfo)[1]
    @test rerea ≈ a

    reinfo2 = U1reshapeinfo((D^2,D^2,D^2,D^2), (D,D,D,D,D,D,D,D), a.dir, indqn, indims, ifZ2)
    rerea2 = U1reshape(rea, D,D,D,D,D,D,D,D; reinfo = reinfo2)[1]
    @test rerea2 ≈ a

    # (χ,D,D,χ) -> (χ,D^2,χ)
    D, χ = 2, 5
    indqn = [[0, 1], [0, 1], [0, 1], [0, 1]]
    indims = [[2, 3], [1, 1], [1, 1], [2, 3]]
    a = randU1(Array, ComplexF64, χ,D,D,χ; dir=[-1,1,-1,1], indqn=indqn, indims=indims, ifZ2=ifZ2)
    rea, reinfo  = U1reshape(a, χ,D^2,χ; reinfo = (nothing, nothing, nothing, indqn, indims, nothing, nothing))
    rerea = U1reshape(rea, χ,D,D,χ; reinfo = reinfo)[1]
    @test rerea ≈ a

    reinfo2 = U1reshapeinfo((χ,D^2,χ), (χ,D,D,χ), a.dir, indqn, indims, ifZ2)
    rerea2 = U1reshape(rea, χ,D,D,χ; reinfo = reinfo2)[1]
    @test rerea2 ≈ a
end