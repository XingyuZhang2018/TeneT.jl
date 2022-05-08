using BenchmarkTools
using VUMPS
using VUMPS: qrpos,lqpos,leftorth,rightorth,leftenv,rightenv,ACenv,Cenv,LRtoC,ALCtoAC,ACCtoALAR,obs_FL,obs_FR
using KrylovKit
using CUDA
using Test
using OMEinsum
using Random
CUDA.allowscalar(false)

@testset "OMEinsum with $symmetry $atype{$dtype} " for atype in [CuArray], dtype in [ComplexF64], symmetry in [:U1]
    Random.seed!(100)
    indD = [0, 1]
    indχ = [-1, 0, 1]
    dimsD = [1, 1]
    dimsχ = [1, 2, 1]
    D = sum(dimsD)
    χ = sum(dimsχ)
    for χ in [χ]
        println("D = $(D) χ = $(χ)")
        FL = symmetryreshape(randinitial(Val(symmetry), atype, dtype, χ, D, D, χ; dir = [-1, -1, 1, 1], indqn = [indχ, indD, indD, indχ], indims = [dimsχ, dimsD, dimsD, dimsχ]), χ, D^2, χ; reinfo = (nothing, nothing, nothing, [indχ, indD, indD, indχ], [dimsχ, dimsD, dimsD, dimsχ], nothing, nothing))[1]
        M = symmetryreshape(randinitial(Val(symmetry), atype, dtype, D, D, D, D, D, D, D, D; dir = [1,-1,-1,1,-1, 1, 1,-1], indqn = [indD for _ in 1:8], indims = [dimsD for _ in 1:8]), D^2, D^2, D^2, D^2; reinfo = (nothing, nothing, nothing, [indD for _ in 1:8], [dimsD for _ in 1:8], nothing, nothing))[1]
        AL = symmetryreshape(randinitial(Val(symmetry), atype, dtype, χ, D, D, χ; dir = [1, -1, 1, -1], indqn = [indχ, indD, indD, indχ], indims = [dimsχ, dimsD, dimsD, dimsχ]), χ, D^2, χ; reinfo = (nothing, nothing, nothing, [indχ, indD, indD, indχ], [dimsχ, dimsD, dimsD, dimsχ], nothing, nothing))[1]
        # @show FL.pn M.pn AL.pn
        # @time CUDA.@sync ein"((adf,abc),dgeb),fgh -> ceh"(FL,AL,M,conj(AL))
        # t = minimum(@benchmark(CUDA.@sync ein"((adf,abc),dgeb),fgh -> ceh"($FL,$AL,$M,conj($AL)))).time / 1e9
        # # @time CUDA.@sync ein"adf,abc -> fdbc"(FL,AL)
        @btime CUDA.@sync ein"((adf,abc),dgeb),fgh -> ceh"($FL,$AL,$M,conj($AL))
        # message = "$D    $χ    $(round(t,digits=5))\n"
        # logfile = open("./timing/wang_contraction_$(atype)_$(symmetry)symmetry_d$(D)_.log", "a")
        # write(logfile, message)
        # close(logfile)
    end

    # a = atype(rand(dtype,100,100))
    # b = atype(rand(dtype,100,100))
    # @btime $a * $b
end

@testset "KrylovKit with $symmetry $atype{$dtype}" for atype in [CuArray], dtype in [ComplexF64], symmetry in [:none]
    Random.seed!(100)
    D = 4
    χ = 50
    FL = randinitial(Val(symmetry), atype, dtype, χ, D^2, χ)
    M = randinitial(Val(symmetry), atype, dtype, D^2, D^2, D^2, D^2)
    AL = randinitial(Val(symmetry), atype, dtype, χ, D^2, χ)
    # @time CUDA.@sync λs, FLs, info = eigsolve(FL -> ein"((adf,abc),dgeb),fgh -> ceh"(FL,AL,M,conj(AL)), FL, 1, :LM; ishermitian = false)
    @btime CUDA.@sync λs, FLs, info = eigsolve(FL -> ein"((adf,abc),dgeb),fgh -> ceh"(FL,$AL,$M,conj($AL)), $FL, 1, :LM; ishermitian = false)

    λs, FLs, info = eigsolve(FL -> ein"((adf,abc),dgeb),fgh -> ceh"(FL,AL,M,conj(AL)), FL, 1, :LM; ishermitian = false)
    λl, FL = λs[1], FLs[1]
    dFL = randinitial(Val(symmetry), atype, dtype, χ, D^2, χ)
    # @time CUDA.@sync ξl, info = linsolve(FR -> ein"((ceh,abc),dgeb),fgh -> adf"(AL, FR, M, conj(AL)), dFL, -λl, 1)
    @btime CUDA.@sync ξl, info = linsolve(FR -> ein"((ceh,abc),dgeb),fgh -> adf"($AL, FR, $M, conj($AL)), $dFL, -$λl, 1)
end

@testset "qr and lq with $symmetry $atype{$dtype}" for atype in [CuArray], dtype in [ComplexF64], symmetry in [:none, :Z2]
    Random.seed!(100)
    D = 4
    χ = 50
    A = randinitial(Val(symmetry), atype, dtype, D^2*χ, χ)
    @btime CUDA.@sync qrpos($A)
    B = randinitial(Val(symmetry), atype, dtype, χ, D^2*χ)
    @btime CUDA.@sync lqpos($B)
end

@testset "leftorth and rightorth with $symmetry $atype{$dtype}" for atype in [CuArray], dtype in [ComplexF64], symmetry in [:none, :Z2]
    Random.seed!(100)
    D = 4
    χ = 50
    A = [randinitial(Val(symmetry), atype, dtype, χ, D^2, χ) for i in 1:2, j in 1:2]
    @btime CUDA.@sync leftorth($A)
    @btime CUDA.@sync rightorth($A)
end

@testset "leftenv and rightenv with $symmetry $atype{$dtype}" for atype in [CuArray], dtype in [ComplexF64], symmetry in [:none]
    Random.seed!(100)
    χ, D = 20, 4
    A = [symmetryreshape(randinitial(Val(symmetry), atype, dtype, χ,D,D,χ; dir = [-1, -1, 1, 1]), χ,D^2,χ)[1] for i in 1:Ni, j in 1:Nj]
    T = [randinitial(Val(symmetry), atype, dtype, D,D,4,D,D; dir = [-1,-1,1,1,1], q=[1]) for i in 1:Ni, j in 1:Nj]
    TT = [ein"abcde, fgchi -> gbhdiefa"(T[i,j], conj(T[i,j])) for i in 1:Ni, j in 1:Nj]
    M = [symmetryreshape(TT[i,j], D^2,D^2,D^2,D^2)[1] for i in 1:Ni, j in 1:Nj]

    AL, = leftorth(A)
    @btime CUDA.@sync leftenv($AL, $AL, $M)
    # _, AR, = rightorth(A)
    # @btime CUDA.@sync rightenv($AR, $AR, $M)
end

@testset "ACenv and Cenv with $(symmetry) $atype{$dtype}" for atype in [CuArray], dtype in [ComplexF64], symmetry in [:none]
    Random.seed!(100)
    D = 4
    χ = 50
    A = [randinitial(Val(symmetry), atype, dtype, χ, D^2, χ; dir = [1,1,-1]) for i in 1:2, j in 1:2]
    M = [randinitial(Val(symmetry), atype, dtype, D^2, D^2, D^2, D^2; dir = [-1,1,1,-1]) for i in 1:2, j in 1:2]

    AL, L = leftorth(A)
    λL,FL = leftenv(AL, AL, M)
    R, AR, = rightorth(A)
    λR,FR = rightenv(AR, AR, M)

    C = LRtoC(L, R)
    AC = ALCtoAC(AL, C)

    @btime CUDA.@sync ACenv($AC, $FL, $M, $FR)
    @btime CUDA.@sync Cenv($C, $FL, $FR)
end