using BenchmarkTools
using VUMPS
using VUMPS: qrpos,lqpos,leftorth,rightorth,leftenv,rightenv,ACenv,Cenv,LRtoC,ALCtoAC,ACCtoALAR,obs_FL,obs_FR
using KrylovKit
using CUDA
using LinearAlgebra: dot
using Test
using OMEinsum
using ProfileView
using Random
using LinearAlgebra
using CUDA:i32
LinearAlgebra.BLAS.set_num_threads(8)
CUDA.allowscalar(false)

@testset "OMEinsum with $symmetry $atype{$dtype} " for atype in [Array], dtype in [ComplexF64], symmetry in [:U1]
    Random.seed!(100)
    indD = [0, 1, 2]
    indχ = [-2, -1, 0, 1, 2]
    dimsD = [1, 2, 1]
    dimsχ = [1, 4, 6, 4, 1]    
    D = sum(dimsD)
    χ = sum(dimsχ)
    # for χ in [χ]
        println("D = $(D) χ = $(χ)")
        AL = symmetryreshape(randinitial(Val(symmetry), atype, dtype, χ, D, D, χ; dir = [-1, -1, 1, 1], indqn = [indχ, indD, indD, indχ], indims = [dimsχ, dimsD, dimsD, dimsχ]), χ, D^2, χ; reinfo = (nothing, nothing, nothing, [indχ, indD, indD, indχ], [dimsχ, dimsD, dimsD, dimsχ], nothing, nothing))[1]
        M = symmetryreshape(randinitial(Val(symmetry), atype, dtype, D, D, D, D, D, D, D, D; dir = [1,-1,-1,1,-1, 1, 1,-1], indqn = [indD for _ in 1:8], indims = [dimsD for _ in 1:8]), D^2, D^2, D^2, D^2; reinfo = (nothing, nothing, nothing, [indD for _ in 1:8], [dimsD for _ in 1:8], nothing, nothing))[1]
        FL = symmetryreshape(randinitial(Val(symmetry), atype, dtype, χ, D, D, χ; dir = [1, -1, 1, -1], indqn = [indχ, indD, indD, indχ], indims = [dimsχ, dimsD, dimsD, dimsχ]), χ, D^2, χ; reinfo = (nothing, nothing, nothing, [indχ, indD, indD, indχ], [dimsχ, dimsD, dimsD, dimsχ], nothing, nothing))[1]

        # @time CUDA.@sync ein"((adf,abc),dgeb),fgh -> ceh"(FL,AL,M,conj(AL))
        FLm = ein"((adf,abc),dgeb),fgh -> ceh"(FL,AL,M,conj(AL))
        @show FLm.qn sort(FLm.qn)
#         FLm2 = ein"((adf,abc),dgeb),fgh -> ceh"(FLm,AL,M,conj(AL))
#         @show indexin(FLm.qn, FL.qn) indexin(FLm2.qn, FL.qn) indexin(FLm2.qn, FLm.qn)
# # 
    #     # t = minimum(@benchmark(CUDA.@sync ein"((adf,abc),dgeb),fgh -> ceh"($FL,$AL,$M,conj($AL)))).time / 1e9
    #     # # @time CUDA.@sync ein"adf,abc -> fdbc"(FL,AL)
        # @btime CUDA.@sync ein"((adf,abc),dgeb),fgh -> ceh"($FL,$AL,$M,conj($AL))
    #     # message = "$D    $χ    $(round(t,digits=5))\n"
    #     # logfile = open("./timing/wang_contraction_$(atype)_$(symmetry)symmetry_d$(D)_.log", "a")
    #     # write(logfile, message)
    #     # close(logfile)
    # end
    # function profile_test(n)
    #     for _ = 1:n
    #         CUDA.@sync ein"((adf,abc),dgeb),fgh -> ceh"(FL,AL,M,conj(AL))
    #     end
    # end
    # ProfileView.@profview profile_test(1)
    # ProfileView.@profview profile_test(10)
    # a = atype(rand(dtype,100,100))
    # b = atype(rand(dtype,100,100))
    # @btime $a * $b
end

@testset "dot with $atype{$dtype} " for atype in [CuArray], dtype in [ComplexF64]
    Random.seed!(100)

    N = 100
    A = [atype(rand(dtype, N, N)) for _ in 1:20]
    B = [atype(rand(dtype, N, N)) for _ in 1:20]

    @btime CUDA.@sync dot(vcat(vec.($A)...), vcat(vec.($B)...))
    @btime CUDA.@sync dot(vcat(vec.($A)...), vcat(vec.($B)...))

end


 @testset "dot with $atype{$dtype} " for atype in [CuArray], dtype in [ComplexF64]
    Random.seed!(100)

    N = 100
    A = [atype(rand(dtype, N, N)) for _ in 1:10]
    B = [atype(rand(dtype, N, N)) for _ in 1:10]
    
    mydot1(A, B) = dot(vcat(map(x->@view(x[:]), A)...), vcat(map(x->@view(x[:]), B)...))
    function mydot2(A, B)
        sA = length.(A)
        sB = length.(B)
        At = CUDA.zeros(dtype, sum(sA))
        Bt = CUDA.zeros(dtype, sum(sB))
        for i in 1:length(sA)
            At[sum(sA[1:i-1])+1:sum(sA[1:i])] = @view(A[i][:])
            Bt[sum(sB[1:i-1])+1:sum(sB[1:i])] = @view(B[i][:])
        end
        dot(At, Bt)
    end
    mydot3(A, B) = dot(map(Array, A), map(Array, B))
    # @btime CUDA.@sync dot($A, $B)
    # @btime CUDA.@sync $mydot1($A, $B)
    # @test dot(A, B) ≈ mydot2(A, B)
    @btime CUDA.@sync $mydot3($A, $B)
end

@testset "dot with $atype{$dtype} " for atype in [CuArray], dtype in [ComplexF64]
    Random.seed!(100)
    indD = [0, 1, 2]
    indχ = [-2, -1, 0, 1, 2]
    dimsD = [1, 3, 4]
    dimsχ = [10, 20, 40, 20, 10]
    D = sum(dimsD)
    χ = sum(dimsχ)
    # for χ in [χ]
    println("D = $(D) χ = $(χ)")
    AL = symmetryreshape(randinitial(Val(:U1), atype, dtype, χ, D, D, χ; dir = [-1, -1, 1, 1], indqn = [indχ, indD, indD, indχ], indims = [dimsχ, dimsD, dimsD, dimsχ]), χ, D^2, χ; reinfo = (nothing, nothing, nothing, [indχ, indD, indD, indχ], [dimsχ, dimsD, dimsD, dimsχ], nothing, nothing))[1]

    @show AL.qn
    @show sort(AL.qn)
    # A = vcat(map(vec, AL.tensor)...)
    # @btime norm($A)
    # @btime norm(vcat(map(vec, $AL.tensor)...))
    # @btime dot($A, $A)
    # foo(A) = (A .= A[[1:100; 101:200; 201:end]]; dot(A, A))
    # @btime $foo($A)

    # function foo1(A)
    #     atensor = vcat(map(vec, A.tensor)...)
    #     dot(atensor, atensor)
    # end
    # function foo2(A)
    #     map(dot, A.tensor, A.tensor)
    # end
    # @test foo(AL) ≈ dot(AL, AL)

    # @btime CUDA.@sync dot($AL, $AL)
    # @btime CUDA.@sync $foo1($AL)
    # @btime CUDA.@sync $foo2($AL)
end



@testset "KrylovKit with $symmetry $atype{$dtype}" for atype in [CuArray], dtype in [ComplexF64], symmetry in [:U1]
    Random.seed!(100)
    indD = [0, 1, 2]
    indχ = [-2, -1, 0, 1, 2]
    dimsD = [1, 2, 1]
    dimsχ = [1, 4, 6, 4, 1]
    D = sum(dimsD)
    χ = sum(dimsχ)

    AL = symmetryreshape(randinitial(Val(symmetry), atype, dtype, χ, D, D, χ; dir = [-1, -1, 1, 1], indqn = [indχ, indD, indD, indχ], indims = [dimsχ, dimsD, dimsD, dimsχ]), χ, D^2, χ; reinfo = (nothing, nothing, nothing, [indχ, indD, indD, indχ], [dimsχ, dimsD, dimsD, dimsχ], nothing, nothing))[1]
    M = symmetryreshape(randinitial(Val(symmetry), atype, dtype, D, D, D, D, D, D, D, D; dir = [1,-1,-1,1,-1, 1, 1,-1], indqn = [indD for _ in 1:8], indims = [dimsD for _ in 1:8]), D^2, D^2, D^2, D^2; reinfo = (nothing, nothing, nothing, [indD for _ in 1:8], [dimsD for _ in 1:8], nothing, nothing))[1]
    FL = symmetryreshape(randinitial(Val(symmetry), atype, dtype, χ, D, D, χ; dir = [1, -1, 1, -1], indqn = [indχ, indD, indD, indχ], indims = [dimsχ, dimsD, dimsD, dimsχ]), χ, D^2, χ; reinfo = (nothing, nothing, nothing, [indχ, indD, indD, indχ], [dimsχ, dimsD, dimsD, dimsχ], nothing, nothing))[1]
    # ProfileView.@profview CUDA.@sync [eigsolve(FL -> ein"((adf,abc),dgeb),fgh -> ceh"(FL,AL,M,conj(AL)), FL, 1, :LM; ishermitian = false) for _ in 1:10]
    @btime CUDA.@sync λs, FLs, info = eigsolve(FL -> ein"((adf,abc),dgeb),fgh -> ceh"(FL,$AL,$M,conj($AL)), $FL, 1, :LM; ishermitian = false)

    # λs, FLs, info = eigsolve(FL -> ein"((adf,abc),dgeb),fgh -> ceh"(FL,AL,M,conj(AL)), FL, 1, :LM; ishermitian = false)
    # λl, FL = λs[1], FLs[1]
    # dFL = zero(FL)
    # dFL -= Array(ein"abc,abc ->"(conj(FL), dFL))[] * FL
    # @btime CUDA.@sync ξl, info = linsolve(FR -> ein"((ceh,abc),dgeb),fgh -> adf"(FR, $AL, $M, conj($AL)), conj($dFL), -$λl, 1) 
end

@testset "qr and lq with $symmetry $atype{$dtype}" for atype in [Array], dtype in [ComplexF64], symmetry in [:U1]
    Random.seed!(100)
    indχ = collect(-5:5)
    dimsχ = [10 for _ in 1:11]
    χ = sum(dimsχ)
    A = randinitial(Val(symmetry), atype, dtype, χ, χ; dir = [-1, 1], indqn = [indχ, indχ], indims = [dimsχ, dimsχ])
    # qrpos(A)
    @btime CUDA.@sync qrpos($A)
    @btime CUDA.@sync lqpos($A)
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