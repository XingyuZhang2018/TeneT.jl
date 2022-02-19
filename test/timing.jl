using BenchmarkTools
using VUMPS
using VUMPS: qrpos,lqpos,leftorth,rightorth,leftenv,rightenv,ACenv,Cenv,LRtoC,ALCtoAC,ACCtoALAR,obs_FL,obs_FR
using KrylovKit
using CUDA
using Test
using OMEinsum
using Random
CUDA.allowscalar(false)

# d = 4, χ = 50 for CUDA@v3.8.0 RTX 2060-6G
# 9.839 ms (466 allocations: 24.27 KiB)
# Test Summary:                           |
# OMEinsum with none CuArray{ComplexF64}  | No tests
#   3.652 ms (5584 allocations: 271.00 KiB)
# Test Summary:                         |  
# OMEinsum with Z2 CuArray{ComplexF64}  | No tests

# d = 4, χ = 50 for CUDA@v3.5.0 RTX 2060-6G
# 9.834 ms (441 allocations: 22.88 KiB)
# Test Summary:                           |
# OMEinsum with none CuArray{ComplexF64}  | No tests
#   3.586 ms (5510 allocations: 265.25 KiB)
# Test Summary:                         |
# OMEinsum with Z2 CuArray{ComplexF64}  | No tests
@testset "OMEinsum with $symmetry $atype{$dtype} " for atype in [CuArray], dtype in [ComplexF64], symmetry in [:none, :Z2]
    Random.seed!(100)
    d = 4
    χ = 50
    FL = randinitial(Val(symmetry), atype, dtype, χ, d^2, χ)
    M = randinitial(Val(symmetry), atype, dtype, d^2, d^2, d^2, d^2)
    AL = randinitial(Val(symmetry), atype, dtype, χ, d^2, χ)
    # @time CUDA.@sync ein"((adf,abc),dgeb),fgh -> ceh"(FL,AL,M,conj(AL))
    @btime CUDA.@sync ein"((adf,abc),dgeb),fgh -> ceh"($FL,$AL,$M,conj($AL))
end

# d = 4, χ = 50 for CUDA@v3.8.0 RTX 2060-6G
# 360.902 ms (15038 allocations: 874.06 KiB)
# 129.586 ms (7066 allocations: 388.33 KiB)
# Test Summary:                           |
# KrylovKit with none CuArray{ComplexF64} | No tests
# 350.291 ms (197057 allocations: 9.04 MiB)
# 96.013 ms (81128 allocations: 3.85 MiB)
# Test Summary:                         |
# KrylovKit with Z2 CuArray{ComplexF64} | No tests

# d = 4, χ = 50 for CUDA@v3.5.0 RTX 2060-6G
# 404.415 ms (14390 allocations: 829.30 KiB)
# 141.568 ms (6764 allocations: 366.58 KiB)
# Test Summary:                           |
# KrylovKit with none CuArray{ComplexF64} | No tests
# 361.144 ms (195064 allocations: 8.88 MiB)
# 95.778 ms (80084 allocations: 3.76 MiB)
# Test Summary:                         |
# KrylovKit with Z2 CuArray{ComplexF64} | No tests
@testset "KrylovKit with $symmetry $atype{$dtype}" for atype in [CuArray], dtype in [ComplexF64], symmetry in [:none, :Z2]
    Random.seed!(100)
    d = 4
    χ = 50
    FL = randinitial(Val(symmetry), atype, dtype, χ, d^2, χ)
    M = randinitial(Val(symmetry), atype, dtype, d^2, d^2, d^2, d^2)
    AL = randinitial(Val(symmetry), atype, dtype, χ, d^2, χ)
    # @time CUDA.@sync λs, FLs, info = eigsolve(FL -> ein"((adf,abc),dgeb),fgh -> ceh"(FL,AL,M,conj(AL)), FL, 1, :LM; ishermitian = false)
    @btime CUDA.@sync λs, FLs, info = eigsolve(FL -> ein"((adf,abc),dgeb),fgh -> ceh"(FL,$AL,$M,conj($AL)), $FL, 1, :LM; ishermitian = false)

    λs, FLs, info = eigsolve(FL -> ein"((adf,abc),dgeb),fgh -> ceh"(FL,AL,M,conj(AL)), FL, 1, :LM; ishermitian = false)
    λl, FL = λs[1], FLs[1]
    dFL = randinitial(Val(symmetry), atype, dtype, χ, d^2, χ)
    # @time CUDA.@sync ξl, info = linsolve(FR -> ein"((ceh,abc),dgeb),fgh -> adf"(AL, FR, M, conj(AL)), dFL, -λl, 1)
    @btime CUDA.@sync ξl, info = linsolve(FR -> ein"((ceh,abc),dgeb),fgh -> adf"($AL, FR, $M, conj($AL)), $dFL, -$λl, 1)
end

# d = 100 for CUDA@v3.8.0 RTX 2060-6G
#     3.020 ms (211 allocations: 11.56 KiB)
#     3.575 ms (255 allocations: 950.48 KiB)
#   Test Summary:                           |
#   qr and lq with none CuArray{ComplexF64} | No tests
#     3.256 ms (886 allocations: 46.38 KiB)
#     3.897 ms (1034 allocations: 520.47 KiB)
#   Test Summary:                         |
#   qr and lq with Z2 CuArray{ComplexF64} | No tests

# d = 100 for CUDA@v3.5.0 RTX 2060-6G
#     2.845 ms (143 allocations: 6.94 KiB)
#     3.311 ms (197 allocations: 946.45 KiB)
#   Test Summary:                           |
#   qr and lq with none CuArray{ComplexF64} | No tests
#     3.011 ms (750 allocations: 37.12 KiB)
#     3.559 ms (918 allocations: 512.41 KiB)
#   Test Summary:                         |
#   qr and lq with Z2 CuArray{ComplexF64} | No tests
@testset "qr and lq with $symmetry $atype{$dtype}" for atype in [CuArray], dtype in [ComplexF64], symmetry in [:none, :Z2]
    Random.seed!(100)
    d = 100
    A = randinitial(Val(symmetry), atype, dtype, d, d)
    @btime CUDA.@sync qrpos($A)
    @btime CUDA.@sync lqpos($A)
end

# d = 4, χ = 50 for CUDA@v3.8.0 RTX 2060-6G

# d = 4, χ = 50 for CUDA@v3.5.0 RTX 2060-6G
# 1.828 s (354138 allocations: 23.37 MiB)
# 1.821 s (329325 allocations: 21.41 MiB)
# Test Summary:                                        |
# leftorth and rightorth with none CuArray{ComplexF64} | No tests
# 4.092 s (2174126 allocations: 96.30 MiB)
# 4.088 s (2175414 allocations: 96.37 MiB)
# Test Summary:                                      |
# leftorth and rightorth with Z2 CuArray{ComplexF64} | No tests
@testset "leftorth and rightorth with $symmetry $atype{$dtype}" for atype in [CuArray], dtype in [ComplexF64], symmetry in [:none, :Z2]
    Random.seed!(100)
    d = 4
    χ = 50
    A = [randinitial(Val(symmetry), atype, dtype, χ, d^2, χ) for i in 1:2, j in 1:2]
    @btime CUDA.@sync leftorth($A)
    @btime CUDA.@sync rightorth($A)
end

# d = 4, χ = 50 for CUDA@v3.8.1 RTX 2060-6G
#     2.765 s (128788 allocations: 9.17 MiB)
#     2.717 s (149190 allocations: 10.06 MiB)
#   Test Summary:                                      |
#   leftenv and rightenv with none CuArray{ComplexF64} | No tests
#     1.894 s (1592272 allocations: 77.18 MiB)
#     1.853 s (1645205 allocations: 80.13 MiB)
#   Test Summary:                                    |
#   leftenv and rightenv with Z2 CuArray{ComplexF64} | No tests

# d = 4, χ = 50 for CUDA@v3.5.0 RTX 2060-6G
# 3.286 s (137542 allocations: 10.15 MiB)
# 3.329 s (162111 allocations: 11.17 MiB)
# Test Summary:                                      |
# leftenv and rightenv with none CuArray{ComplexF64} | No tests
# 2.372 s (1510705 allocations: 72.73 MiB)
# 2.591 s (1562721 allocations: 75.65 MiB)
# Test Summary:                                    |
# leftenv and rightenv with Z2 CuArray{ComplexF64} | No tests
@testset "leftenv and rightenv with $symmetry $atype{$dtype}" for atype in [CuArray], dtype in [ComplexF64], symmetry in [:none, :Z2]
    Random.seed!(100)
    d = 4
    χ = 50
    A = [randinitial(Val(symmetry), atype, dtype, χ, d^2, χ) for i in 1:2, j in 1:2]
    M = [randinitial(Val(symmetry), atype, dtype, d^2, d^2, d^2, d^2) for i in 1:2, j in 1:2]

    AL, = leftorth(A)
    @btime CUDA.@sync leftenv($AL, $AL, $M)
    _, AR, = rightorth(A)
    @btime CUDA.@sync rightenv($AR, $AR, $M)
end

# d = 4, χ = 50 for CUDA@v3.8.1 RTX 2060-6G
#     2.703 s (114543 allocations: 6.02 MiB)
#     286.202 ms (78459 allocations: 4.34 MiB)
#   Test Summary:                                |
#   ACenv and Cenv with none CuArray{ComplexF64} | No tests
#     1.830 s (1553011 allocations: 71.37 MiB)
#     784.976 ms (679158 allocations: 31.28 MiB)
#   Test Summary:                              |
#   ACenv and Cenv with Z2 CuArray{ComplexF64} | No tests

# d = 4, χ = 50 for CUDA@v3.5.0 RTX 2060-6G
#     3.247 s (113480 allocations: 5.96 MiB)
#     358.911 ms (102818 allocations: 6.28 MiB)
#   Test Summary:                                |
#   ACenv and Cenv with none CuArray{ComplexF64} | No tests
#     2.293 s (1492782 allocations: 68.34 MiB)
#     832.162 ms (650838 allocations: 29.65 MiB)
#   Test Summary:                              |
#   ACenv and Cenv with Z2 CuArray{ComplexF64} | No tests
@testset "ACenv and Cenv with $(symmetry) $atype{$dtype}" for atype in [CuArray], dtype in [ComplexF64], symmetry in [:none, :Z2]
    Random.seed!(100)
    d = 4
    χ = 50
    A = [randinitial(Val(symmetry), atype, dtype, χ, d^2, χ) for i in 1:2, j in 1:2]
    M = [randinitial(Val(symmetry), atype, dtype, d^2, d^2, d^2, d^2) for i in 1:2, j in 1:2]

    AL, L = leftorth(A)
    λL,FL = leftenv(AL, AL, M)
    R, AR, = rightorth(A)
    λR,FR = rightenv(AR, AR, M)

    C = LRtoC(L, R)
    AC = ALCtoAC(AL, C)

    @btime CUDA.@sync ACenv($AC, $FL, $M, $FR)
    @btime CUDA.@sync Cenv($C, $FL, $FR)
end