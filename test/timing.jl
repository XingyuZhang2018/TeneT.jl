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

# d = 4, χ = 50 for CUDA@v3.5.0 a100-40G
#     254.449 μs (425 allocations: 21.91 KiB)
# Test Summary:                           |
# OMEinsum with none CuArray{ComplexF64}  | No tests
#     1.202 ms (5486 allocations: 264.09 KiB)
# Test Summary:                         |
# OMEinsum with Z2 CuArray{ComplexF64}  | No tests

# d = 4, χ = 50 for CUDA@v3.5.0 a100-80G
#   250.449 μs (416 allocations: 21.06 KiB)
# Test Summary:                           |
# OMEinsum with none CuArray{ComplexF64}  | No tests
#   1.201 ms (5486 allocations: 264.09 KiB)
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
    # @time CUDA.@sync ein"adf,abc -> fdbc"(FL,AL)
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

# d = 4, χ = 50 for CUDA@v3.5.0 a100-40G
# 40.639 ms (13910 allocations: 800.23 KiB)
# 7.259 ms (5419 allocations: 293.20 KiB)
# Test Summary:                           |
# KrylovKit with none CuArray{ComplexF64} | No tests
# 173.317 ms (194344 allocations: 8.85 MiB)
# 45.002 ms (79720 allocations: 3.74 MiB)
# Test Summary:                         |
# KrylovKit with Z2 CuArray{ComplexF64} | No tests

# d = 4, χ = 50 for CUDA@v3.5.0 a100-80G
# 42.264 ms (13910 allocations: 800.23 KiB)
# 7.426 ms (5421 allocations: 293.28 KiB)
# Test Summary:                           |
# KrylovKit with none CuArray{ComplexF64} | No tests
# 158.590 ms (194345 allocations: 8.85 MiB)
# 42.357 ms (79722 allocations: 3.74 MiB)
# Test Summary:                         |
# KrylovKit with Z2 CuArray{ComplexF64} | No tests
@testset "KrylovKit with $symmetry $atype{$dtype}" for atype in [CuArray], dtype in [ComplexF64], symmetry in [:none, :Z2]
    Random.seed!(100)
    d = 6
    χ = 100
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

# d = 100 for CUDA@v3.5.0 a100-40G
# 770.927 μs (143 allocations: 6.94 KiB)
# 1.010 ms (187 allocations: 945.86 KiB)
# Test Summary:                           |
# qr and lq with none CuArray{ComplexF64} | No tests
# 906.098 μs (750 allocations: 37.12 KiB)
# 1.142 ms (898 allocations: 511.22 KiB)
# Test Summary:                         |
# qr and lq with Z2 CuArray{ComplexF64} | No tests

# d = 100 for CUDA@v3.5.0 a100-80G
# 768.718 μs (143 allocations: 6.94 KiB)
# 1.039 ms (187 allocations: 945.86 KiB)
# Test Summary:                           |
# qr and lq with none CuArray{ComplexF64} | No tests
# 914.727 μs (750 allocations: 37.12 KiB)
# 1.167 ms (898 allocations: 511.22 KiB)
# Test Summary:                         |
# qr and lq with Z2 CuArray{ComplexF64} | No tests
@testset "qr and lq with $symmetry $atype{$dtype}" for atype in [CuArray], dtype in [ComplexF64], symmetry in [:none, :Z2]
    Random.seed!(100)
    d = 6
    χ = 100
    A = randinitial(Val(symmetry), atype, dtype, d^2*χ, χ)
    @btime CUDA.@sync qrpos($A)
    B = randinitial(Val(symmetry), atype, dtype, χ, d^2*χ)
    @btime CUDA.@sync lqpos($B)
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

# d = 4, χ = 50 for CUDA@v3.5.0 a100-40G
# 755.389 ms (349018 allocations: 23.17 MiB)
# 718.426 ms (323477 allocations: 21.16 MiB)
# Test Summary:                                        |
# leftorth and rightorth with none CuArray{ComplexF64} | No tests
# 1.782 s (2167114 allocations: 95.96 MiB)
# 1.781 s (2168596 allocations: 96.04 MiB)
# Test Summary:                                      |
# leftorth and rightorth with Z2 CuArray{ComplexF64} | No tests

# d = 4, χ = 50 for CUDA@v3.5.0 a100-80G
# 760.734 ms (349018 allocations: 23.17 MiB)
# 721.484 ms (323477 allocations: 21.16 MiB)
# Test Summary:                                        |
# leftorth and rightorth with none CuArray{ComplexF64} | No tests
# 1.792 s (2160864 allocations: 95.77 MiB)
# 1.787 s (2162328 allocations: 95.85 MiB)
# Test Summary:                                      |
# leftorth and rightorth with Z2 CuArray{ComplexF64} | No tests
@testset "leftorth and rightorth with $symmetry $atype{$dtype}" for atype in [CuArray], dtype in [ComplexF64], symmetry in [:none, :Z2]
    Random.seed!(100)
    d = 6
    χ = 100
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

# d = 4, χ = 50 for CUDA@v3.5.0 a100-40G
# 212.909 ms (109479 allocations: 8.26 MiB)
# 218.005 ms (125559 allocations: 8.99 MiB)
# Test Summary:                                      |
# leftenv and rightenv with none CuArray{ComplexF64} | No tests
# 918.450 ms (1452311 allocations: 70.92 MiB)
# 968.583 ms (1497911 allocations: 73.63 MiB)
# Test Summary:                                    |
# leftenv and rightenv with Z2 CuArray{ComplexF64} | No tests

# d = 4, χ = 50 for CUDA@v3.5.0 a100-80G
# 222.463 ms (109479 allocations: 8.26 MiB)
# 236.117 ms (125559 allocations: 8.99 MiB)
# Test Summary:                                      |
# leftenv and rightenv with none CuArray{ComplexF64} | No tests
# 930.803 ms (1452311 allocations: 70.92 MiB)
# 944.380 ms (1497911 allocations: 73.63 MiB)
# Test Summary:                                    |
# leftenv and rightenv with Z2 CuArray{ComplexF64} | No tests
@testset "leftenv and rightenv with $symmetry $atype{$dtype}" for atype in [CuArray], dtype in [ComplexF64], symmetry in [:none, :Z2]
    Random.seed!(100)
    d = 6
    χ = 100
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

# d = 4, χ = 50 for CUDA@v3.5.0 a100-40G
# 214.210 ms (105210 allocations: 5.60 MiB)
# 118.558 ms (75258 allocations: 4.16 MiB)
# Test Summary:                                |
# ACenv and Cenv with none CuArray{ComplexF64} | No tests
# 954.592 ms (1431126 allocations: 66.36 MiB)
# 385.743 ms (646510 allocations: 29.45 MiB)
# Test Summary:                              |
# ACenv and Cenv with Z2 CuArray{ComplexF64} | No tests

# d = 4, χ = 50 for CUDA@v3.5.0 a100-80G
# 213.190 ms (105210 allocations: 5.60 MiB)
# 118.246 ms (75258 allocations: 4.16 MiB)
# Test Summary:                                |
# ACenv and Cenv with none CuArray{ComplexF64} | No tests
# 867.181 ms (1431126 allocations: 66.36 MiB)
# 396.909 ms (646510 allocations: 29.45 MiB)
# Test Summary:                              |
# ACenv and Cenv with Z2 CuArray{ComplexF64} | No tests
@testset "ACenv and Cenv with $(symmetry) $atype{$dtype}" for atype in [CuArray], dtype in [ComplexF64], symmetry in [:none, :Z2]
    Random.seed!(100)
    d = 6
    χ = 100
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