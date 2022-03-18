using BenchmarkTools
using VUMPS
using VUMPS: qrpos,lqpos,leftorth,rightorth,leftenv,rightenv,ACenv,Cenv,LRtoC,ALCtoAC,ACCtoALAR,obs_FL,obs_FR
using KrylovKit
using CUDA
using Test
using OMEinsum
using Random
CUDA.allowscalar(false)

# a100-40g                                              |  # 2060-6g                                            
# none         76.190 μs (152 allocations: 7.56 KiB)    |  1.524 ms (157 allocations: 7.86 KiB)  
# Z2-hard     401.979 μs (616 allocations: 29.06 KiB)   |  846.100 μs (616 allocations: 29.09 KiB)
# Z2-OMEinsum 314.069 μs (1454 allocations: 75.80 KiB)  |  908.300 μs (1436 allocations: 75.02 KiB)
@testset "hard-coded with $symmetry $atype{$dtype} " for atype in [CuArray], dtype in [ComplexF64], symmetry in [:Z2]
    Random.seed!(100)
    d = 4
    χ = 50
    FL = randinitial(Val(symmetry), atype, dtype, χ, d^2, χ; dir = [-1,1,1])
    M = randinitial(Val(symmetry), atype, dtype, d^2, d^2, d^2, d^2; dir = [-1,1,1,-1])
    AL = randinitial(Val(symmetry), atype, dtype, χ, d^2, χ; dir = [1,1,-1])

    function contractZ2(FL,AL)
        FLtensor = FL.tensor
        ALtensor = AL.tensor
        #result
        restensor = []
        
        # permutedims
        FLtensor .= map(x -> permutedims(x, [2, 3, 1]), FLtensor)[[1, 4, 2, 3]]
        # reshape
        FLtensor = map(x->reshape(x, 800, 25), FLtensor)
        ALtensor = map(x->reshape(x, 25, 800), ALtensor)
    
        # bulktimes
        # even 
        Amatrix = CUDA.zeros(ComplexF64, 1600, 25) # direct initial in GPU
        Amatrix[1:800, :] .= FLtensor[1]
        Amatrix[801:1600, :] .= FLtensor[2]
        Bmatrix = CUDA.zeros(ComplexF64, 25, 1600)
        Bmatrix[:, 1:800] .= ALtensor[1]
        Bmatrix[:, 801:1600] .= ALtensor[4]
    
        Cmatrix = Amatrix * Bmatrix
        push!(restensor, Cmatrix[1:800, 1:800])
        push!(restensor, Cmatrix[1:800, 801:1600])
        push!(restensor, Cmatrix[801:1600, 1:800])
        push!(restensor, Cmatrix[801:1600, 801:1600])
    
        # odd
        Amatrix[1:800, :] .= FLtensor[3]
        Amatrix[801:1600, :] .= FLtensor[4]
        Bmatrix[:, 1:800] .= ALtensor[2]
        Bmatrix[:, 801:1600] .= ALtensor[3]
    
        Cmatrix = Amatrix * Bmatrix
        push!(restensor, Cmatrix[1:800, 1:800])
        push!(restensor, Cmatrix[1:800, 801:1600])
        push!(restensor, Cmatrix[801:1600, 1:800])
        push!(restensor, Cmatrix[801:1600, 801:1600])
    
        # reshape
        restensor = map(x->reshape(x, 32, 25, 32, 25), restensor)
    
        # permutedims
        map(x -> permutedims(x, (2, 1, 3, 4)), restensor)[[1, 2, 3, 4, 7, 8, 5, 6]]
        # Z2Array(resparity, restensor, ressize, resdims, 1)
    end

    res = ein"adf,abc -> fdbc"(FL,AL)
    # @test contractZ2(FL,AL) ≈ res.tensor
    @btime CUDA.@sync $contractZ2($FL,$AL)
    @btime CUDA.@sync ein"adf,abc -> fdbc"($FL,$AL)
end