using BenchmarkTools
using VUMPS
using CUDA
using Test
using OMEinsum
using Random
CUDA.allowscalar(false)

# a100-40g                                              |  # 2060-6g                                            
# none         76.190 μs (152 allocations: 7.56 KiB)    |  1.524 ms (157 allocations: 7.86 KiB)  
# Z2-hard     201.979 μs (616 allocations: 29.06 KiB)   |  846.100 μs (616 allocations: 29.09 KiB)
# Z2-OMEinsum 314.069 μs (1454 allocations: 75.80 KiB)  |  908.300 μs (1436 allocations: 75.02 KiB)
@testset "hard-coded with $symmetry $atype{$dtype} " for atype in [CuArray], dtype in [ComplexF64], symmetry in [:U1]
    Random.seed!(100)
    d = 8
    χ = 50
    FL = randinitial(Val(symmetry), atype, dtype, χ, d^2, χ; dir = [-1,1,1])
    M = randinitial(Val(symmetry), atype, dtype, d^2, d^2, d^2, d^2; dir = [-1,1,1,-1])
    AL = randinitial(Val(symmetry), atype, dtype, χ, d^2, χ; dir = [1,1,-1])

    function contract1(FL,AL)
        FLtensor = FL.tensor
        ALtensor = AL.tensor
        #result
        restensor = []
        
        # permutedims
        FLtensor .= map(x -> permutedims(x, [2, 3, 1]), FLtensor)
        # reshape
        FLtensor = map((x,y) ->reshape(x, y), FLtensor, [(56, 7), (168, 19), (168, 18), (152, 19), (456, 18), (456, 7), (144, 18), (432, 7), (432, 19)])
        ALtensor = map((x,y) ->reshape(x, y), ALtensor, [(7, 56), (18, 168), (19, 168), (19, 152), (7, 456), (18, 456), (18, 144), (19, 432), (7, 432)])
    
        # bulktimes
        # 0
        Amatrix = CUDA.zeros(ComplexF64, 944, 7) # direct initial in GPU
        Amatrix[1:56, :] .= FLtensor[1]
        Amatrix[57:512, :] .= FLtensor[6]
        Amatrix[513:944, :] .= FLtensor[8]
        Bmatrix = CUDA.zeros(ComplexF64, 7, 944)
        Bmatrix[:, 1:56] .= ALtensor[1]
        Bmatrix[:, 57:512] .= ALtensor[5]
        Bmatrix[:, 513:944] .= ALtensor[9]
    
        Cmatrix = Amatrix * Bmatrix
        push!(restensor, Cmatrix[1:56, 1:56])
        push!(restensor, Cmatrix[1:56, 57:512])
        push!(restensor, Cmatrix[1:56, 513:944])
        push!(restensor, Cmatrix[57:512, 1:56])
        push!(restensor, Cmatrix[57:512, 57:512])
        push!(restensor, Cmatrix[57:512, 513:944])
        push!(restensor, Cmatrix[513:944, 1:56])
        push!(restensor, Cmatrix[513:944, 57:512])
        push!(restensor, Cmatrix[513:944, 513:944])

        # 2
        Amatrix = CUDA.zeros(ComplexF64, 752, 19)
        Amatrix[1:168, :] .= FLtensor[2]
        Amatrix[169:320, :] .= FLtensor[4]
        Amatrix[321:752, :] .= FLtensor[9]
        Bmatrix = CUDA.zeros(ComplexF64, 19, 752)
        Bmatrix[:, 1:168] .= ALtensor[3]
        Bmatrix[:, 169:320] .= ALtensor[4]
        Bmatrix[:, 321:752] .= ALtensor[8]
    
        Cmatrix = Amatrix * Bmatrix
        push!(restensor, Cmatrix[1:168, 1:168])
        push!(restensor, Cmatrix[1:168, 169:320])
        push!(restensor, Cmatrix[1:168, 321:752])
        push!(restensor, Cmatrix[169:320, 1:168])
        push!(restensor, Cmatrix[169:320, 169:320])
        push!(restensor, Cmatrix[169:320, 321:752])
        push!(restensor, Cmatrix[321:752, 1:168])
        push!(restensor, Cmatrix[321:752, 169:320])
        push!(restensor, Cmatrix[321:752, 321:752])

        # 1
        Amatrix = CUDA.zeros(ComplexF64, 768, 18)
        Amatrix[1:168, :] .= FLtensor[3]
        Amatrix[169:624, :] .= FLtensor[5]
        Amatrix[625:768, :] .= FLtensor[7]
        Bmatrix = CUDA.zeros(ComplexF64, 18, 768)
        Bmatrix[:, 1:168] .= ALtensor[2]
        Bmatrix[:, 169:624] .= ALtensor[6]
        Bmatrix[:, 625:768] .= ALtensor[7]

        Cmatrix = Amatrix * Bmatrix
        push!(restensor, Cmatrix[1:168, 1:168])
        push!(restensor, Cmatrix[1:168, 169:624])
        push!(restensor, Cmatrix[1:168, 625:768])
        push!(restensor, Cmatrix[169:624, 1:168])
        push!(restensor, Cmatrix[169:624, 169:624])
        push!(restensor, Cmatrix[169:624, 625:768])
        push!(restensor, Cmatrix[625:768, 1:168])
        push!(restensor, Cmatrix[625:768, 169:624])
        push!(restensor, Cmatrix[625:768, 625:768])

        # reshape
        restensor = map((x,y) ->reshape(x, y...), restensor, [[8, 7, 8, 7], [8, 7, 24, 19], [8, 7, 24, 18], [24, 19, 8, 7], [24, 19, 24, 19], [24, 19, 24, 18], [24, 18, 8, 7], [24, 18, 24, 19], [24, 18, 24, 18], [24, 7, 24, 7], [24, 7, 8, 19], [24, 7, 24, 18], [8, 19, 24, 7], [8, 19, 8, 19], [8, 19, 24, 18], [24, 18, 24, 7], [24, 18, 8, 19], [24, 18, 24, 18], [24, 7, 24, 7], [24, 7, 24, 19], [24, 7, 8, 18], [24, 19, 24, 7], [24, 19, 24, 19], [24, 19, 8, 18], [8, 18, 24, 7], [8, 18, 24, 19], [8, 18, 8, 18]])
    
        # permutedims
        map(x -> permutedims(x, (2, 1, 3, 4)), restensor)
        # Z2tensor(resparity, restensor, ressize, resdims, 1)
    end

    res = ein"adf,abc -> fdbc"(FL,AL)
    # @test contract1(FL,AL) ≈ res.tensor
    @btime CUDA.@sync $contract1($FL,$AL)
    @btime CUDA.@sync ein"adf,abc -> fdbc"($FL,$AL)
end