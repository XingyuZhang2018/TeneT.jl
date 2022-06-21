include("./exampletensors.jl")
include("./exampleobs.jl")

using CUDA
using LinearAlgebra: norm
using Random
using Test
using VUMPS: parity_conserving
using Zygote


@testset "$(Ni)x$(Nj) rand forward with $(symmetry) symmetry $atype array" for Ni = [2], Nj = [2], atype = [Array], symmetry in [:U1]
    Random.seed!(123)
    # T = asSymmetryArray(m, Val(symmetry); dir = [-1,-1,1,1,1])
    # T = randinitial(Val(symmetry), atype, ComplexF64, 2,2,4,2,2; dir = [-1,-1,1,1,1])
    d = 2
    D = 4
    χ = 20
    q = [0]
    T = atype(rand(ComplexF64, D,D,d,D,D))
    # T = T + permutedims(conj(T), [4,2,3,1,5])
    indqn = getqrange(D,D,d,D,D)
    indims = u1bulkdims(D,D,d,D,D)
    T = asSymmetryArray(T, Val(symmetry); dir = [-1,-1,1,1,1], indqn = indqn, indims = indims, q = q)
    T = asArray(T; indqn = indqn, indims = indims)

    T = asSymmetryArray(T, Val(symmetry); dir = [-1,-1,1,1,1], indqn = indqn, indims = indims, q = q)
    m = ein"abcde, fgchi -> gbhdiefa"(T, conj(T))
    remori = asArray(m; indqn = getqrange(D,D,D,D,D,D,D,D), indims = u1bulkdims(D,D,D,D,D,D,D,D))
    mρ = ein"abcde, fgjhi -> gbhdiefajc"(T, conj(T))
    reinfo = [], [], [], getqrange(D,D,D,D,D,D,D,D), u1bulkdims(D,D,D,D,D,D,D,D), [], [] 
    rem, reinfo = symmetryreshape(m, D^2,D^2,D^2,D^2; reinfo = reinfo)
    reinfo = [], [], [], getqrange(D,D,D,D,D,D,D,D,d,d), u1bulkdims(D,D,D,D,D,D,D,D,d,d), [], [] 
    remρ, = symmetryreshape(mρ, D^2,D^2,D^2,D^2, d, d; reinfo = reinfo)
    β = 1
    M = [β * rem for i in 1:Ni, j in 1:Nj]
    indD, indχ = getqrange(D, χ)
    dimsD, dimsχ = u1bulkdims(D, χ)

    indD, indχ = getqrange(D)[1], [-2, -1, 0, 1, 2]
    dimsD, dimsχ = [1, 1], [1, 5, 8, 5, 1]
    env = obs_env(M; χ = χ, verbose = true, savefile = false, infolder = "./example/data/$(Ni)x$(Nj)rand/$symmetry/", outfolder = "./example/data/$(Ni)x$(Nj)rand/$symmetry/", maxiter = 10, miniter = 10, updown = false, U1info = (indD, indχ, dimsD, dimsχ), show_every = 1)
    # ρmatrix(M, T, env, remρ)
    Zsymmetry = Z(env, M)
    @show Zsymmetry 

    # T = asArray(T)
    # m = ein"abcde, fgchi -> gbhdiefa"(T, conj(T))
    # @test remori ≈ m
    # rem, reinfo = symmetryreshape(m, D^2,D^2,D^2,D^2)
    # M = [β * rem for i in 1:Ni, j in 1:Nj]
    # env = obs_env(M; χ = χ, verbose = true, savefile = false, infolder = "./example/data/$(Ni)x$(Nj)rand/$(symmetry)_none/", outfolder = "./example/data/$(Ni)x$(Nj)rand/$(symmetry)_none/", maxiter = 10, miniter = 10, updown = false, U1info = (indD, indχ, dimsD, dimsχ))
    # Znone = Z(env, M)
    # @show Znone
    # @show norm(Zsymmetry-Znone)
    # @test Zsymmetry ≈ Znone
end

# @testset "$(Ni)x$(Nj) rand backward with $(symmetry) symmetry $atype array" for Ni = [1], Nj = [1], atype = [Array], symmetry in [:U1]
#     Random.seed!(100)
#     m = randinitial(Val(symmetry), atype, ComplexF64, 4, 4, 4, 4; dir = [-1,1,1,-1])
#     function foo(β)
#         M = [β * m for i in 1:Ni, j in 1:Nj]
#         env = obs_env(M; χ = 10, verbose = true, savefile = true, infolder = "./example/data/$(Ni)x$(Nj)rand/$symmetry/", outfolder = "./example/data/$(Ni)x$(Nj)rand/$symmetry/", maxiter = 10, miniter = 10, updown = false)
#         real(Z(env, M))
#     end
#     # @show foo(0.2)
#     @show Zygote.gradient(foo, 0.2)
#     # M = map(asArray, M)
#     # @show Zygote.gradient(foo, 0.2)
# end