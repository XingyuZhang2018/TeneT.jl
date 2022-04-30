include("./exampletensors.jl")
include("./exampleobs.jl")

using CUDA
using LinearAlgebra: norm
using Random
using Test
using VUMPS: parity_conserving
using Zygote


@testset "$(Ni)x$(Nj) rand forward with $(symmetry) symmetry $atype array" for Ni = [2], Nj = [2], atype = [CuArray], symmetry in [:U1]
    Random.seed!(100)
    # T = asSymmetryArray(m, Val(symmetry); dir = [-1,-1,1,1,1])
    # T = randinitial(Val(symmetry), atype, ComplexF64, 2,2,4,2,2; dir = [-1,-1,1,1,1])
    d = 2
    D = 2
    χ = 10
    q = [1]
    T = atype(rand(ComplexF64, D,D,d,D,D))
    # T = T + permutedims(conj(T), [4,2,3,1,5])
    T = asSymmetryArray(T, Val(:U1); dir = [-1,-1,1,1,1], q=q)
    T = asArray(T)

    T = asSymmetryArray(T, Val(symmetry); dir = [-1,-1,1,1,1], q=q)
    m = ein"abcde, fgchi -> gbhdiefa"(T, conj(T))
    remori = asArray(m)
    mρ = ein"abcde, fgjhi -> gbhdiefajc"(T, conj(T))
    rem, reinfo = symmetryreshape(m, D^2,D^2,D^2,D^2)
    remρ, = symmetryreshape(mρ, D^2,D^2,D^2,D^2, d,d)
    β = 1
    M = [β * rem for i in 1:Ni, j in 1:Nj]
    env = obs_env(M; χ = χ, verbose = true, savefile = false, infolder = "./example/data/$(Ni)x$(Nj)rand/$symmetry/", outfolder = "./example/data/$(Ni)x$(Nj)rand/$symmetry/", maxiter = 10, miniter = 10, updown = false)
    ρmatrix(M, T, env, remρ)
    Zsymmetry = Z(env, M)
    @show Zsymmetry 

    # T = asArray(T)
    # m = ein"abcde, fgchi -> gbhdiefa"(T, conj(T))
    # @test remori ≈ m
    # rem, reinfo = symmetryreshape(m, D^2,D^2,D^2,D^2)
    # M = [β * rem for i in 1:Ni, j in 1:Nj]
    # env = obs_env(M; χ = χ, verbose = true, savefile = false, infolder = "./example/data/$(Ni)x$(Nj)rand/$(symmetry)_none/", outfolder = "./example/data/$(Ni)x$(Nj)rand/$(symmetry)_none/", maxiter = 10, miniter = 10, updown = false)
    # Znone = Z(env, M)
    # @show Znone
    # @show norm(Zsymmetry-Znone)
    # @test Zsymmetry ≈ Znone
end

@testset "$(Ni)x$(Nj) rand backward with $(symmetry) symmetry $atype array" for Ni = [1], Nj = [1], atype = [Array], symmetry in [:U1]
    Random.seed!(100)
    m = randinitial(Val(symmetry), atype, ComplexF64, 4, 4, 4, 4; dir = [-1,1,1,-1])
    function foo(β)
        M = [β * m for i in 1:Ni, j in 1:Nj]
        env = obs_env(M; χ = 10, verbose = true, savefile = true, infolder = "./example/data/$(Ni)x$(Nj)rand/$symmetry/", outfolder = "./example/data/$(Ni)x$(Nj)rand/$symmetry/", maxiter = 10, miniter = 10, updown = false)
        real(Z(env, M))
    end
    # @show foo(0.2)
    @show Zygote.gradient(foo, 0.2)
    # M = map(asArray, M)
    # @show Zygote.gradient(foo, 0.2)
end