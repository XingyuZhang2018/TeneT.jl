include("./exampletensors.jl")
include("./exampleobs.jl")

using Random
using Test
using CUDA
using LinearAlgebra: norm
using Zygote

@testset "$(Ni)x$(Nj) rand forward with $(symmetry) symmetry $atype array" for Ni = [1], Nj = [1], atype = [Array], symmetry in [:none, :Z2]
    Random.seed!(100)

    # rand M initial test, M can be changed to yours
    d, D, χ = 4, 2, 10
    T = atype(rand(ComplexF64, D,D,d,D,D))
    q = [0]
    T = asSymmetryArray(T, Val(:Z2); dir = [-1,-1,1,1,1], q = q)
    m = ein"abcde, fgchi -> gbhdiefa"(T, conj(T))
    rem, reinfo = symmetryreshape(m, D^2,D^2,D^2,D^2)
    rem = asArray(rem)
    β = 1
    M = [β * rem for i in 1:Ni, j in 1:Nj]

    # obs_env api
    M, ALu, Cu, ARu, ALd, Cd, ARd, FL, FR, FLu, FRu = obs_env(M, Val(symmetry); χ = χ, verbose = true, savefile = false, infolder = "./example/data/$(Ni)x$(Nj)rand/$symmetry/", outfolder = "./example/data/$(Ni)x$(Nj)rand/$symmetry/", maxiter = 10, miniter = 10, updown = false)
end