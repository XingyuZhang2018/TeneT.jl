# Rectangular M4 smoke: env-level Slice2D gather hoisting on an N1xN2 grid.
using Test
using MPI
using LinearAlgebra
using Random
using TeneT
using TeneT: slice2d_grid, slice2d_scatter, slice2d_gather,
             leftenv, leftenv_slice2d, rightenv, rightenv_slice2d,
             ACenv, ACenv_slice2d, VUMPS, General, StructArray

MPI.Init()
const comm = MPI.COMM_WORLD
const rank = MPI.Comm_rank(comm)

const N1 = parse(Int, get(ENV, "TENET_SLICE2D_N1", "2"))
const N2 = parse(Int, get(ENV, "TENET_SLICE2D_N2", "3"))
@assert MPI.Comm_size(comm) == N1 * N2 "test_slice2d_m4_rect.jl expects N1*N2 ranks"

function build_cell(Ni, Nj, χ, D; d=2, seed=42, pattern=nothing)
    Random.seed!(seed)
    pat = pattern === nothing ? reshape(collect(1:Ni*Nj), Ni, Nj) : pattern
    nuniq = length(unique(pat))
    A1 = StructArray([rand(ComplexF64, χ, D, D, χ) for _ in 1:nuniq], pat)
    A2 = StructArray([rand(ComplexF64, χ, D, D, χ) for _ in 1:nuniq], pat)
    M  = StructArray([rand(ComplexF64, D, D, D, D, d) for _ in 1:nuniq], pat)
    E  = StructArray([rand(ComplexF64, χ, D, D, χ) for _ in 1:nuniq], pat)
    return A1, A2, M, E
end

scatter_sa(SA, g) = StructArray([slice2d_scatter(t, g) for t in SA.data], SA.pattern)
gather_sa(SA, g) = StructArray([slice2d_gather(t, g) for t in SA.data], SA.pattern)

function phase_ok(a_full, b_full; rtol=1e-7)
    an = a_full ./ norm(a_full)
    bn = b_full ./ norm(b_full)
    imax = argmax(abs.(bn))
    ph = an[imax] / bn[imax]
    return isapprox(abs(ph), 1; rtol) && isapprox(an, bn .* ph; rtol)
end

function check_env_pair(λc, Ec, λref, Eref, g; rtol=1e-7)
    Ec_full = gather_sa(Ec, g)
    for idx in 1:length(Eref.data)
        @test λc.data[idx] ≈ λref.data[idx] rtol=rtol
        @test phase_ok(Ec_full.data[idx], Eref.data[idx]; rtol)
    end
end

@testset "leftenv_slice2d rectangular parity" begin
    g = slice2d_grid(N1, N2)
    χ, D = 7, 2
    pat = [1 2; 2 1]
    ALu, ALd, M, FL = build_cell(2, 2, χ, D; seed=10100, pattern=pat)
    alg = VUMPS(General(); ifsimple_eig=true, power_iter=12, forloop_iter=2, verbosity=0)
    λref, FLref = leftenv(ALu, ALd, M, FL; alg)
    λc, FLc = leftenv_slice2d(scatter_sa(ALu, g), scatter_sa(ALd, g), M, scatter_sa(FL, g), g; alg)
    check_env_pair(λc, FLc, λref, FLref, g)
end

@testset "rightenv_slice2d rectangular parity" begin
    g = slice2d_grid(N1, N2)
    χ, D = 7, 2
    pat = [1 2; 2 1]
    ARu, ARd, M, FR = build_cell(2, 2, χ, D; seed=10200, pattern=pat)
    alg = VUMPS(General(); ifsimple_eig=true, power_iter=12, forloop_iter=2, verbosity=0)
    λref, FRref = rightenv(ARu, ARd, M, FR; alg)
    λc, FRc = rightenv_slice2d(scatter_sa(ARu, g), scatter_sa(ARd, g), M, scatter_sa(FR, g), g; alg)
    check_env_pair(λc, FRc, λref, FRref, g)
end

@testset "ACenv_slice2d rectangular parity" begin
    g = slice2d_grid(N1, N2)
    χ, D = 7, 2
    pat = [1 2; 2 1]
    AC, FL, M, FR = build_cell(2, 2, χ, D; seed=10300, pattern=pat)
    alg = VUMPS(General(); ifsimple_eig=true, power_iter=12, forloop_iter=2, verbosity=0)
    λref, ACref = ACenv(AC, FL, M, FR; alg)
    λc, ACc = ACenv_slice2d(scatter_sa(AC, g), scatter_sa(FL, g), M, scatter_sa(FR, g), g; alg)
    check_env_pair(λc, ACc, λref, ACref, g)
end

println("rank $rank: test_slice2d_m4_rect.jl done")
