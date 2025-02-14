using Test
using TeneT

@testset "retract!" begin
    Random.seed!(100)
    χ = 5
    d = 2
    x = reshape([rand(ComplexF64, χ,d,χ)], (1,1))

    retract!(x)
    @test ein"abc,abd->cd"(x[1],conj(x[1])) ≈ I(χ)
end

@testset "project!" begin
    Random.seed!(100)
    χ = 5
    d = 2
    g = reshape([rand(ComplexF64, χ,d,χ)], (1,1))
    A = reshape([rand(ComplexF64, χ,d,χ)], (1,1))
    AL = retract!(A)
    project_AL!(g, AL)
    @test ein"abc,abd->cd"(g[1], conj(AL[1])) ≈ zeros(χ,χ) atol = 1e-12

    AR = permute_fronttail.(AL)
    @test ein"abc,dbc->ad"(AR[1],conj(AR[1])) ≈ I(χ)
    project_AR!(g, AR)
    @test ein"abc,dbc->ad"(g[1], conj(AR[1])) ≈ zeros(χ,χ) atol = 1e-12
end

@testset "project" begin
    Random.seed!(100)
    χ = 5
    d = 2
    g = reshape([rand(ComplexF64, χ,d,χ)], (1,1))
    A = reshape([rand(ComplexF64, χ,d,χ)], (1,1))
    AL = retract!(A)
    g = project_AL(g, AL)
    @test ein"abc,abd->cd"(g[1], conj(AL[1])) ≈ zeros(χ,χ) atol = 1e-12

    AR = permute_fronttail.(AL)
    @test ein"abc,dbc->ad"(AR[1],conj(AR[1])) ≈ I(χ)
    g = project_AR(g, AR)
    @test ein"abc,dbc->ad"(g[1], conj(AR[1])) ≈ zeros(χ,χ) atol = 1e-12
end