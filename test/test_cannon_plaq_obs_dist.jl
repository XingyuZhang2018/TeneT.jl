# Plaquette Cannon distributed observable post-measurement smoke.
# Run via: julia --project=. test/run_test_cannon_plaq_obs_dist.jl
using Test, MPI, LinearAlgebra, Random, Printf
using TeneT
using TeneT: cannon_grid, cannon_scatter, VUMPS, Plaquette, Square, J1J2,
             StructArray, PlaquetteVUMPSEnv, GradientOptimize,
             magnetization_value, cor_len_value

MPI.Init()
const comm = MPI.COMM_WORLD
const rank = MPI.Comm_rank(comm)
@assert MPI.Comm_size(comm) == 4 "test_cannon_plaq_obs_dist.jl expects exactly 4 ranks"
say(s) = (rank == 0 && (println(s); flush(stdout)))

scatter_sa(SA, g) = StructArray([cannon_scatter(t, g) for t in SA.data], SA.pattern)
relerr(a, b) = abs(a - b) / max(abs(b), eps(real(eltype([b]))))

@testset "Plaquette Cannon mag/xi run on block env" begin
    g = cannon_grid(2, 2)
    chi, D, d = 8, 2, 2
    pat = [1 3; 2 4]
    nu = length(unique(pat))
    Random.seed!(9500)
    sa(dims) = StructArray([rand(ComplexF64, dims...) for _ in 1:nu], pat)

    A = sa((D, D, D, D, d))
    AL = sa((chi, D, D, chi))
    C = StructArray([Matrix(qr(rand(ComplexF64, chi, chi)).Q) for _ in 1:nu], pat)
    FLu = sa((chi, D, D, chi))
    FLo = sa((chi, D, D, chi))

    model = J1J2(lattice=Square(), S=0.5, J1=1.0, J2=0.5,
                 ifrotate=true, couplingtype=:uniform, bondratio=1.0)
    alg_s = VUMPS{Plaquette{Square}}(grid=nothing, ifsimple_eig=true,
                                     ifparallel=false, forloop_iter=1,
                                     power_iter=2, power_iter_obs=2,
                                     maxiter=1, maxiter_ad=0, verbosity=0)
    alg_c = VUMPS{Plaquette{Square}}(grid=g, ifsimple_eig=true,
                                     ifparallel=false, forloop_iter=1,
                                     power_iter=2, power_iter_obs=2,
                                     maxiter=1, maxiter_ad=0, verbosity=0)
    params_s = GradientOptimize(model=model, pattern=pat, boundary_alg=alg_s,
                                forloop_iter=1, ifplot=false)
    params_c = GradientOptimize(model=model, pattern=pat, boundary_alg=alg_c,
                                forloop_iter=1, ifplot=false)

    env_s = PlaquetteVUMPSEnv(AL, C, FLu, FLo)
    env_c = PlaquetteVUMPSEnv(scatter_sa(AL, g), C, scatter_sa(FLu, g), scatter_sa(FLo, g))

    mag_s = magnetization_value(model, A, env_s, params_s)
    mag_c = magnetization_value(model, A, env_c, params_c)
    xi_s = cor_len_value(env_s, params_s, A; method=:mps)
    xi_c = cor_len_value(env_c, params_c, A; method=:mps)

    if rank == 0
        say(@sprintf("  mag serial=%.12g cannon=%.12g", real(mag_s[1]), real(mag_c[1])))
        say(@sprintf("  xi  serial=%.12g cannon=%.12g", real(xi_s), real(xi_c)))
        @test relerr(real(mag_c[1]), real(mag_s[1])) <= 1e-10
        @test relerr(real(xi_c), real(xi_s)) <= 1e-8
    end
end

rank == 0 && println("Plaquette Cannon distributed observable post-measurement smoke done.")
