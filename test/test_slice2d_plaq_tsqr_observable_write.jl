# Plaquette Slice2D default TSQR observable write smoke (4 ranks, 2x2 grid, CPU).
# This exercises observable() through write_obs_log in an isolated temp folder.
using Test, MPI, LinearAlgebra, Random, OptimKit, Zygote
using TeneT
using TeneT: J1J2, Square, VUMPS, Plaquette, GradientOptimize,
             slice2d_grid, observable

MPI.Init()
const comm = MPI.COMM_WORLD
const rank = MPI.Comm_rank(comm)
@assert MPI.Comm_size(comm) == 4 "test_slice2d_plaq_tsqr_observable_write.jl expects exactly 4 ranks"

say(s) = (rank == 0 && (println(s); flush(stdout)))

function restriction_ipeps(A)
    Ar = Zygote.Buffer(A)
    Ar[:, :, :, :, :, 1] = A[:, :, :, :, :, 1]
    Ar[:, :, :, :, :, 1] += permutedims(Ar[:, :, :, :, :, 1], (4, 3, 2, 1, 5))
    Ar[:, :, :, :, :, 2] = permutedims(Ar[:, :, :, :, :, 1], (1, 4, 3, 2, 5))
    Ar[:, :, :, :, :, 3] = permutedims(Ar[:, :, :, :, :, 1], (3, 2, 1, 4, 5))
    Ar[:, :, :, :, :, 4] = permutedims(Ar[:, :, :, :, :, 1], (3, 4, 1, 2, 5))
    return copy(Ar)
end

@testset "Plaquette Slice2D default TSQR observable writes log" begin
    D, d, chi = 2, 2, 8
    pattern = [1 3; 2 4]
    folder = rank == 0 ? mktempdir(; cleanup=false) : ""
    folder = MPI.bcast(folder, 0, comm)

    Random.seed!(9100)
    A = rand(ComplexF64, D, D, D, D, d, maximum(pattern))
    A ./= norm(vec(A))

    model = J1J2(lattice=Square(), S=0.5, J1=1.0, J2=0.5,
                 ifrotate=true, couplingtype=:uniform, bondratio=1.0)
    alg = VUMPS{Plaquette{Square}}(grid=slice2d_grid(2, 2),
                                   ifsimple_eig=true,
                                   ifparallel=false,
                                   forloop_iter=1,
                                   maxiter=1,
                                   miniter=0,
                                   maxiter_ad=0,
                                   miniter_ad=0,
                                   power_iter=1,
                                   power_iter_ad=1,
                                   power_iter_obs=1,
                                   show_every=1,
                                   tol=1e9,
                                   verbosity=0)
    params = GradientOptimize(model=model, pattern=pattern, boundary_alg=alg,
                              optimizer=LBFGS(5; maxiter=1, verbosity=0, gradtol=1e-3),
                              maxiter_restart=1,
                              folder=folder,
                              verbosity=0,
                              ifSU=false,
                              ifprecondition=false,
                              iter_precond=0,
                              reuse_env=true,
                              ifsave_env=false,
                              ifload_env=false,
                              ifsave_lbfgs=false,
                              ifload_lbfgs=false,
                              save_every=0,
                              ifplot=false,
                              forloop_iter=1)

    e, mag, xi = observable(A, chi, params; restriction_ipeps)
    MPI.Barrier(comm)

    if rank == 0
        obs_log = joinpath(folder, "D$D", "observable", "chi$chi.log")
        unicode_obs_log = joinpath(folder, "D$D", "observable", "χ$chi.log")
        path = isfile(unicode_obs_log) ? unicode_obs_log : obs_log
        say("  observable smoke folder = $folder")
        say("  observable smoke log = $path")

        @test isfile(path)
        body = read(path, String)
        @test occursin("energy_per_site:", body)
        @test occursin("magnetization_norm_per_site:", body)
        @test occursin("correlation_length:", body)
        @test filesize(path) > 0
        @test isfinite(real(e[1]))
        @test isfinite(real(mag[1]))
        @test !isnan(real(xi))
    end

    e_skip, mag_skip, xi_skip = observable(A, chi, params; restriction_ipeps, cor_len_method=:none)
    MPI.Barrier(comm)

    if rank == 0
        obs_log = joinpath(folder, "D$D", "observable", "chi$chi.log")
        unicode_obs_log = joinpath(folder, "D$D", "observable", "χ$chi.log")
        path = isfile(unicode_obs_log) ? unicode_obs_log : obs_log
        body = read(path, String)

        @test occursin("energy_per_site:", body)
        @test occursin("magnetization_norm_per_site:", body)
        @test occursin("correlation_length:\nNaN", body)
        @test isfinite(real(e_skip[1]))
        @test isfinite(real(mag_skip[1]))
        @test xi_skip === nothing
    end
end

rank == 0 && println("Plaquette Slice2D default TSQR observable write smoke done.")
