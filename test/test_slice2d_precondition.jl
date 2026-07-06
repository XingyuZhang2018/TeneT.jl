# Slice2D distributed preconditioner gates (4 ranks, 2x2 grid, CPU).
# Run: julia --project=. test/run_test_slice2d_precondition.jl
using Test, MPI, LinearAlgebra, Random, OptimKit, Zygote
using TeneT
using TeneT: slice2d_grid, slice2d_scatter, StructArray,
             VUMPS, Plaquette, Square, J1J2, Heisenberg, GradientOptimize,
             PlaquetteVUMPSRuntime, Mumap, Mumap_slice2d_dist,
             precondition_invese_single_envir, init_ipeps, optimise_ipeps

MPI.Init()
const comm = MPI.COMM_WORLD
const rank = MPI.Comm_rank(comm)
@assert MPI.Comm_size(comm) == 4 "test_slice2d_precondition.jl expects exactly 4 ranks"

say(s) = (rank == 0 && (println(s); flush(stdout)))
scatter_sa(SA, g) = StructArray([slice2d_scatter(t, g) for t in SA.data], SA.pattern)
scatter_rt(rt, g) = PlaquetteVUMPSRuntime(scatter_sa(rt.AL, g), rt.C, scatter_sa(rt.FL, g))
relerr(a, b) = norm(a .- b) / max(norm(b), eps(real(eltype(b))))

function restriction_ipeps(A)
    Ar = Zygote.Buffer(A)
    Ar[:, :, :, :, :, 1] = A[:, :, :, :, :, 1]
    Ar[:, :, :, :, :, 1] += permutedims(Ar[:, :, :, :, :, 1], (4, 3, 2, 1, 5))
    Ar[:, :, :, :, :, 2] = permutedims(Ar[:, :, :, :, :, 1], (1, 4, 3, 2, 5))
    Ar[:, :, :, :, :, 3] = permutedims(Ar[:, :, :, :, :, 1], (3, 2, 1, 4, 5))
    Ar[:, :, :, :, :, 4] = permutedims(Ar[:, :, :, :, :, 1], (3, 4, 1, 2, 5))
    return copy(Ar)
end

function tiny_model()
    return J1J2(lattice=Square(), S=0.5, J1=1.0, J2=0.5,
                ifrotate=true, couplingtype=:uniform, bondratio=1.0)
end

function tiny_alg(grid)
    return VUMPS{Plaquette{Square}}(grid=grid,
                                    ifsimple_eig=true,
                                    ifparallel=false,
                                    forloop_iter=1,
                                    maxiter=1,
                                    miniter=0,
                                    maxiter_ad=1,
                                    miniter_ad=0,
                                    power_iter=1,
                                    power_iter_ad=1,
                                    power_iter_obs=1,
                                    show_every=1,
                                    tol=1e9,
                                    verbosity=0)
end

function tiny_params(grid; maxit=1, model=tiny_model())
    folder = rank == 0 ? mktempdir(; cleanup=false) : ""
    folder = MPI.bcast(folder, 0, comm)
    return GradientOptimize(model=model,
                            pattern=[1 3; 2 4],
                            boundary_alg=tiny_alg(grid),
                            optimizer=LBFGS(5; maxiter=maxit, verbosity=0, gradtol=1e-3),
                            folder=folder,
                            verbosity=0,
                            ifSU=false,
                            ifprecondition=true,
                            iter_precond=0,
                            reuse_env=false,
                            ifsave_env=false,
                            ifload_env=false,
                            ifsave_lbfgs=false,
                            ifload_lbfgs=false,
                            save_every=0,
                            ifplot=false,
                            forloop_iter=1)
end

@testset "Mumap_slice2d_dist forward parity" begin
    g = slice2d_grid(2, 2)
    Random.seed!(6100)
    chi, D, d = 8, 2, 2
    AC = rand(ComplexF64, chi, D, D, chi)
    ACd = rand(ComplexF64, chi, D, D, chi)
    FL = rand(ComplexF64, chi, D, D, chi)
    FR = rand(ComplexF64, chi, D, D, chi)
    Mu = rand(ComplexF64, D, D, D, D, d)

    ref = Mumap(AC, ACd, FL, FR, Mu)
    got = Mumap_slice2d_dist(slice2d_scatter(AC, g),
                             slice2d_scatter(ACd, g),
                             slice2d_scatter(FL, g),
                             slice2d_scatter(FR, g),
                             Mu,
                             g; forloop_iter=1)
    err = relerr(got, ref)
    say("  Mumap_slice2d_dist relerr = $err")
    @test err <= 1e-10
end

@testset "Plaquette Slice2D direct precondition call stays block-distributed" begin
    g = slice2d_grid(2, 2)
    Random.seed!(6200 + rank)
    D, d, chi = 2, 2, 8
    pattern = [1 3; 2 4]
    nu = maximum(pattern)
    params = tiny_params(g; maxit=1)

    A = rand(ComplexF64, D, D, D, D, d, nu)
    A ./= norm(vec(A))
    grad = rand(ComplexF64, size(A)...)
    AL = StructArray([rand(ComplexF64, chi, D, D, chi) for _ in 1:nu], pattern)
    C = StructArray([Matrix(qr(rand(ComplexF64, chi, chi)).Q) for _ in 1:nu], pattern)
    FL = StructArray([rand(ComplexF64, chi, D, D, chi) for _ in 1:nu], pattern)
    rt = scatter_rt(PlaquetteVUMPSRuntime(AL, C, FL), g)
    fdelta = [1.0, 1e-2, 1.0, 0.0]

    pg = precondition_invese_single_envir(A, grad, rt, params, restriction_ipeps, fdelta, 0)
    @test size(pg) == size(grad)
    @test all(isfinite, real.(pg))
    @test all(isfinite, imag.(pg))
    @test norm(pg) > 0
end

@testset "Slice2D preconditioner rejects unsupported Plaquette models" begin
    g = slice2d_grid(2, 2)
    Random.seed!(6300 + rank)
    D, d, chi = 2, 2, 8
    pattern = [1 3; 2 4]
    nu = maximum(pattern)
    model = Heisenberg(lattice=Square(), S=0.5, Jx=1.0, Jy=1.0, Jz=1.0)
    params = tiny_params(g; maxit=1, model=model)

    A = rand(ComplexF64, D, D, D, D, d, nu)
    grad = rand(ComplexF64, size(A)...)
    AL = StructArray([rand(ComplexF64, chi, D, D, chi) for _ in 1:nu], pattern)
    C = StructArray([Matrix(qr(rand(ComplexF64, chi, chi)).Q) for _ in 1:nu], pattern)
    FL = StructArray([rand(ComplexF64, chi, D, D, chi) for _ in 1:nu], pattern)
    rt = scatter_rt(PlaquetteVUMPSRuntime(AL, C, FL), g)
    fdelta = [1.0, 1e-2, 1.0, 0.0]

    @test_throws ArgumentError precondition_invese_single_envir(A, grad, rt, params, restriction_ipeps, fdelta, 0)
end

@testset "Plaquette Slice2D optimise_ipeps smoke with ifprecondition=true" begin
    g = slice2d_grid(2, 2)
    Random.seed!(6400)
    params = tiny_params(g; maxit=1)
    A0 = init_ipeps(; atype=Array, etype=Float64, No=0, D=2, χ=8, params)
    Aopt, e_final, eg, fgnum, history = optimise_ipeps(A0, [8], params; restriction_ipeps)
    @test size(Aopt) == size(A0)
    @test isfinite(real(e_final))
    @test length(history) >= 1
end

say("all Slice2D preconditioner gates done.")
