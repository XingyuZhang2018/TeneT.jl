using CUDA
using MPI
using TeneT
using Random
using LinearAlgebra
using OptimKit
using Zygote
using Printf
using TeneT: cannon_grid

MPI.Init()
const COMM = MPI.COMM_WORLD
const RANK = MPI.Comm_rank(COMM)
const NPROCS = MPI.Comm_size(COMM)
const NGRID = isqrt(NPROCS)
@assert NGRID * NGRID == NPROCS "Cannon needs a square process count; got nprocs=$NPROCS"

say(msg) = (RANK == 0 && (println(msg); flush(stdout)))
geti(k, default) = parse(Int, get(ENV, k, string(default)))
getf(k, default) = parse(Float64, get(ENV, k, string(default)))
getb(k, default) = lowercase(get(ENV, k, string(default))) in ("1", "true", "yes", "y")

function ckpt_method(k, default)
    s = lowercase(get(ENV, k, default))
    if s in ("plain", "none")
        return TeneT.Plain()
    elseif s == "recompute"
        return TeneT.Recompute()
    elseif s in ("offload", "offload_recompute")
        return TeneT.OffloadRecompute()
    else
        error("unknown checkpoint method for $k: $s")
    end
end

const D = geti("D", 16)
const J2 = getf("J2", 0.5)
const CHI_LOAD = geti("CHI_LOAD", 768)
const NO_LOAD = geti("NO_LOAD", 20)
const CHI_OPT = geti("CHI_OPT", 1024)
const OPT_MAXITER = geti("OPT_MAXITER", 20)
const VUMPS_MAXITER = geti("VUMPS_MAXITER", 30)
const MAXITER_AD = geti("MAXITER_AD", 4)
const MINITER_AD = geti("MINITER_AD", 4)
const POWER_ITER = geti("POWER_ITER", 5)
const POWER_ITER_AD = geti("POWER_ITER_AD", 5)
const POWER_ITER_OBS = geti("POWER_ITER_OBS", 20)
const SHOW_EVERY = geti("SHOW_EVERY", 1)
const FORLOOP_ITER = geti("FORLOOP_ITER", 32)
const SAVE_EVERY = geti("SAVE_EVERY", 1)
const ENV_TOL = getf("ENV_TOL", 1e-8)
const SEED = geti("SEED", 42)
const SAVE_ENV = getb("SAVE_ENV", false)
const DISTRIBUTED_QR = getb("DISTRIBUTED_QR", true)
const DATA_ROOT = get(ENV, "DATA_ROOT", joinpath(pkgdir(TeneT), "data"))

const atype = CuArray
const etype = Float64
const pattern = [1 3; 2 4]

Random.seed!(SEED)

function maybe_reclaim()
    GC.gc(true)
    try
        CUDA.reclaim()
    catch
    end
end

function restriction_ipeps(A)
    Ar = Zygote.Buffer(A)
    Ar[:, :, :, :, :, 1] = A[:, :, :, :, :, 1]
    Ar[:, :, :, :, :, 1] += permutedims(Ar[:, :, :, :, :, 1], (4, 3, 2, 1, 5))
    Ar[:, :, :, :, :, 2] = permutedims(Ar[:, :, :, :, :, 1], (1, 4, 3, 2, 5))
    Ar[:, :, :, :, :, 3] = permutedims(Ar[:, :, :, :, :, 1], (3, 2, 1, 4, 5))
    Ar[:, :, :, :, :, 4] = permutedims(Ar[:, :, :, :, :, 1], (3, 4, 1, 2, 5))
    return copy(Ar)
end

function make_boundary()
    alg = VUMPS{Plaquette{Square}}(
        ifsimple_eig=true,
        distributed_qr=DISTRIBUTED_QR,
        ifparallel=false,
        forloop_iter=FORLOOP_ITER,
        maxiter=VUMPS_MAXITER,
        miniter=0,
        maxiter_ad=MAXITER_AD,
        miniter_ad=MINITER_AD,
        power_iter=POWER_ITER,
        power_iter_ad=POWER_ITER_AD,
        power_iter_obs=POWER_ITER_OBS,
        show_every=SHOW_EVERY,
        tol=ENV_TOL,
        step_checkpoint=ckpt_method("STEP_CKPT", "recompute"),
        verbosity=(RANK == 0 ? 3 : 0),
    )
    alg.grid = cannon_grid(NGRID, NGRID)
    return alg
end

function make_params(model, folder)
    GradientOptimize(
        model=model,
        pattern=pattern,
        boundary_alg=make_boundary(),
        optimizer=LBFGS(200; maxiter=OPT_MAXITER,
                        verbosity=(RANK == 0 ? 4 : 0),
                        gradtol=1e-7,
                        linesearch=HagerZhangLineSearch(maxfg=5)),
        maxiter_restart=1,
        folder=folder,
        show_every=SHOW_EVERY,
        save_every=SAVE_EVERY,
        verbosity=(RANK == 0 ? 4 : 0),
        ifSU=false,
        SUτ=0,
        ifprecondition=false,
        iter_precond=0,
        bond_checkpoint=ckpt_method("BOND_CKPT", "recompute"),
        reuse_env=true,
        ifsave_env=SAVE_ENV,
        ifload_env=false,
        ifsave_lbfgs=false,
        ifload_lbfgs=false,
        ifplot=false,
    )
end

function run_smoke(A, chi::Int, params::GradientOptimize; restriction_ipeps)
    Dloc = TeneT._ipeps_bond_dimension(A)
    rt = TeneT.initialize_env(A, Dloc, chi, params; restriction_ipeps)
    rtprime = deepcopy(rt)
    fdelta = [1.0, 1.0, 0.0, 0.0]
    MPI.Barrier(COMM)
    CUDA.synchronize()
    t0 = time()
    e, back = Zygote.pullback(A) do x
        TeneT._G_cache[] = nothing
        real(TeneT.energy(restriction_ipeps(x), rt, rtprime, fdelta, params))
    end
    CUDA.synchronize()
    say(@sprintf("SMOKE forward: energy_chi%d = %.15f; Eimag = %.3e; time = %.2f s",
                 chi, e, fdelta[4], time() - t0))
    maybe_reclaim()
    MPI.Barrier(COMM)
    t1 = time()
    g = back(one(e))[1]
    CUDA.synchronize()
    say(@sprintf("SMOKE backward: |g| = %.6e; time = %.2f s", norm(g), time() - t1))
    maybe_reclaim()
end

function optimise_ipeps_noobs(A, chi::Int, params::GradientOptimize; restriction_ipeps)
    Dloc = TeneT._ipeps_bond_dimension(A)
    say("=== initialize_env for AD optimize chi=$chi ===")
    rt = TeneT.initialize_env(A, Dloc, chi, params; restriction_ipeps)
    rtprime = deepcopy(rt)
    fdelta = [1.0, 1.0, 0.0, 0.0]

    function fenergy(x)
        TeneT._G_cache[] = nothing
        xr = restriction_ipeps(x)
        return real(TeneT.energy(xr, rt, rtprime, fdelta, params))
    end

    function fg(x)
        MPI.Barrier(COMM)
        CUDA.synchronize()
        t1 = time()
        e, vjp = Zygote.pullback(fenergy, x)
        CUDA.synchronize()
        say(@sprintf("forward calculation took %.2f s; energy_chi%d = %.15f; Eimag = %.3e",
                     time() - t1, chi, e, fdelta[4]))
        maybe_reclaim()

        MPI.Barrier(COMM)
        CUDA.synchronize()
        t2 = time()
        g = vjp(one(e))[1]
        CUDA.synchronize()
        say(@sprintf("backward calculation took %.2f s; |g| = %.6e",
                     time() - t2, norm(g)))
        maybe_reclaim()
        return e, g
    end

    t0 = time()
    state_path = joinpath(params.folder, "D$(Dloc)", "lbfgs_checkpoint")
    Aopt, e, eg, fgnum, history = TeneT.optimize_reload(
        fg, A, params.optimizer;
        resume_from=params.ifload_lbfgs ? joinpath(state_path, "chi$chi.jld2") : nothing,
        save_state_to=params.ifsave_lbfgs ? joinpath(state_path, "chi$chi.jld2") : nothing,
        save_every=params.save_every,
        precondition=(x, g) -> g,
        inner=TeneT._inner,
        finalize!=(x, f, g, iter) -> TeneT._finalize!(
            x, f, g, iter, rt, rtprime, Dloc, chi, params, t0, fdelta
        ),
    )
    return Aopt, e, eg, fgnum, history
end

function main()
    model = J1J2(lattice=Square(), S=0.5, J1=1.0, J2=J2,
                 ifrotate=true, couplingtype=:uniform, bondratio=1.0)
    folder = joinpath(DATA_ROOT, "$model", "$pattern", "VUMPS_Plaquette", "Float64", "seed$SEED")
    params = make_params(model, folder)

    say("=== Cannon Plaquette distributed_qr AD optimize ===")
    say("nprocs=$NPROCS grid=$(NGRID)x$(NGRID) D=$D J2=$J2 seed=$SEED")
    say("load chi=$CHI_LOAD No.$NO_LOAD -> optimize chi=$CHI_OPT OPT_MAXITER=$OPT_MAXITER")
    say("VUMPS_MAXITER=$VUMPS_MAXITER MAXITER_AD=$MAXITER_AD MINITER_AD=$MINITER_AD")
    say("power_iter=$POWER_ITER power_iter_ad=$POWER_ITER_AD power_iter_obs=$POWER_ITER_OBS")
    say("forloop_iter=$FORLOOP_ITER tol=$ENV_TOL distributed_qr=$DISTRIBUTED_QR save_env=$SAVE_ENV")
    say("step_checkpoint=$(typeof(params.boundary_alg.step_checkpoint)) bond_checkpoint=$(typeof(params.bond_checkpoint))")
    say("folder=$folder")
    RANK == 0 && CUDA.functional() && say("rank0 GPU=$(CUDA.name(CUDA.device())) CUDA_LAUNCH_BLOCKING=$(get(ENV, "CUDA_LAUNCH_BLOCKING", "unset"))")

    A = init_ipeps(; atype, etype, No=NO_LOAD, D, χ=CHI_LOAD, params)
    MPI.Barrier(COMM)
    CUDA.synchronize()
    say("=== load complete ===")

    if OPT_MAXITER == 0
        run_smoke(A, CHI_OPT, params; restriction_ipeps)
    else
        Aopt, e, eg, fgnum, history = optimise_ipeps_noobs(A, CHI_OPT, params; restriction_ipeps)
        say("=== optimize complete ===")
        say("final_energy_chi$CHI_OPT = $e")
        say("final_gradient_norm = $(norm(eg))")
        say("fg_evaluations = $fgnum")
        say("history = $history")
        maybe_reclaim()
    end
end

try
    main()
catch err
    RANK == 0 && (showerror(stdout, err, catch_backtrace()); println(); flush(stdout))
    MPI.Abort(COMM, 1)
finally
    MPI.Finalize()
end
