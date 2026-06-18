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
split_csv(s) = [strip(x) for x in split(s, ",") if !isempty(strip(x))]
getfs(k, default) = parse.(Float64, split_csv(get(ENV, k, default)))
getis(k, default) = parse.(Int, split_csv(get(ENV, k, default)))

const D = geti("D", 16)
const J2_LIST = getfs("J2_LIST", "0.5,0.52")
const CHI_LOAD = geti("CHI_LOAD", 768)
const NO_LOAD = geti("NO_LOAD", 20)
const CHI_OBS_LIST = getis("CHI_OBS_LIST", "768,1024,2048")
const XI_CHIS = Set(getis("XI_CHIS", "768"))
const SEED = geti("SEED", 42)
const VUMPS_MAXITER = geti("VUMPS_MAXITER", 20)
const SHOW_EVERY = geti("SHOW_EVERY", 1)
const POWER_ITER = geti("POWER_ITER", 5)
const POWER_ITER_OBS = geti("POWER_ITER_OBS", 20)
const FORLOOP_ITER = geti("FORLOOP_ITER", 32)
const ENV_TOL = getf("ENV_TOL", 1e-8)
const SAVE_ENV = getb("SAVE_ENV", false)
const DISTRIBUTED_QR = getb("DISTRIBUTED_QR", true)
const DATA_ROOT = get(ENV, "DATA_ROOT", joinpath(pkgdir(TeneT), "data"))
const OUTPUT_ROOT = get(ENV, "OUTPUT_ROOT", DATA_ROOT)

Random.seed!(SEED)
const atype = CuArray
const etype = Float64
const pattern = [1 3; 2 4]

function restriction_ipeps(A)
    Ar = Zygote.Buffer(A)
    Ar[:, :, :, :, :, 1] = A[:, :, :, :, :, 1]
    Ar[:, :, :, :, :, 1] += permutedims(Ar[:, :, :, :, :, 1], (4, 3, 2, 1, 5))
    Ar[:, :, :, :, :, 2] = permutedims(Ar[:, :, :, :, :, 1], (1, 4, 3, 2, 5))
    Ar[:, :, :, :, :, 3] = permutedims(Ar[:, :, :, :, :, 1], (3, 2, 1, 4, 5))
    Ar[:, :, :, :, :, 4] = permutedims(Ar[:, :, :, :, :, 1], (3, 4, 1, 2, 5))
    return copy(Ar)
end

function make_model(j2)
    J1J2(lattice=Square(), S=0.5, J1=1.0, J2=j2,
         ifrotate=true, couplingtype=:uniform, bondratio=1.0)
end

function make_boundary()
    alg = VUMPS{Plaquette{Square}}(
        ifsimple_eig=true,
        distributed_qr=DISTRIBUTED_QR,
        ifparallel=false,
        forloop_iter=FORLOOP_ITER,
        maxiter=VUMPS_MAXITER,
        miniter=0,
        maxiter_ad=0,
        miniter_ad=0,
        power_iter=POWER_ITER,
        power_iter_ad=5,
        power_iter_obs=POWER_ITER_OBS,
        show_every=SHOW_EVERY,
        tol=ENV_TOL,
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
        optimizer=LBFGS(200; maxiter=1, verbosity=0, gradtol=1e-7,
                        linesearch=HagerZhangLineSearch(maxfg=5)),
        forloop_iter=FORLOOP_ITER,
        maxiter_restart=1,
        verbosity=(RANK == 0 ? 4 : 0),
        folder=folder,
        ifSU=false,
        SUτ=0,
        ifprecondition=false,
        iter_precond=0,
        reuse_env=true,
        ifsave_env=SAVE_ENV,
        ifload_env=false,
        ifsave_lbfgs=false,
        ifload_lbfgs=false,
        save_every=0,
        ifplot=false,
    )
end

function maybe_reclaim()
    GC.gc(true)
    try
        CUDA.reclaim()
    catch
    end
end

function run_obs_case(A, params, j2, chi_obs)
    compute_xi = chi_obs in XI_CHIS
    say(@sprintf("=== CASE start J2=%.4g chi_obs=%d compute_xi=%s ===", j2, chi_obs, string(compute_xi)))
    MPI.Barrier(COMM)
    CUDA.synchronize()
    t0 = time()
    e, mag, xi = observable(A, chi_obs, params; restriction_ipeps,
                            cor_len_method=(compute_xi ? :mps : :none))
    CUDA.synchronize()
    MPI.Barrier(COMM)
    elapsed = time() - t0
    say(@sprintf("=== CASE done J2=%.4g chi_obs=%d in %.1f sec ===", j2, chi_obs, elapsed))
    say("energy[J2=$(j2),chi=$(chi_obs)] = $(e[1])")
    say("magnetization[J2=$(j2),chi=$(chi_obs)] = $(mag[1])")
    if xi === nothing
        say("correlation_length[J2=$(j2),chi=$(chi_obs)] = skipped")
    else
        say("correlation_length[J2=$(j2),chi=$(chi_obs)] = $xi")
    end
    maybe_reclaim()
end

function main()
    say("=== Cannon Plaquette J1J2 Square No.20 batch obs ===")
    say("nprocs=$NPROCS grid=$(NGRID)x$(NGRID) D=$D seed=$SEED")
    say("J2_LIST=$(J2_LIST) load chi=$CHI_LOAD No.$NO_LOAD obs chis=$(CHI_OBS_LIST) xi_chis=$(collect(XI_CHIS))")
    say("distributed_qr=$DISTRIBUTED_QR maxiter=$VUMPS_MAXITER maxiter_ad=0 miniter_ad=0")
    say("show_every=$SHOW_EVERY power_iter=$POWER_ITER power_iter_obs=$POWER_ITER_OBS tol=$ENV_TOL forloop=$FORLOOP_ITER save_env=$SAVE_ENV")
    RANK == 0 && CUDA.functional() && say("rank0 GPU=$(CUDA.name(CUDA.device())) CUDA_LAUNCH_BLOCKING=$(get(ENV, "CUDA_LAUNCH_BLOCKING", "unset"))")

    for j2 in J2_LIST
        model = make_model(j2)
        load_folder = joinpath(DATA_ROOT, "$model", "$pattern", "VUMPS_Plaquette", "Float64", "seed$SEED")
        folder = joinpath(OUTPUT_ROOT, "$model", "$pattern", "VUMPS_Plaquette", "Float64", "seed$SEED")
        say(@sprintf("=== LOAD start J2=%.4g from %s ===", j2, load_folder))
        params_load = make_params(model, load_folder)
        A = init_ipeps(; atype, etype, No=NO_LOAD, D, χ=CHI_LOAD, params=params_load)
        MPI.Barrier(COMM)
        CUDA.synchronize()
        say(@sprintf("=== LOAD done J2=%.4g chi=%d No.%d ===", j2, CHI_LOAD, NO_LOAD))

        for chi_obs in CHI_OBS_LIST
            params = make_params(model, folder)
            run_obs_case(A, params, j2, chi_obs)
        end
        maybe_reclaim()
    end
    say("=== batch obs done ===")
end

try
    main()
catch err
    RANK == 0 && (showerror(stdout, err, catch_backtrace()); println(); flush(stdout))
    MPI.Abort(COMM, 1)
finally
    MPI.Finalize()
end
