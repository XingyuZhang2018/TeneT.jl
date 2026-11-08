using TeneT
using CUDA
using JLD2
using LinearAlgebra
using Random

length(ARGS) == 6 ||
    error("Usage: julia J1J2p_tm_spectrum_sofia.jl CHECKPOINT RUN_ROOT CHI NLEVELS SECTOR SEED")

checkpoint = abspath(ARGS[1])
run_root = abspath(ARGS[2])
chi = parse(Int, ARGS[3])
nlevels = parse(Int, ARGS[4])
sector = ARGS[5]
seed = parse(Int, ARGS[6])

sector in ("trivial", "non-trivial") ||
    error("SECTOR must be trivial or non-trivial")
isfile(checkpoint) || error("Checkpoint does not exist: $checkpoint")

CUDA.device!(0)
CUDA.allowscalar(false)
Random.seed!(seed)

pattern = [1 2;
           2 1]
model = J1J2p(
    lattice=Honeycomb(:brickwall_v),
    S=0.5,
    J1=1.0,
    J2p=0.5,
    ifrotate=false,
    couplingtype=:uniform,
    bondratio=1.0,
)
boundary_alg = VUMPS{General}(
    ifsimple_eig=true,
    ifupdown=false,
    ifparallelupdown=false,
    ifparallel=false,
    forloop_iter=2,
    maxiter=100,
    miniter=1,
    power_iter=1,
    power_iter_obs=40,
    show_every=5,
    tol=1e-10,
    verbosity=3,
)
params = GradientOptimize(
    model=model,
    pattern=pattern,
    boundary_alg=boundary_alg,
    verbosity=3,
    folder=run_root,
    ifsave_env=true,
    ifload_env=true,
    ifsave_lbfgs=false,
    ifload_lbfgs=false,
    ifplot=false,
)

function oneside_restriction(A)
    B = similar(A)
    for site in axes(A, 6)
        Asite = @view A[:, :, :, :, :, site]
        Bsite = @view B[:, :, :, :, :, site]
        Bsite .= Asite .+ permutedims(Asite, (1, 4, 3, 2, 5))
    end
    return B / norm(B)
end

A = CuArray(JLD2.load(checkpoint, "bcipeps"))
A = oneside_restriction(A)
D = TeneT._ipeps_bond_dimension(A)
ifdomainwall = sector == "non-trivial"
result_dir = joinpath(run_root, "D$(D)", "TM_spectrum", sector)
mkpath(result_dir)

function spectrum_log(k_over_pi)
    filename = "k$(TeneT._tm_k_filename(k_over_pi)).log"
    return joinpath(run_root, "D$(D)", "TM_spectrum", sector, filename)
end

function read_completed_point(k_over_pi)
    path = spectrum_log(k_over_pi)
    isfile(path) || return nothing
    values = parse.(Float64, filter(line -> !isempty(line), readlines(path)))
    return length(values) == nlevels ? values : nothing
end

function write_outputs(results)
    ordered_k = sort(collect(keys(results)))
    csv_path = joinpath(result_dir, "spectrum.csv")
    open(csv_path, "w") do io
        println(io, "k_over_pi,k,sector,band,gap")
        for k_over_pi in ordered_k
            for (band, gap) in enumerate(results[k_over_pi])
                println(io, "$k_over_pi,$(k_over_pi * pi),$sector,$band,$gap")
            end
        end
    end

    plot_TM_spectrum(
        results;
        save_path=joinpath(result_dir, "spectrum.png"),
        xlimits=(-1, 1),
        xlabel="k / pi",
        ylabel="gap",
        title="J1-J2p honeycomb, J2p=0.5, D=$D, chi=$chi, $sector",
    )
    return nothing
end

grid = collect(-1.0:0.1:1.0)
compute_order = [0.0; filter(k -> !iszero(k), grid)]
results = Dict{Float64, Vector{Float64}}()

@info "TM spectrum run" checkpoint run_root chi nlevels sector
@info "CUDA device" CUDA.device() CUDA.name(CUDA.device())

for (point, k_over_pi) in enumerate(compute_order)
    completed = read_completed_point(k_over_pi)
    if completed === nothing
        Random.seed!(seed + 1000 * point)
        @info "Computing spectrum point" k_over_pi sector
        gaps = TM_spectrum(
            nlevels,
            k_over_pi,
            A,
            chi,
            params;
            restriction_ipeps=identity,
            ifdomainwall,
        )
        results[k_over_pi] = Array(real.(gaps))
    else
        @info "Reusing completed spectrum point" k_over_pi sector
        results[k_over_pi] = completed
    end
    write_outputs(results)
    CUDA.reclaim()
end

@info "TM spectrum complete" sector result_dir
