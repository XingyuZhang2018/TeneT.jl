# LBFGS warmup: produce a converged iPEPS A (D=2, χ=8, rotated AFM Heisenberg)
# under the EXACT same params as the fixed-point framework's make_default_params.
# Output saved to data/lbfgs_warmup_D2_chi8.jld2 (key "A_raw").

using TeneT
using JLD2
using Random
using LinearAlgebra
using OptimKit

Random.seed!(42)

const D = 2
const χ = 8
const atype = Array
const etype = Float64

# Match make_default_params exactly, but enable LBFGS optimizer fields.
const boundary_alg = VUMPS{General}(maxiter=30, miniter=1, tol=1e-10,
                                    ifupdown=false,
                                    ifsimple_eig=true,
                                    ifparallel=false,
                                    forloop_iter=1,
                                    verbosity=0)

const model = Heisenberg(lattice=Square(), S=0.5,
                         Jx=1.0, Jy=1.0, Jz=1.0, ifrotate=true,
                         couplingtype=:uniform, bondratio=1.0)

const folder = joinpath(@__DIR__, "..", "data", "lbfgs_warmup")
mkpath(folder)

const params = GradientOptimize(
    model = model,
    pattern = ones(Int, 1, 1),
    boundary_alg = boundary_alg,
    optimizer = LBFGS(20; maxiter=40, verbosity=2, gradtol=1e-6,
                      linesearch=HagerZhangLineSearch(maxfg=5)),
    forloop_iter = 1,
    maxiter_restart = 1,
    verbosity = 3,
    folder = folder,
    ifSU = false,
    SUτ = 0,
    ifprecondition = false,
    iter_precond = 0,
    reuse_env = true,
    ifsave_env = false,
    ifload_env = false,
    ifsave_lbfgs = false,
    ifload_lbfgs = false,
)

println("="^60)
println("LBFGS warmup: D=$D, χ=$χ, model=$model")
println("Output folder: $folder")
println("="^60)

A = init_ipeps(; atype=atype, etype=etype, No=0, D=D, χ=χ, params=params)
println("Initial A shape: ", size(A))
println("Initial A norm:  ", norm(A))

function restriction_ipeps(A)
    A = local_min_norm(A, params)
    return A
end

const t0 = time()
optimise_ipeps(A, χ, 0, params; restriction_ipeps)
const t_total = time() - t0

println("="^60)
println("LBFGS warmup done in $(round(t_total, digits=1)) sec")
println("="^60)

# After optimise_ipeps writes data/D{D}/ipeps/χ{χ}/No.{N}.jld2 files,
# load the most recent and copy to a stable "warmup" path.
const ipeps_dir = joinpath(folder, "Heisenberg_Square(S=0.5,Jx=1.0,Jy=1.0,Jz=1.0,ifrotate=true,couplingtype=uniform)",
                            "[1;;]", "VUMPS_General", "$etype", "seed42",
                            "D$D", "ipeps", "χ$χ")
println("Looking for output ipeps in: $ipeps_dir")
if isdir(ipeps_dir)
    files = filter(f -> endswith(f, ".jld2"), readdir(ipeps_dir))
    isempty(files) && error("No .jld2 found in $ipeps_dir")
    nums = [parse(Int, replace(f, "No." => "", ".jld2" => "")) for f in files]
    last_no = maximum(nums)
    src = joinpath(ipeps_dir, "No.$(last_no).jld2")
    println("Final iPEPS: $src")

    A_final = load(src, "bcipeps"; iotype=IOStream)
    out_jld2 = joinpath(folder, "lbfgs_warmup_D$(D)_chi$(χ).jld2")
    save(out_jld2, "A_raw", A_final, "model", model, "D", D, "χ", χ)
    println("Saved to: $out_jld2")
else
    @warn "Output ipeps directory not found: $ipeps_dir — check params.folder"
end
