using TeneT
using Random
using CUDA
using OptimKit
using LinearAlgebra
using Zygote

# ============================================================================
# Hermitian-VUMPS pi6 test, parameterised by ARGS.
#   julia --project=. J1J2p_pi6_hermitian_DChi.jl  D  χ  J2p
# ============================================================================

length(ARGS) >= 3 || error("usage: J1J2p_pi6_hermitian_DChi.jl D χ J2p [No]")
D    = parse(Int,     ARGS[1])
χ    = parse(Int,     ARGS[2])
J2p_ = parse(Float64, ARGS[3])
No   = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 0   # 0 = random init, >0 = load checkpoint

seed = 88
Random.seed!(seed)
atype = CuArray
etype = Float64

# 4-site cell, Ni=2 Nj=2 — same as the D=3 verification run.
pattern = [1 2;
           3 4]

model = J1J2p(lattice         = Honeycomb(:brickwall_pi6),
              S               = 0.5,
              J1              = 1.0,
              J2p             = J2p_,
              ifrotate        = false,
              couplingtype    = :uniform,
              bondratio       = 1.0)

folder = joinpath(homedir(),
                  "honeycomb/data/$model/$pattern/pi6_hermitian/seed$seed/")
mkpath(folder)

# Hermitian VUMPS: down env mirrored from up env.
boundary_alg = VUMPS{General}(ifupdown          = true,
                              ifdownfromup      = true,
                              ifsimple_eig      = true,
                              ifparallelupdown  = false,
                              ifparallel        = false,
                              # Memory: split each *map contraction into 32 D-dim chunks,
                              # so peak GPU allocation in FLmap/FRmap/ACmap is ~D²/32.
                              # Trades a small amount of compute for ~32× lower peak.
                              forloop_iter      = 32,
                              maxiter           = 30,
                              miniter           = 0,
                              maxiter_ad        = 4,
                              miniter_ad        = 4,
                              power_iter        = 1,
                              power_iter_ad     = 5,
                              power_iter_obs    = 40,
                              show_every        = 5,
                              tol               = 1e-10,
                              verbosity         = 3,
)
params = GradientOptimize(model              = model,
                          pattern            = pattern,
                          boundary_alg       = boundary_alg,
                          optimizer          = LBFGS(20; maxiter = 4,
                                                         verbosity = 3,
                                                         gradtol   = 1e-6,
                                                         linesearch = HagerZhangLineSearch(maxfg = 5)),
                          forloop_iter       = 32,
                          maxiter_restart    = 10,
                          verbosity          = 3,
                          folder             = folder,
                          ifSU               = false,
                          SUτ                = 0,
                          ifprecondition     = true,
                          iter_precond       = 0,
                          reuse_env          = true,
                          ifsave_env         = false,
                          ifload_env         = false,
                          ifsave_lbfgs       = false,
                          ifload_lbfgs       = false,
)

# Random init (No=0) or load saved checkpoint (No>0).
# Quirk: optimise_ipeps saves files under `D = size(A,1)` which is 1 in pi6
# (left leg is the trivial dim), so the on-disk path is D1/ipeps/χN/. Pass
# D_load = 1 to init_ipeps when loading, regardless of the physical D.
D_load = No == 0 ? D : 1
A = init_ipeps(; atype=Array, etype, No, D=D_load, χ, params)
A = dumu_symmetrize(A)   # idempotent for already-symmetric tensors
A = atype(A)

let
    A_h  = Array(A)
    A_sw = permutedims(A_h, (1, 4, 3, 2, 5, 6))
    rel  = norm(A_h - A_sw) / max(norm(A_h), eps(Float64))
    @info "initial D↔U asymmetry (should be ≈ 0)" D χ J2p_ rel
end

function restriction_ipeps(A)
    return dumu_symmetrize(A)
end

@info "Starting optimisation" model pattern D χ atype
optimise_ipeps(A, χ, 0, params; restriction_ipeps);

# Final D↔U asymmetry check — scan saved checkpoints (D1/ipeps/χN/No.*.jld2)
# and load the highest-numbered one.
try
    ckpt_dir = joinpath(params.folder, "D1", "ipeps", "χ$(χ)")
    nos = Int[]
    if isdir(ckpt_dir)
        for f in readdir(ckpt_dir)
            m = match(r"^No\.(\d+)\.jld2$", f)
            m === nothing && continue
            push!(nos, parse(Int, m.captures[1]))
        end
        sort!(nos)
    end
    isempty(nos) && error("no saved checkpoints in $ckpt_dir")
    A_final = init_ipeps(; atype=Array, etype, No=last(nos), D=1, χ, params)
    A_h     = Array(A_final)
    A_sw    = permutedims(A_h, (1, 4, 3, 2, 5, 6))
    rel     = norm(A_h - A_sw) / max(norm(A_h), eps(Float64))
    @info "final D↔U asymmetry after optimisation (should remain ≈ 0)" D χ J2p_ last(nos) rel
catch e
    @warn "Could not load final ipeps for asymmetry check" exception=(e, catch_backtrace())
end
