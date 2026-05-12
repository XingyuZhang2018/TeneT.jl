using TeneT
using Random
using OptimKit
using LinearAlgebra
using Zygote

# ============================================================================
# J1-J2p on the honeycomb lattice using the rotated-by-π/6 brickwall embedding.
#
# Compared with the standard `Honeycomb(:brickwall)` mode:
#   * The trivial (dim-1) leg of each iPEPS tensor is HORIZONTAL (L for
#     even-parity sites, R for odd-parity sites), not vertical.
#   * Every iPEPS row has equally non-trivial U and D bonds, so the up and
#     down environments of the VUMPS boundary are related by mirror reflection.
#     This lets us set `ifdownfromup=true` for ~2× speed-up.
#   * The pattern dimension that was "long" (Nj=6) in the original embedding
#     becomes "tall" (Ni=6) here — rows and columns swap roles.
# ============================================================================

seed = 88
Random.seed!(seed)
atype = Array
etype = Float64
D, χ, χshift, maxiter_restart = 3, 8, 0, 100

# 6×2 pattern (the 90°-rotated analog of the 2×6 used by the original brickwall).
pattern = [1 2;
           3 4;
           5 6;
           2 1;
           4 3;
           6 5]

model = J1J2p(lattice         = Honeycomb(:brickwall_pi6),
              S               = 0.5,
              J1              = 1.0,
              J2p             = 0.5,
              ifrotate        = false,
              couplingtype    = :plaquette,
              bondratio       = 0.1)

No = 0
folder = joinpath(pkgdir(TeneT), "data/$model/$pattern/VUMPS_General/$etype/seed$seed/")

# `ifdownfromup=true` exploits the up-down symmetry of the pi6 embedding.
boundary_alg = VUMPS{General}(ifupdown          = true,
                              ifdownfromup      = true,
                              ifsimple_eig      = true,
                              ifparallelupdown  = false,
                              ifparallel        = false,
                              ifcheckpoint      = false,
                              forloop_iter      = 1,
                              maxiter           = 30,
                              miniter           = 0,
                              maxiter_ad        = 4,
                              miniter_ad        = 4,
                              power_iter        = 1,
                              power_iter_ad     = 5,
                              power_iter_obs    = 40,
                              show_every        = 10,
                              tol               = 1e-10,
                              verbosity         = 3,
)
params = GradientOptimize(model              = model,
                          pattern            = pattern,
                          boundary_alg       = boundary_alg,
                          optimizer          = LBFGS(200; maxiter = 10,
                                                          verbosity = 4,
                                                          gradtol   = 1e-7,
                                                          linesearch = HagerZhangLineSearch(maxfg = 5)),
                          ifcheckpoint       = false,
                          forloop_iter       = 1,
                          maxiter_restart    = maxiter_restart,
                          verbosity          = 4,
                          folder             = folder,
                          ifSU               = false,
                          SUτ                = 0,
                          ifprecondition     = true,
                          iter_precond       = 0,
                          reuse_env          = true,
                          ifsave_env         = true,
                          ifload_env         = true,
                          ifsave_lbfgs       = true,
                          ifload_lbfgs       = false,
)

A = init_ipeps(; atype, etype, No, D, χ, params)

# Optional iPEPS restriction (set sites that should share the same tensor).
function restriction_ipeps(A)
    B = Zygote.Buffer(A)
    # Sites 1, 4, 5 share tensor 1; sites 2, 3, 6 share tensor 2 (90°-rotated
    # version of the original restriction).
    for i in 1:length(A)
        if i in [1, 4, 5]
            B[:, :, :, :, :, i] = A[:, :, :, :, :, 1]
        elseif i in [2, 3, 6]
            B[:, :, :, :, :, i] = A[:, :, :, :, :, 2]
        end
    end
    B = copy(B)
    return B / norm(B)
end

optimise_ipeps(A, 8, χshift, params; restriction_ipeps);
