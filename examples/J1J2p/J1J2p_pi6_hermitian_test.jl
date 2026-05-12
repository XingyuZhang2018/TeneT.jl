using TeneT
using Random
using CUDA
using OptimKit
using LinearAlgebra
using Zygote

# ============================================================================
# Hermitian-VUMPS test for J1-J2' on the rotated-by-π/6 brickwall honeycomb
# embedding. The iPEPS tensor is constrained to be invariant under the D↔U
# leg swap, which makes every M-tensor mirror-symmetric in the column
# direction. Up and down VUMPS environments are then exactly related by
# reflection, so `ifdownfromup = true` is exact (no doubled boundary cost).
# ============================================================================

seed = 88
Random.seed!(seed)
atype = CuArray            # H200 GPU on Sofia
etype = Float64
D, χ = 3, 16

# Ni = 2, Nj = 2: minimum 4-site cell — 2 honeycomb unit cells. The pattern
# keeps each unique tensor on a single parity (i+j mod 2): tensors 1, 4 sit at
# even (i+j) sites (trivial L leg after lattice_map); tensors 2, 3 at odd
# (i+j) sites (trivial R leg). With Ni=2 the J2V bond (i, j) ↔ (i+2, j) loops
# back in unit-cell labels — that's a *formal* self-wrap, still a real NNN
# bond between two distinct physical sites; `contract_o3_V` evaluates it
# correctly with the same tensor at both endpoints.
pattern = [1 2;
           3 4]

model = J1J2p(lattice         = Honeycomb(:brickwall_pi6),
              S               = 0.5,
              J1              = 1.0,
              J2p             = 0.3,
              ifrotate        = false,
              couplingtype    = :uniform,
              bondratio       = 1.0)

No = 0
folder = joinpath(homedir(), "honeycomb/data/$model/$pattern/pi6_hermitian/seed$seed/")
mkpath(folder)

# ifdownfromup=true is exact under D↔U symmetric A.
boundary_alg = VUMPS{General}(ifupdown          = true,
                              ifdownfromup      = true,
                              ifsimple_eig      = true,
                              ifparallelupdown  = false,
                              ifparallel        = false,
                              forloop_iter      = 1,
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
                          optimizer          = LBFGS(20; maxiter = 8,
                                                         verbosity = 3,
                                                         gradtol   = 1e-6,
                                                         linesearch = HagerZhangLineSearch(maxfg = 5)),
                          forloop_iter       = 1,
                          maxiter_restart    = 100,
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

# --- Initial state: random + D↔U symmetrisation ---------------------------
A = init_ipeps(; atype=Array, etype, No, D, χ, params)
A = dumu_symmetrize(A)
A = atype(A)

# Sanity check: the random init is now exact under D↔U swap.
let
    A_h  = Array(A)
    A_sw = permutedims(A_h, (1, 4, 3, 2, 5, 6))
    rel  = norm(A_h - A_sw) / max(norm(A_h), eps(Float64))
    @info "initial D↔U asymmetry (should be ≈ 0)" rel
end

# --- Restriction: project to D↔U symmetric subspace each step -------------
function restriction_ipeps(A)
    return dumu_symmetrize(A)
end

@info "Starting optimisation" model pattern D χ atype
optimise_ipeps(A, χ, 0, params; restriction_ipeps);

# Final check after optimisation
A_final = init_ipeps(; atype=Array, etype, No=1, D, χ, params)
let
    A_h  = Array(A_final)
    A_sw = permutedims(A_h, (1, 4, 3, 2, 5, 6))
    rel  = norm(A_h - A_sw) / max(norm(A_h), eps(Float64))
    @info "final D↔U asymmetry after optimisation (should remain ≈ 0)" rel
end
