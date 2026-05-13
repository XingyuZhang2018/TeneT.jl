using TeneT
using Random
using CUDA
using OptimKit
using LinearAlgebra
using Zygote

# One-sided VUMPS variant of the :brickwall_v J1J2p example. Uses
# `VUMPS{General}(; ifupdown=false)` so `init_env` returns a single
# VUMPSRuntime and `ObsEnv` produces an `OnesideVUMPSEnv`. The down
# environment is reconstructed from the up env via the model's
# `obs_index` trait. For `J1J2p{Honeycomb{:brickwall_v}}` under single-site
# restriction that trait is `obs_index(_, i, Ni) = i`, which requires each
# iPEPS tensor to be u-d self-symmetric. The `restriction_ipeps` below
# enforces that symmetry explicitly by symmetrizing A on the (d, u) legs
# before the parity mapping.
seed = 88
Random.seed!(seed)
atype = Array
etype = Float64
D, χ, χshift, maxiter_restart = 2, 16, 0, 100

# Minimal 2×2 brickwall pattern. Site 1 occupies positions (1,1) and (2,2)
# — both even parity; site 2 occupies (1,2) and (2,1) — both odd parity.
# `restriction_ipeps` therefore maps site 1 → tensor 1 (even) and site 2 →
# tensor 2 (odd).
pattern = [1 2;
           2 1]

# J1J2p model on the vertical-orientation honeycomb brickwall.
# couplingtype=:plaquette is intentionally NOT used here — it's deferred until
# the (i,j)→bondratio mapping for the :brickwall_v pattern is derived.
# Use :uniform for now.
model = J1J2p(lattice=Honeycomb(:brickwall_v),
              S=0.5, J1=1.0, J2p=0.5,
              ifrotate=false,
              couplingtype=:uniform, bondratio=1.0)
No = 0
folder = joinpath(pkgdir(TeneT), "data/$model/$pattern/VUMPS_Oneside/$etype/seed$seed/")

# One-sided VUMPS: only one fixed-point per column. We pass `ifupdown=false`
# and `ifparallelupdown=false` explicitly to disable the up-down dual VUMPS
# code path. `ObsEnv` will then build an `OnesideVUMPSEnv` (since
# `params.model` is forwarded from optimise_ipeps), wiring the model's
# `obs_index` trait into `leftenv`/`rightenv` for the obs envs.
boundary_alg = VUMPS{General}(; ifsimple_eig=true,
                                ifupdown=false,
                                ifparallelupdown=false,
                                ifparallel=false,
                                forloop_iter=1,
                                maxiter=30,
                                miniter=0,
                                maxiter_ad=4,
                                miniter_ad=4,
                                power_iter=1,
                                power_iter_ad=5,
                                power_iter_obs=40,
                                show_every=10,
                                tol=1e-10,
                                verbosity=3,
)
params = GradientOptimize(model=model,
                          pattern=pattern,
                          boundary_alg=boundary_alg,
                          optimizer=LBFGS(200; maxiter=100, verbosity=4, gradtol=1e-7, linesearch=HagerZhangLineSearch(maxfg=5)),
                          forloop_iter=1,
                          maxiter_restart=maxiter_restart,
                          verbosity=4,
                          folder=folder,
                          ifSU=false,
                          SUτ=0,
                          ifprecondition=true,
                          iter_precond=0,
                          reuse_env=true,
                          ifsave_env=true,
                          ifload_env=true,
                          ifsave_lbfgs=true,
                          ifload_lbfgs=false,
)
A = init_ipeps(; atype, etype, No, D, χ, params)

# Restriction: map the 2 unique sites in the pattern to 2 independent tensors
# by parity, AND enforce per-tensor U-D self-symmetry on the two
# representatives.
#
# For :brickwall_v pattern [1 2; 2 1]:
#   Even-parity site: 1 — positions (1,1) and (2,2)
#   Odd-parity site:  2 — positions (1,2) and (2,1)
#
# One-sided requirement: each iPEPS tensor must satisfy
# A[l,d,r,u,p] = A[l,u,r,d,p] (d ↔ u swap is a symmetry). This is what makes
# `obs_index = i` correct for J1J2p :brickwall_v under the one-sided env.
# We enforce it via `A + permutedims(A, (1,4,3,2,5))`, which projects onto
# the U-D-symmetric subspace (doubles the norm — divided out by `norm(B)`).
function restriction_ipeps(A)
    B = Zygote.Buffer(A)
    # Site 1 (even-parity representative): enforce U-D self-symmetry
    B[:,:,:,:,:,1] = A[:,:,:,:,:,1]
    B[:,:,:,:,:,1] += permutedims(B[:,:,:,:,:,1], (1,4,3,2,5))
    # Site 2 (odd-parity representative): enforce U-D self-symmetry
    B[:,:,:,:,:,2] = A[:,:,:,:,:,2]
    B[:,:,:,:,:,2] += permutedims(B[:,:,:,:,:,2], (1,4,3,2,5))
    B = copy(B)
    return B / norm(B)
end

optimise_ipeps(A, χ, χshift, params; restriction_ipeps);
