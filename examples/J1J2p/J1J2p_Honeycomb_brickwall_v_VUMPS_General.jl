using TeneT
using Random
using CUDA
using OptimKit
using LinearAlgebra
using Zygote

seed = 88
Random.seed!(seed)
atype = Array
etype = Float64
D, χ, χshift, maxiter_restart = 3, 8, 0, 100

# 6×2 pattern (column-major friendly): each unique site (1..6) appears twice,
# in a brickwall arrangement rotated 90° from the :brickwall_h pattern.
pattern = [1 4;
           2 5;
           3 6;
           4 1;
           5 2;
           6 3]

# J1J2p model on the vertical-orientation honeycomb brickwall.
# couplingtype=:plaquette is intentionally NOT used here — it's deferred until
# the (i,j)→bondratio mapping for the :brickwall_v pattern is derived.
# Use :uniform for now (Stage 1 benchmark).
model = J1J2p(lattice=Honeycomb(:brickwall_v),
              S=0.5, J1=1.0, J2p=0.3,
              ifrotate=false,
              couplingtype=:uniform, bondratio=1.0)
No = 0
folder = joinpath(pkgdir(TeneT), "data/$model/$pattern/VUMPS_General/$etype/seed$seed/")

boundary_alg = VUMPS{General}(ifupdown=true,
                              ifdownfromup=false,
                              ifsimple_eig=true,
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
                          optimizer=LBFGS(200; maxiter=10, verbosity=4, gradtol=1e-7, linesearch=HagerZhangLineSearch(maxfg=5)),
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

# Restriction: map the 6 unique sites to 2 independent tensors by parity.
# For :brickwall_v pattern [1 4; 2 5; 3 6; 4 1; 5 2; 6 3]:
#   Even-parity sites: 1, 3, 5 (positions (1,1),(3,1),(5,1) and (4,2),(6,2),(2,2))
#   Odd-parity sites:  2, 4, 6 (positions (2,1),(4,1),(6,1) and (5,2),(1,2),(3,2))
function restriction_ipeps(A)
    B = Zygote.Buffer(A)
    for i in 1:length(A)
        if i in [1, 3, 5]
            B[:,:,:,:,:,i] = A[:,:,:,:,:,1]
        elseif i in [2, 4, 6]
            B[:,:,:,:,:,i] = A[:,:,:,:,:,2]
        end
    end
    B = copy(B)
    return B / norm(B)
end

optimise_ipeps(A, 8, χshift, params; restriction_ipeps);
