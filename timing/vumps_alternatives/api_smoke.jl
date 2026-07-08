# timing/vumps_alternatives/api_smoke.jl
# Verifies every TeneT internal used by this research dir. Run me after any rebase.
using TeneT, Random, KrylovKit, Zygote
for sym in (:FLmap_parallel, :ACmap_parallel, :Cmap, :ALCtoAC_map, :qrpos,
            :_to_front, :_c4v_local_tensor, :leftenv_c4v, :ACenv_c4v, :Cenv_c4v)
    @assert isdefined(TeneT, sym) "TeneT.$sym missing"
end
Random.seed!(42)
model = Ising(lattice=Square(), beta=0.43)
M0 = MPO(model, C4v; atype=Array)
M  = TeneT._c4v_local_tensor(M0)
# NOTE (shape correction, 2026-07-08): MPO(model, C4v) builds a RANK-5 tensor
# (4 virtual legs + trivial physical leg, size (2,2,2,2,1)); _c4v_local_tensor
# passes rank-5 through unchanged. Hence AL/FL/AC are RANK-4 (χ, 2, 2, χ),
# not the rank-3 (χ, 2, χ) one might expect from a bare rank-4 Ising tensor.
@assert ndims(M) == 5 && size(M,1) == 2
χ = 16
alg = VUMPS{C4v}(; ifsimple_eig=true, ifparallel=false, power_iter=5, maxiter=10,
                   maxiter_ad=1, miniter_ad=1, verbosity=0)
rt = init_env(M0, χ, alg)
@assert size(rt.AL) == (χ, 2, 2, χ) && size(rt.C) == (χ, χ) && size(rt.FL) == (χ, 2, 2, χ)
FL2 = TeneT.FLmap_parallel(rt.FL, rt.AL, conj(rt.AL), M; ifparallel=false, forloop_iter=1)
@assert size(FL2) == size(rt.FL)
AC  = TeneT.ALCtoAC_map(rt.AL, rt.C)
AC2 = TeneT.ACmap_parallel(AC, rt.FL, rt.FL, M; ifparallel=false, forloop_iter=1)
@assert size(AC2) == size(AC)
C2  = TeneT.Cmap(rt.C, rt.FL, rt.FL)
@assert size(C2) == size(rt.C)
println("API SMOKE OK")
